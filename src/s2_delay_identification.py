#!/usr/bin/env python3
"""
=== 第 2 项：三层延迟辨识流水线（报告 6.2 / 11.2）===

报告把延迟拆成三层，本脚本逐层辨识并**在已知答案的仿真数据上验证方法正确性**：

    第一层  物料输送时间   —— 从进料到出口检测点（含管道+旋转阀+输送带）
    第二层  工艺动态时间   —— 热质传递的纯滞后 θ + 时间常数 τ（FOPDT 拟合）
    第三层  测量链路时间   —— 水分仪滤波、通讯、归档

为什么这一步最重要？
    报告 6.2 警告："若直接用同一行的蒸汽流量预测同一行出口水分，
    可能产生时间错配和虚假相关。"
    而互相关法在闭环控制下会失真（操纵量跟随水分变化），
    所以必须用多种方法交叉验证。

本脚本提供 4 种辨识方法并对比：
    ① 互相关（初筛）
    ② 阶跃响应拟合（最可靠，报告 6.2 推荐）
    ③ 系统辨识 ARX（sysidentpy / 最小二乘）
    ④ 递推最小二乘 RLS（在线辨识，MPC 可实时用）

验证方式：
    仿真器知道真实延迟（theta_steam≈14s, 以及可变延迟部分），
    我们把辨识结果和真值对比，误差应 < 2 秒。

运行：
    cd ~/hdt_dryer_mpc
    python src/s2_delay_identification.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal

from common import FIG, REP, COLORS, PALETTE
from dryer_sim import (simulate_hdt_full, STATE_CODES, stable_mask,
                       data_quality_summary)

# ---------------------------------------------------------------- 工具


def cross_correlation_lag(x, y, max_lag=120):
    """① 互相关法：返回使 |r| 最大的滞后（秒）。

    注意：闭环控制下操纵量会跟随误差变化，互相关的峰值可能失真，
    所以只作为初筛，不作为最终结论。
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    n = min(len(x), len(y))
    rs = []
    for k in range(max_lag):
        a, b = x[:n - k], y[k:n]
        rs.append(np.dot(a, b) / len(a))
    rs = np.array(rs)
    return int(np.argmax(np.abs(rs))), rs


def fopdt_response(t, K, tau, theta, du, y0):
    """FOPDT 阶跃响应解析解（t 以阶跃时刻为 0，单位秒）。"""
    t = np.asarray(t, float)
    y = np.full_like(t, y0, dtype=float)
    m = t > theta
    y[m] = y0 + K * du * (1.0 - np.exp(-(t[m] - theta) / tau))
    return y


def fit_fopdt_from_step(t, y, t_step, du, y0=None, max_window=120.0):
    """② 从阶跃响应拟合 K、τ、θ。

    Parameters
    ----------
    t        : 时间轴（秒）
    y        : 输出序列
    t_step   : 阶跃发生时刻（秒）
    du       : 阶跃幅值（操纵量变化量）
    y0       : 阶跃前稳态值（None 则自动取阶跃前均值）
    """
    t = np.asarray(t, float); y = np.asarray(y, float)
    if y0 is None:
        pre = y[(t < t_step) & (t > t_step - 60)]
        y0 = np.nanmean(pre)
    tr = t - t_step
    # 只取阶跃后 0~120 秒窗口
    # 自适应窗口：不能超过下一个阶跃事件，否则会被后续扰动污染。
    # 报告 6.2 强调"共同扰动和趋势会影响解释"，这就是典型表现。
    m = (tr >= 0) & (tr <= max_window) & np.isfinite(y)
    tr, yr = tr[m], y[m]
    if len(tr) < 30:
        return None
    try:
        popt, _ = curve_fit(
            lambda tt, K, tau, theta: fopdt_response(tt, K, tau, theta, du, y0),
            tr, yr, p0=[-0.02 * np.sign(du) if du else -0.02, 20.0, 12.0],
            bounds=([-10.0, 1.0, 0.0], [10.0, 300.0, 120.0]), maxfev=40000)
        K, tau, theta = popt
        resid = yr - fopdt_response(tr, K, tau, theta, du, y0)
        return dict(K=float(K), tau=float(tau), theta=float(theta),
                    rmse=float(np.sqrt(np.mean(resid ** 2))))
    except Exception:
        return None


def arx_least_squares(y, u, na=3, nb=3, delay=0):
    """③ ARX 模型最小二乘辨识：
        y(k) = Σ aᵢ y(k-i) + Σ bⱼ u(k-delay-j-1)

    返回 (a, b, 拟合优度)。b 的峰值位置 ≈ 主导滞后。
    """
    y = np.asarray(y, float); u = np.asarray(u, float)
    n = len(y)
    rows, Y = [], []
    for k in range(max(na, nb + delay) + 1, n):
        if not np.isfinite(y[k]):
            continue
        row = [y[k - i] for i in range(1, na + 1)]
        row += [u[k - delay - j - 1] for j in range(nb)]
        if not np.all(np.isfinite(row)):
            continue
        rows.append(row); Y.append(y[k])
    X = np.array(rows); Y = np.array(Y)
    if len(Y) < 50:
        return None
    th, *_ = np.linalg.lstsq(X, Y, rcond=None)
    a, b = th[:na], th[na:]
    pred = X @ th
    ss_res = np.sum((Y - pred) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2) + 1e-12
    return dict(a=float(a[0]), a_all=a, b_all=b, r2=float(1 - ss_res / ss_tot))


def rls_identify(y, u, na=2, nd=12, lam=0.999):
    """④ 递推最小二乘（RLS）：在线估计，MPC 可实时更新模型。

    把 u 从 t-1 到 t-nd 全部作为回归量，看不出"延迟"，
    但 b 系数的能量分布能反映滞后位置。每次迭代都是 O(n²)，适合在线。
    """
    y = np.asarray(y, float); u = np.asarray(u, float)
    n = len(y)
    start = max(na, nd) + 1
    p = na + nd
    theta = np.zeros(p)
    P = np.eye(p) * 1e4
    b_hist = []
    for k in range(start, n):
        phi = [y[k - i] for i in range(1, na + 1)] + [u[k - j - 1] for j in range(nd)]
        phi = np.array(phi)
        if not (np.all(np.isfinite(phi)) and np.isfinite(y[k])):
            continue
        # RLS 更新
        denom = lam + phi @ P @ phi
        K = (P @ phi) / denom
        theta = theta + K * (y[k] - phi @ theta)
        P = (P - np.outer(K, phi @ P)) / lam
        if k % 50 == 0:
            b_hist.append(theta[na:].copy())
    if not b_hist:
        return None
    b_final = np.mean(b_hist[-5:], axis=0)
    # 用 |b| 的质心估计滞后
    w = np.abs(b_final)
    lag_est = float(np.sum(np.arange(1, nd + 1) * w) / (w.sum() + 1e-12))
    return dict(lag_centroid=lag_est, b_profile=b_final)


# ---------------------------------------------------------------- 主流程


def main():
    print("=" * 76)
    print("第 2 项：三层延迟辨识流水线")
    print("=" * 76)

    # ---------- 数据：用无故障、固定延迟的数据做"已知答案"考试 ----------
    df_clean = simulate_hdt_full(n_steps=2400, seed=7,
                                 include_faults=False,
                                 variable_delay=False,
                                 include_states=False)
    t = df_clean["step"].values.astype(float)      # 秒
    y_true = df_clean["outlet_moisture_true"].values
    u_true = df_clean["steam_flow_true"].values
    y_obs = df_clean["outlet_moisture"].values
    REPORT_DELAY_STEAM = 14.0     # 仿真器设定：蒸汽→出口水分 纯滞后 14 秒

    print(f"\n数据: {len(df_clean)} 点, 1 Hz, 无故障, 固定延迟")
    print(f"仿真器真实蒸汽通道延迟 = {REPORT_DELAY_STEAM:.0f} 秒  ← 我们要找的就是它")

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    # ================================================================
    # 第一层：物料输送时间
    # ================================================================
    log("\n" + "-" * 76)
    log("第一层：物料输送时间（进料 → 出口检测点）")
    log("-" * 76)
    log("  方法：累计投料量 + 管道输送距离估算")
    log("  仿真设定：进料 3800 kg/h，输送管路约 3.5 秒停留")
    log("  说明：真实项目需用 §11.2 的批次追踪数据或各段用时来校正。")
    # 仿真里物料在途时间我们用 feed_rate 的累计量体现
    feed_mean = df_clean["feed_rate_true"].mean()
    log(f"  平均进料量 = {feed_mean:.0f} kg/h")
    log(f"  → 物料输送层估计 ≈ 3.5 s（仿真设定，真实需现场测量）")

    # ================================================================
    # 第二层：工艺动态时间（核心）
    # ================================================================
    log("\n" + "-" * 76)
    log("第二层：工艺动态时间（纯滞后 θ + 时间常数 τ）")
    log("-" * 76)

    # ① 互相关
    lag_cc, rs = cross_correlation_lag(u_true, y_true, max_lag=60)
    log(f"\n  ① 互相关法")
    log(f"     峰值滞后 = {lag_cc} s, |r| = {abs(rs[lag_cc]):.3f}")
    log(f"     ⚠️ 互相关把 θ 和 τ 混在一起，通常给出偏大的估计")

    # 找阶跃事件：优先用设定值 sp 的跳变点（物理上真正的阶跃开始），
    # 这是现场最可靠的事件来源。若无 sp（真实数据未必有），退化为
    # "滑窗中位数差分"检测。
    if "steam_flow_sp" in df_clean.columns:
        sp = df_clean["steam_flow_sp"].values
        dsp = np.diff(sp)
        idx = np.where(np.abs(dsp) > 5.0)[0]
        steps = []
        for i in idx:
            if not steps or i - steps[-1][-1] > 30:
                steps.append([i])
            else:
                steps[-1].append(i)
        step_starts = [int(g[0]) for g in steps]
        log("  事件来源：steam_flow_sp 设定值跳变（现场最可靠）")
    else:
        W = 25
        u_med = pd.Series(u_true).rolling(W, center=True, min_periods=W // 2).median().values
        jump = np.full(len(u_true), np.nan)
        jump[W: len(u_true) - W] = u_med[2 * W:] - u_med[:len(u_true) - 2 * W]
        valid = np.zeros(len(u_true), bool); valid[W: len(u_true) - W] = True
        cand = np.where(valid & (np.abs(np.nan_to_num(jump)) > 8.0))[0]
        steps = []
        for i in cand:
            if not steps or i - steps[-1][-1] > 60:
                steps.append([i])
            else:
                steps[-1].append(i)
        step_starts = [int(g[int(np.argmax(np.abs(np.nan_to_num(jump[np.array(g)]))))]) for g in steps]
        log("  事件来源：操纵量滑窗差分（无设定值时）")

    log(f"\n  检测到 {len(step_starts)} 个显著操纵量变化事件")

    # ② 阶跃响应拟合
    log(f"\n  ② 阶跃响应 FOPDT 拟合（报告 6.2 推荐）")
    fits = []
    for i_ev, st in enumerate(step_starts[:8]):
        pre = np.nanmedian(u_true[max(0, st - 30):st])
        post = np.nanmedian(u_true[st + 5:min(len(u_true), st + 35)])
        du = post - pre
        if abs(du) < 6:
            continue
        # 窗口截到下一个事件前 20 秒，避免串扰
        if i_ev + 1 < len(step_starts):
            gap = step_starts[i_ev + 1] - st - 20
        else:
            gap = 120
        win = float(np.clip(gap, 60, 120))
        r = fit_fopdt_from_step(t, y_true, float(st), du, max_window=win)
        if r:
            fits.append(r)
            log(f"     事件@{st:4d}s  Δu={du:+7.1f}  "
                f"K={r['K']:+.5f}  τ={r['tau']:5.1f}s  θ={r['theta']:5.1f}s  "
                f"(拟合RMSE={r['rmse']:.4f})")
    if fits:
        # ⚠️ 关键：不能对所有事件简单取中位数。
        # 报告 6.2 警告"共同扰动和趋势会影响辨识"。
        # 事件质量差别很大：与背景正弦波动叠加时拟合 RMSE 会明显变大。
        # 因此按拟合质量加权，只用高质量事件做最终估计。
        rmse_arr = np.array([f["rmse"] for f in fits])
        w = 1.0 / (rmse_arr ** 2 + 1e-6)
        w = w / w.sum()
        theta_w = float(np.sum(w * np.array([f["theta"] for f in fits])))
        tau_w = float(np.sum(w * np.array([f["tau"] for f in fits])))
        K_w = float(np.sum(w * np.array([f["K"] for f in fits])))

        good = [f for f in fits if f["rmse"] < 0.05]
        if good:
            theta_med = float(np.median([f["theta"] for f in good]))
            tau_med = float(np.median([f["tau"] for f in good]))
            K_med = float(np.median([f["K"] for f in good]))
            log(f"     ✓ 高质量事件(RMSE<0.05) {len(good)} 个，用它们定值")
        else:
            best = fits[int(np.argmin(rmse_arr))]
            theta_med, tau_med, K_med = best["theta"], best["tau"], best["K"]
            log(f"     ⚠️ 无高质量事件，取拟合最好的那个(RMSE={best['rmse']:.4f})")

        log(f"\n     → 加权参考值：θ = {theta_w:.1f} s, τ = {tau_w:.1f} s")
        log(f"     → 最终采用：θ = {theta_med:.1f} s, τ = {tau_med:.1f} s, K = {K_med:+.5f}")
        log(f"     → 与真值 {REPORT_DELAY_STEAM:.0f}s 相比，误差 {theta_med - REPORT_DELAY_STEAM:+.1f} s")
        log(f"     ⚠️ 注意：只有拟合 RMSE < 0.05 的事件才可信；")
        log(f"        其余事件受背景正弦波动污染，θ 会被系统性高估。")
        log(f"     ⚠️ τ 的可辨识性差：120s 窗口仅覆盖 {120/max(tau_med,1):.2f} 个 τ，")
        log(f"        而 τ 收敛需 3~5 倍时间常数。θ 比 τ 更可靠 —— 这也符合")
        log(f"        报告 6.2 的建议：先定滞后，再逐步放宽窗口估 τ。")
    else:
        theta_med = tau_med = K_med = float("nan")

    # ③ ARX
    log(f"\n  ③ ARX 最小二乘辨识")
    # nb 要覆盖到 ~40 秒滞后，否则延迟被截断
    arx = arx_least_squares(y_true, u_true, na=2, nb=40)
    if arx:
        b = np.abs(arx["b_all"])
        log(f"     a1={arx['a']:+.4f} (自回归), R²={arx['r2']:.4f}")
        # ⚠️ b 的峰值位置不等于纯滞后：因为 y 有强自回归(a1=0.73)，
        # 输入的影响被分散到多个滞后项上。正确做法是看 |b| 的加权质心。
        wsum = b.sum() + 1e-12
        centroid = float(np.sum(np.arange(1, len(b) + 1) * b) / wsum)
        log(f"     |b| 质心滞后 = {centroid:.1f} 步（比峰值更稳健）")
        # 找到 b 显著非零的起始位置
        thr = 0.25 * b.max()
        first_sig = int(np.argmax(b > thr)) + 1
        log(f"     |b| 首次显著(>25%峰值)于第 {first_sig} 步")
        log(f"     b 前12项: {np.round(arx['b_all'][:12], 5)}")

    # ④ RLS
    log(f"\n  ④ 递推最小二乘 RLS（可在线用）")
    rls = rls_identify(y_true, u_true, na=2, nd=20)
    if rls:
        log(f"     |b| 质心滞后 = {rls['lag_centroid']:.2f} 步")
        log(f"     ⚠️ RLS 质心会受多阶惯性影响，通常偏大，适合做在线监控而非精确标定")

    # ================================================================
    # 第三层：测量链路时间
    # ================================================================
    log("\n" + "-" * 76)
    log("第三层：测量链路时间（仪表滤波 + 通讯 + 归档）")
    log("-" * 76)
    # 用真实值 vs 观测值比较，识别仪表滤波时间常数
    ok = np.isfinite(y_obs) & np.isfinite(y_true)
    resid = y_obs[ok] - y_true[ok]
    log(f"  观测 - 真值 残差：均值={resid.mean():+.4f}, std={resid.std():.4f}")
    # 拟合仪表一阶滤波：用观测对真值做一阶跟踪
    a = 0.0
    best = None
    for alpha in np.arange(0.05, 0.95, 0.01):
        est = np.zeros(len(y_true))
        est[0] = y_true[0]
        for k in range(1, len(y_true)):
            est[k] = est[k-1] + alpha * (y_true[k] - est[k-1])
        e = np.nanmean((y_obs - est) ** 2)
        if best is None or e < best[1]:
            best = (alpha, e)
    alpha_best = best[0]
    tau_meas = (1 - alpha_best) / alpha_best * 1.0     # dt = 1s
    log(f"  仪表一阶滤波：α={alpha_best:.2f} → 时间常数 τ_meas ≈ {tau_meas:.1f} s")
    log(f"  （仿真设定 2.5 s，吻合则说明方法有效）")
    log(f"  另外需现场记录：通讯刷新周期、历史库归档周期（仿真中未建模）")

    # ================================================================
    # 汇总
    # ================================================================
    log("\n" + "=" * 76)
    log("三层延迟辨识汇总")
    log("=" * 76)
    log(f"  第一层  物料输送      ≈ 3.5 s   (仿真设定，真实现场测量)")
    log(f"  第二层  工艺纯滞后 θ  ≈ {theta_med:.1f} s   (真值 {REPORT_DELAY_STEAM:.0f} s)")
    log(f"          工艺时间常数 τ ≈ {tau_med:.1f} s")
    log(f"  第三层  测量滤波 τ    ≈ {tau_meas:.1f} s")
    total = 3.5 + theta_med + tau_meas
    log(f"\n  合成总滞后（到 63% 响应）≈ {total:.1f} s")
    log(f"  → MPC 的预测步长 Np 应 ≥ {total:.0f} 秒，窗口 L 建议 ≥ {max(60, total*3):.0f} 秒")

    # ================================================================
    # 验证：可变延迟能否被辨识出来
    # ================================================================
    log("\n" + "=" * 76)
    log("附加验证：可变延迟（报告 11.2）")
    log("=" * 76)
    df_var = simulate_hdt_full(n_steps=2400, seed=7, include_faults=False,
                               variable_delay=True, include_states=False)
    true_delay = df_var["delay_steam_true"].values
    log(f"  真值延迟范围: {true_delay.min():.1f} ~ {true_delay.max():.1f} s")
    log(f"  均值 {true_delay.mean():.2f} s, std {true_delay.std():.2f} s")
    log("  → 延迟随进料负荷变化，若用固定延迟建模，高/低负荷段都会有偏差。")
    log("    报告 11.2 建议：把进料量/累计投料量作为辅助特征，或分段辨识。")

    # ================================================================
    # 绘图
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    ax = axes[0, 0]
    lags = np.arange(len(rs))
    ax.plot(lags, rs, color=COLORS["target"], lw=1.5)
    ax.axvline(lag_cc, ls="--", c="k", label=f"互相关峰值 {lag_cc}s")
    ax.axvline(REPORT_DELAY_STEAM, ls=":", c="gray", label=f"真值 {REPORT_DELAY_STEAM:.0f}s")
    if fits:
        ax.axvline(theta_med, ls="-.", c=COLORS["good"], label=f"FOPDT θ={theta_med:.1f}s")
    ax.set_xlabel("滞后（秒）"); ax.set_ylabel("互相关 |r|")
    ax.set_title("① 互相关（初筛）", fontweight="bold"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    if fits:
        good_fits = [f for f in fits if f["rmse"] < 0.05]
        src = good_fits if good_fits else fits
        ax.hist([f["theta"] for f in src], bins=10, color=COLORS["pred"], alpha=.85)
        if good_fits:
            ax.set_title("② 阶跃拟合（仅高质量事件）", fontweight="bold")
        else:
            ax.set_title("② 阶跃拟合（全部事件，含污染）", fontweight="bold")
        ax.axvline(REPORT_DELAY_STEAM, ls=":", c="gray", lw=2, label=f"真值 {REPORT_DELAY_STEAM:.0f}s")
        ax.axvline(theta_med, ls="--", c=COLORS["good"], lw=2, label=f"辨识中位数 {theta_med:.1f}s")
        ax.set_xlabel("纯滞后 θ（秒）"); ax.set_ylabel("事件数")
        ax.set_title("② 阶跃拟合结果分布", fontweight="bold"); ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "无有效阶跃事件", ha="center", transform=ax.transAxes)

    ax = axes[1, 0]
    if arx:
        ax.bar(np.arange(1, len(arx["b_all"]) + 1), arx["b_all"], color=COLORS["good"])
        ax.axvline(REPORT_DELAY_STEAM, ls=":", c="gray", lw=2, label=f"真值 {REPORT_DELAY_STEAM:.0f}s")
        ax.set_xlabel("滞后（步）"); ax.set_ylabel("b 系数")
        ax.set_title(f"③ ARX 系数（R²={arx['r2']:.3f}）", fontweight="bold"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    # 展示一个阶跃的拟合效果
    if step_starts:
        st = step_starts[0]
        du = u_true[min(st + 5, len(u_true)-1)] - u_true[max(st - 5, 0)]
        gap = (step_starts[1] - st - 20) if len(step_starts) > 1 else 120
        r = fit_fopdt_from_step(t, y_true, float(st), du, max_window=float(np.clip(gap, 60, 240)))
        w = slice(max(0, st - 40), min(len(t), st + 130))
        ax.plot(t[w] - t[st], y_true[w], color=COLORS["target"], lw=1.5, label="实际（真值）")
        if r:
            tr = t[w] - t[st]
            y0 = np.nanmean(y_true[max(0, st-40):st])
            ax.plot(tr, fopdt_response(tr, r["K"], r["tau"], r["theta"], du, y0),
                    "--", color=COLORS["pred"], lw=1.8,
                    label=f"FOPDT θ={r['theta']:.1f} τ={r['tau']:.0f}")
        ax.set_xlabel("相对阶跃时刻（秒）"); ax.set_ylabel("出口水分 (%)")
        ax.set_title("④ 阶跃响应拟合示意", fontweight="bold"); ax.legend(fontsize=8)

    fig.suptitle("三层延迟辨识（报告 6.2 / 11.2）", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIG / "s2_delay_identification.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    log(f"\n图已保存 -> {out}")

    (REP / "s2_delay_identification.txt").write_text("\n".join(lines), encoding="utf-8")

    pd.DataFrame({"渠道": ["steam→outlet_moisture"], "θ_辨识(s)": [round(theta_med, 2)],
                  "τ_辨识(s)": [round(tau_med, 2)], "K_辨识": [round(K_med, 6)],
                  "θ_真值(s)": [REPORT_DELAY_STEAM],
                  "误差(s)": [round(theta_med - REPORT_DELAY_STEAM, 2)]}
                 ).to_csv(REP / "s2_delay_identification.csv", index=False, encoding="utf-8-sig")
    print(f"报告已保存 -> {REP/'s2_delay_identification.txt'}")


if __name__ == "__main__":
    main()