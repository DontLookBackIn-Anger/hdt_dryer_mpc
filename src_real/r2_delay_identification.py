#!/usr/bin/env python3
"""
=== 真实数据脚本 2：多方法纯滞后辨识（闭环工业数据版）===

与仿真脚本 src/s2_delay_identification.py 严格区分，针对真实加料回潮稳态数据
（已由 r1_clean_and_explore.py 剔除料头/料尾与时间断层）开展多方法交叉验证的滞后辨识：

  ① 原始互相关 vs 一阶差分互相关（预白化，削弱慢漂移与闭环反馈干扰）
  ② ARX 滞后扫描最小二乘（逐一扫描纯滞后 d = 0..90s，寻找 ΔR² 最优滞后）
  ③ 递推最小二乘 RLS 脉冲响应质心（在线估计多滞后权重分布）
  ④ MIMO 联合 FOPDT 变投影辨识（多输入同时拟合，消除输入共线性混叠）

运行：
    cd ~/hdt_dryer_mpc
    python3 src_real/r2_delay_identification.py

产出（全部独立保存在 outputs_real/）：
    outputs_real/figures/r2_1A_crosscorr.png
    outputs_real/figures/r2_1B_crosscorr.png
    outputs_real/figures/r2_transformer_crosscorr.png
    outputs_real/figures/r2_mimo_fopdt_fit.png
    outputs_real/reports/r2_delay_identification.txt
    outputs_real/reports/r2_delay_summary.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from real_common import (
    REAL_PROC, REAL_FIG, REAL_REP, COLORS, EN_LABELS,
    BATCH_MV_COLS, BATCH_DV_COLS, BATCH_AUX_COLS, BATCH_CV_COLS,
    TRANSFORMER_EXOG_COLS, TRANSFORMER_CV_COLS,
    calc_metrics, save_real_report,
)

print("=" * 72)
print("真实数据脚本 2：多方法纯滞后辨识（1A / 1B / transformer）")
print("=" * 72)


# ---------------------------------------------------------------- 辨识算法工具
def cross_corr_profile(x: np.ndarray, y: np.ndarray, max_lag: int = 120, diff: bool = False):
    """计算 x(t - k) 与 y(t) 的互相关曲线（k = 0..max_lag-1）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if diff:
        x = np.diff(x)
        y = np.diff(y)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    n = min(len(x), len(y))
    max_lag = min(max_lag, n // 4)
    rs = np.array([np.dot(x[: n - k], y[k:n]) / (n - k) for k in range(max_lag)])
    best_k = int(np.argmax(np.abs(rs)))
    return best_k, float(rs[best_k]), rs


def arx_delay_scan(y: np.ndarray, u: np.ndarray, max_delay: int = 90, na: int = 2, nb: int = 2):
    """扫描纯滞后 d = 0..max_delay，返回使去趋势 ARX 残差最小的 d 与 ΔR²。"""
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    # 去均值 + 标准化，避免截距主导
    y = (y - y.mean()) / (y.std() + 1e-12)
    u = (u - u.mean()) / (u.std() + 1e-12)
    n = len(y)
    start = max(na, nb + max_delay) + 1
    Y = y[start:]

    # 仅用自回归项作为基线
    X_ar = np.column_stack([y[start - i : n - i] for i in range(1, na + 1)])
    th_ar, *_ = np.linalg.lstsq(X_ar, Y, rcond=None)
    rss_ar = float(np.sum((Y - X_ar @ th_ar) ** 2))

    rss_list = []
    for d in range(max_delay + 1):
        X_u = np.column_stack([u[start - d - j - 1 : n - d - j - 1] for j in range(nb)])
        X = np.hstack([X_ar, X_u])
        th, *_ = np.linalg.lstsq(X, Y, rcond=None)
        rss = float(np.sum((Y - X @ th) ** 2))
        rss_list.append(rss)

    rss_arr = np.array(rss_list)
    best_d = int(np.argmin(rss_arr))
    rel_improve = float((rss_ar - rss_arr[best_d]) / (rss_ar + 1e-12)) * 100.0
    return best_d, rel_improve, rss_arr


def rls_lag_centroid(y: np.ndarray, u: np.ndarray, na: int = 2, nd: int = 80, lam: float = 0.998):
    """递推最小二乘 RLS：对滞后 1..nd 的输入估计脉冲响应 b_j，并用 |b_j| 质心估计主导滞后。"""
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    # 差分预白化后再跑 RLS，防止自回归强惯性掩盖输入系数
    dy = np.diff(y)
    du = np.diff(u)
    dy = (dy - dy.mean()) / (dy.std() + 1e-12)
    du = (du - du.mean()) / (du.std() + 1e-12)

    n = len(dy)
    start = max(na, nd) + 1
    p = na + nd
    theta = np.zeros(p)
    P = np.eye(p) * 100.0

    for k in range(start, n):
        phi = np.concatenate([dy[k - na : k][::-1], du[k - nd : k][::-1]])
        denom = lam + float(phi @ P @ phi)
        K = (P @ phi) / denom
        theta = theta + K * (dy[k] - float(phi @ theta))
        P = (P - np.outer(K, phi @ P)) / lam

    b = theta[na:]
    w = np.abs(b)
    centroid = float(np.sum(np.arange(1, nd + 1) * w) / (w.sum() + 1e-12))
    peak_lag = int(np.argmax(w) + 1)
    return peak_lag, centroid, b


def fopdt_filter(u: np.ndarray, tau: float, theta: float, dt: float = 1.0) -> np.ndarray:
    """离散一阶惯性 + 纯滞后滤波器（增益=1）。"""
    u = np.asarray(u, dtype=float)
    n = len(u)
    d = int(round(theta / dt))
    u_del = np.zeros(n)
    if d < n:
        u_del[d:] = u[: n - d]
    alpha = dt / (tau + dt)
    y = np.zeros(n)
    for k in range(1, n):
        y[k] = y[k - 1] + alpha * (u_del[k] - y[k - 1])
    return y


def identify_mimo_fopdt(df: pd.DataFrame, u_cols: list[str], y_col: str, max_theta: int = 80):
    """MIMO 联合 FOPDT 变投影辨识（Variable Projection）：同时拟合所有外生通道以消除共线性。"""
    U = np.column_stack([
        (df[c].to_numpy(dtype=float) - df[c].mean()) / (df[c].std() + 1e-12)
        for c in u_cols
    ])
    y_raw = df[y_col].to_numpy(dtype=float)
    y0 = float(y_raw.mean())
    y = y_raw - y0
    n_ch = len(u_cols)
    warmup = max_theta + 20

    tau_grid = np.array([5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0])
    theta_grid = np.arange(0, max_theta + 1, 2)

    thetas = np.full(n_ch, 15.0)
    best_tau = 30.0

    for _ in range(2):
        # 1) 固定 thetas 搜 tau
        best_loss = np.inf
        for tau in tau_grid:
            Phi = np.column_stack([fopdt_filter(U[:, i], tau, thetas[i]) for i in range(n_ch)])
            Phi_aug = np.column_stack([Phi, np.ones(len(y))])
            coef, *_ = np.linalg.lstsq(Phi_aug[warmup:], y[warmup:], rcond=None)
            loss = float(np.mean((y[warmup:] - Phi_aug[warmup:] @ coef) ** 2))
            if loss < best_loss:
                best_loss, best_tau = loss, float(tau)

        # 2) 固定 tau 逐通道搜 theta_i
        for ch in range(n_ch):
            best_th, best_l = thetas[ch], np.inf
            for th in theta_grid:
                th_try = thetas.copy()
                th_try[ch] = float(th)
                Phi = np.column_stack([fopdt_filter(U[:, i], best_tau, th_try[i]) for i in range(n_ch)])
                Phi_aug = np.column_stack([Phi, np.ones(len(y))])
                coef, *_ = np.linalg.lstsq(Phi_aug[warmup:], y[warmup:], rcond=None)
                loss = float(np.mean((y[warmup:] - Phi_aug[warmup:] @ coef) ** 2))
                if loss < best_l:
                    best_l, best_th = loss, float(th)
            thetas[ch] = best_th

    # 最终拟合
    Phi = np.column_stack([fopdt_filter(U[:, i], best_tau, thetas[i]) for i in range(n_ch)])
    Phi_aug = np.column_stack([Phi, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(Phi_aug[warmup:], y[warmup:], rcond=None)
    y_pred = y0 + Phi_aug @ coef
    metrics = calc_metrics(y_raw[warmup:], y_pred[warmup:])

    # 将标准化增益换算回物理单位增益 K_phys = K_std / std(u)
    k_phys = [float(coef[i] / (df[u_cols[i]].std() + 1e-12)) for i in range(n_ch)]
    return {
        "tau": best_tau,
        "thetas": thetas.tolist(),
        "k_std": coef[:n_ch].tolist(),
        "k_phys": k_phys,
        "y_pred": y_pred,
        "warmup": warmup,
        **metrics,
    }


# ---------------------------------------------------------------- 主流程
datasets = [
    ("1A", REAL_PROC / "1A_steady.csv", BATCH_MV_COLS + BATCH_DV_COLS + ["return_air_temp"], BATCH_CV_COLS),
    ("1B", REAL_PROC / "1B_steady.csv", BATCH_MV_COLS + BATCH_DV_COLS + ["return_air_temp"], BATCH_CV_COLS),
    ("transformer", REAL_PROC / "transformer_steady.csv", TRANSFORMER_EXOG_COLS, TRANSFORMER_CV_COLS),
]

lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


summary_records = []
mimo_plot_cache = []

for tag, path, exog_cols, cv_cols in datasets:
    if not path.exists():
        raise FileNotFoundError(f"未找到清洗后的稳态文件 {path}，请先运行 python3 src_real/r1_clean_and_explore.py")
    df = pd.read_csv(path)
    log(f"==================== 数据集: {tag} (连续稳态 {len(df)}s) ====================")

    # 画互相关图：2 行（出口含水率 / 出口温度） × len(exog_cols) 列
    fig, axes = plt.subplots(len(cv_cols), len(exog_cols), figsize=(3.4 * len(exog_cols), 5.8), sharex=True)
    if len(exog_cols) == 1:
        axes = axes[:, None]

    for r_i, cv in enumerate(cv_cols):
        y = df[cv].to_numpy(dtype=float)
        log(f"\n  [目标 CV: {cv} ({EN_LABELS.get(cv, cv)})]")
        log(f"  {'输入变量':18s} | {'原始CCF滞后(r)':18s} | {'差分CCF滞后(r)':18s} | {'ARX最优d(ΔRSS%)':18s} | {'RLS峰值/质心':14s}")
        log("  " + "-" * 96)

        for c_i, col in enumerate(exog_cols):
            u = df[col].to_numpy(dtype=float)
            k_raw, r_raw, prof_raw = cross_corr_profile(u, y, max_lag=120, diff=False)
            k_dif, r_dif, prof_dif = cross_corr_profile(u, y, max_lag=120, diff=True)
            d_arx, imp_arx, _ = arx_delay_scan(y, u, max_delay=90)
            p_rls, c_rls, _ = rls_lag_centroid(y, u, nd=80)

            log(
                f"  {col:18s} | {k_raw:3d}s ({r_raw:+.3f})     | {k_dif:3d}s ({r_dif:+.3f})     | "
                f"{d_arx:3d}s ({imp_arx:5.2f}%)    | {p_rls:2d}s / {c_rls:4.1f}s"
            )
            summary_records.append({
                "dataset": tag,
                "target": cv,
                "input": col,
                "ccf_raw_lag": k_raw,
                "ccf_raw_r": round(r_raw, 4),
                "ccf_diff_lag": k_dif,
                "ccf_diff_r": round(r_dif, 4),
                "arx_best_lag": d_arx,
                "arx_rss_improve_pct": round(imp_arx, 3),
                "rls_peak_lag": p_rls,
                "rls_centroid_lag": round(c_rls, 2),
            })

            ax = axes[r_i, c_i]
            lags = np.arange(len(prof_raw))
            ax.plot(lags, prof_raw, color=COLORS["pred"], lw=1.2, label=f"原始 ({k_raw}s)")
            ax.plot(lags, prof_dif, color=COLORS["target"], lw=1.0, alpha=0.85, label=f"差分 ({k_dif}s)")
            ax.axvline(k_raw, color=COLORS["pred"], ls="--", lw=0.8, alpha=0.7)
            ax.axvline(k_dif, color=COLORS["target"], ls=":", lw=0.9, alpha=0.8)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{col} → {cv}\nARX={d_arx}s", fontsize=8.5, fontweight="bold")
            ax.legend(fontsize=7, loc="best")
            if r_i == len(cv_cols) - 1:
                ax.set_xlabel("滞后 (秒)")
            if c_i == 0:
                ax.set_ylabel("互相关 r")

        # MIMO 联合 FOPDT 辨识
        mimo = identify_mimo_fopdt(df, exog_cols, cv, max_theta=80)
        mimo_plot_cache.append((tag, cv, df[cv].to_numpy(), mimo, exog_cols))
        th_str = ", ".join(f"{c}={int(th)}s" for c, th in zip(exog_cols, mimo["thetas"]))
        log(f"  → MIMO 联合 FOPDT: τ={mimo['tau']:.0f}s | RMSE={mimo['rmse']:.4f}, R²={mimo['r2']:.4f}")
        log(f"    各通道滞后 θ: {th_str}")

    fig.suptitle(f"[{tag}] 真实稳态数据：原始 vs 一阶差分互相关（纯滞后初筛）", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REAL_FIG / f"r2_{tag}_crosscorr.png", dpi=120)
    plt.close(fig)
    log()

# ---------------------------------------------------------------- 绘制 MIMO FOPDT 汇总图
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
for idx, (tag, cv, y_true, mimo, exog_cols) in enumerate(mimo_plot_cache):
    ax = axes[idx // 2, idx % 2]
    w = mimo["warmup"]
    t_ax = np.arange(len(y_true))
    ax.plot(t_ax[w:], y_true[w:], color=COLORS["target"], lw=1.1, label="真实值")
    ax.plot(t_ax[w:], mimo["y_pred"][w:], color=COLORS["pred"], lw=1.1, ls="--", label="MIMO FOPDT 拟合")
    ax.set_title(
        f"[{tag}] {cv} | τ={mimo['tau']:.0f}s, RMSE={mimo['rmse']:.4f}, R²={mimo['r2']:.3f}",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("稳态采样序号 (s)")

fig.suptitle("真实数据 MIMO 联合 FOPDT 辨识效果（开环纯外生输入拟合）", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(REAL_FIG / "r2_mimo_fopdt_fit.png", dpi=120)
plt.close(fig)

log("【辨识解读与建模建议】")
log("  1. 原始互相关 r 较高但滞后偏移大时，往往受慢漂移和 PID 闭环反馈影响；一阶差分互相关与 ARX 扫描更能反映瞬态因果响应。")
log("  2. MIMO FOPDT 仅用纯外生输入（不使用目标自身历史 y）做开环仿真拟合，其 R² 反映了当前可测 MV+DV 对目标波动的真实解释力；")
log("     若 R² 明显低于仿真数据，说明真实工况存在未测扰动（如来料品种变化、环境温湿度、蒸汽干度波动等），后续深度学习/MPC 需结合状态反馈（历史 y 或滚动误差校正）。")
log("  3. 综合三套数据辨识出的滞后范围，后续构建多步预测窗口时，建议回看窗口 input_size ≥ 60~90 秒，预测步长 horizon = 30~60 秒。")

pd.DataFrame(summary_records).to_csv(REAL_REP / "r2_delay_summary.csv", index=False)
save_real_report("r2_delay_identification.txt", "\n".join(lines))
print(f"图表已保存 -> {REAL_FIG}")
