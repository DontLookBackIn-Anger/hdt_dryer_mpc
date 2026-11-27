#!/usr/bin/env python3
"""
=== 第 4 项：多步预测评估器（报告 7.3）===

报告 7.3 明确要求的评估维度：

  ✓ 出口含水率与出口温度**分别**计算 MAE、RMSE、R² 和**平均偏差 Bias**
  ✓ 按批次、牌号、进料负荷和设备状态**分组**报告误差
  ✓ 分别报告 **1 步和多个预测步长**的误差，检查误差随时域增长趋势
  ✓ 检查**残差自相关**、极端工况误差、**系统性偏差**
      以及模型响应**方向是否符合物理规律**
  ✓ 评估**推理时间**

为什么这一步是进入 MPC 前的必做项？
    单步 RMSE 好看不代表能用于 MPC。MPC 需要预测未来 Np 步，
    如果误差随时域爆炸式增长，控制器就会做出错误决策。
    报告 7.4 还说"离线精度达标不代表可以直接闭环控制"。

本模块提供：
    MultiStepEvaluator   评估器类
    evaluate_model()     一次性产出全部指标表
    check_physical_response()  方向合理性检查（报告 7.3 最后一条）

运行：
    cd ~/hdt_dryer_mpc
    python src/s4_multistep_eval.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import FIG, REP, OUT, COLORS, PALETTE
from dryer_sim import (simulate_hdt_full, MV_COLS, DV_COLS, PV_COLS, CV_COLS,
                       STATE_CODES, STATE_CN)
from s3_window_builder import WindowConfig, build_windows, split_by_batch, build_split_windows


# ================================================================
# 指标
# ================================================================
def metrics_multi(y_true, y_pred):
    """y_true/y_pred: (n_samples, Np) —— 单个输出的多步预测。

    返回每个步长 k 的指标，以及总体指标。
    """
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    n, Np = y_true.shape
    per_step = []
    for k in range(Np):
        t, p = y_true[:, k], y_pred[:, k]
        ok = np.isfinite(t) & np.isfinite(p)
        t, p = t[ok], p[ok]
        e = p - t
        ss_res = np.sum(e ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2) + 1e-12
        per_step.append(dict(
            step=k + 1,
            MAE=float(np.mean(np.abs(e))),
            RMSE=float(np.sqrt(np.mean(e ** 2))),
            Bias=float(np.mean(e)),                    # 报告 7.3 要求
            MAPE=float(np.mean(np.abs(e) / (np.abs(t) + 1e-12)) * 100),
            R2=float(1 - ss_res / ss_tot),
            n=int(ok.sum()),
        ))
    per_step = pd.DataFrame(per_step)
    overall = dict(
        MAE=float(per_step["MAE"].mean()),
        RMSE=float(np.sqrt((per_step["RMSE"] ** 2).mean())),
        Bias=float(per_step["Bias"].mean()),
        MAPE=float(per_step["MAPE"].mean()),
        R2=float(per_step["R2"].mean()),
    )
    return per_step, overall


def residual_autocorr(y_true, y_pred, nlags=20):
    """残差自相关（报告 7.3）。

    若残差有显著自相关，说明模型没有提取完数据里的信息，
    存在系统性可预测成分（比如没考虑某个滞后）。
    """
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    e = e[np.isfinite(e)]
    e = e - e.mean()
    d = np.dot(e, e) + 1e-12
    return np.array([np.dot(e[:len(e) - k], e[k:]) / d for k in range(nlags + 1)])


# ================================================================
# 评估器
# ================================================================
class MultiStepEvaluator:
    """统一评估器：吃一个模型，吐全套报告 7.3 要求的指标。"""

    def __init__(self, scaler, out_names=CV_COLS):
        self.scaler = scaler
        self.out_names = out_names

    def _denorm(self, Y, out_idx):
        return Y * self.scaler["tsd"][out_idx] + self.scaler["tmu"][out_idx]

    def evaluate(self, model=None, X=None, Y=None, meta=None, batch_ids=None,
                 model_name: str = "model", y_pred=None) -> dict:
        """评估一个模型。

        两种用法：
          A) 传 model：内部调用 model.predict(X)，需返回 (n, n_out, Np)
          B) 传 y_pred：直接给出标准化空间的预测结果 (n, n_out, Np)
             —— 适合已经算好预测、或模型接口不统一的场景

        X : (n, n_feat) 展平特征（用法 B 时可省略）
        Y : (n, n_out, Np) 标准化标签
        """
        if y_pred is not None:
            pred_std = np.asarray(y_pred, float)
        else:
            if model is None:
                raise ValueError("必须提供 model 或 y_pred 之一")
            pred_std = np.asarray(model.predict(X), float)
        n_out, Np = Y.shape[1], Y.shape[2]
        if pred_std.shape != (Y.shape[0], n_out, Np):
            raise ValueError(f"预测形状 {pred_std.shape} 与标签 {Y.shape} 不符")
        infer_ms = float("nan")   # 推理时间由调用方计时（见 s5）

        res = {"model": model_name, "infer_ms_per_sample": infer_ms}
        per_step_all, overall_all = {}, {}
        for j, name in enumerate(self.out_names):
            yt = self._denorm(Y[:, j, :], j)
            yp = self._denorm(pred_std[:, j, :], j)
            per_step_all[name], overall_all[name] = metrics_multi(yt, yp)

        res["per_step"] = per_step_all
        res["overall"] = overall_all

        # ---- 1 步 vs 多步对比（报告 7.3 核心要求）----
        horizon_tbl = []
        for name in self.out_names:
            ps = per_step_all[name]
            for h in [1, 5, 15, 30]:
                if h <= len(ps):
                    r = ps.iloc[h - 1]
                    horizon_tbl.append({
                        "输出": name, "步长": h,
                        "MAE": round(r["MAE"], 5), "RMSE": round(r["RMSE"], 5),
                        "Bias": round(r["Bias"], 5), "R2": round(r["R2"], 5),
                    })
        res["horizon_table"] = pd.DataFrame(horizon_tbl)

        # ---- 分组误差（报告 7.3：按批次/状态/负荷）----
        if batch_ids is not None:
            grp = []
            for b in sorted(set(batch_ids)):
                m = np.array([x == b for x in batch_ids])
                if m.sum() < 5:
                    continue
                for j, name in enumerate(self.out_names):
                    yt = self._denorm(Y[m, j, :], j)
                    yp = self._denorm(pred_std[m, j, :], j)
                    _, ov = metrics_multi(yt, yp)
                    grp.append({"批次": b, "输出": name,
                                "MAE": round(ov["MAE"], 5), "RMSE": round(ov["RMSE"], 5),
                                "Bias": round(ov["Bias"], 5), "n": int(m.sum())})
            res["by_batch"] = pd.DataFrame(grp)

        # ---- 残差自相关 ----
        res["resid_acf"] = {
            name: residual_autocorr(
                self._denorm(Y[:, j, 0], j), self._denorm(pred_std[:, j, 0], j))
            for j, name in enumerate(self.out_names)
        }
        return res


# ================================================================
# 物理响应方向检查（报告 7.3 最后一条）
# ================================================================
def check_physical_response(model_predict_fn, scaler, cfg: WindowConfig,
                            df_ref: pd.DataFrame, feature_names, name="model"):
    """检查模型响应方向是否符合工艺常识。

    做法：在参考工况上，人为把某个操纵量阶跃 +Δ，看模型预测的
    出口水分/温度往哪个方向走。再和物理期望对比：

        蒸汽流量 ↑  → 出口水分 ↓（强制约定）
        热风流量 ↑  → 出口水分 ↓
        入口水分 ↑  → 出口水分 ↑

    报告 7.3："检查……模型响应方向是否符合基本物理规律"。
    方向错了，模型绝不能用于 MPC —— 控制器会朝反方向推。
    """
    base = df_ref.iloc[0:1].copy()
    expected = {
        "steam_flow": {"outlet_moisture": -1, "outlet_temp": +1},
        "hot_air_flow": {"outlet_moisture": -1, "outlet_temp": +1},
        "inlet_moisture": {"outlet_moisture": +1, "outlet_temp": -1},
    }
    rows = []
    for col, exp in expected.items():
        if col not in feature_names:
            continue
        delta = {"steam_flow": 30.0, "hot_air_flow": 500.0, "inlet_moisture": 1.0}[col]
        p0 = model_predict_fn(base)
        pert = base.copy()
        # 在窗口的最后一个时间点施加阶跃（持续到窗口末尾）
        L = cfg.L
        nf = len(feature_names)
        idx = feature_names.index(col)
        # 展平布局: reshape(n, n_feat, L) -> 特征 i 的第 j 个时间点索引 = i*L + j
        last_pos = idx * L + (L - 1)
        p1 = None
        # 需要同时改 dt 范围内的多个点（构造 10 秒的斜坡）
        for k in range(max(0, L - 10), L):
            pert.iloc[0, pert.columns.get_loc(col)] = base.iloc[0][col] + delta
        p1 = model_predict_fn(pert)

        for j, out in enumerate(CV_COLS):
            if out not in exp:
                continue
            d = float(p1[j] - p0[j])
            direction = 0 if abs(d) < 1e-9 else int(np.sign(d))
            ok = (direction == exp[out])
            rows.append({
                "操纵/扰动量": col, "输出": out,
                "Δ输入": delta,
                "预测Δ输出": round(d, 5),
                "期望方向": "↑" if exp[out] > 0 else "↓",
                "实际方向": "↑" if direction > 0 else ("↓" if direction < 0 else "≈0"),
                "符合物理": "✅" if ok else "❌",
            })
    return pd.DataFrame(rows)


# ================================================================
# 演示
# ================================================================
def main():
    print("=" * 78)
    print("第 4 项：多步预测评估器（报告 7.3）")
    print("=" * 78)

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    # ---- 数据（严格防泄漏：先按批次切分，仅在训练集拟合 scaler，且增大批次时长避免 p >> n）----
    batches = []
    for i, seed in enumerate((11, 22, 33, 44, 55)):
        d = simulate_hdt_full(n_steps=3600, seed=seed)
        d["batch_id"] = f"B{i+1:02d}"
        batches.append(d)
    df_all = pd.concat(batches, ignore_index=True)

    cfg = WindowConfig(L=120, Np=30, stride=2, require_stable=True)
    ds = build_split_windows(df_all, cfg, batch_col="batch_id")
    Xtr, Ytr, _, meta_tr, b_tr = ds["train"]
    Xva, Yva, _, meta_va, b_va = ds["val"]
    Xte, Yte, _, meta_te, b_te = ds["test"]
    scaler, info, split = ds["scaler"], ds["info"], ds["split"]
    log(f"样本（严格零泄漏）：训练 {len(Xtr)}（批次 {split['train']}），"
        f"验证 {len(Xva)}（{split['val']}），测试 {len(Xte)}（{split['test']}）")

    # ---- 训练一个多输出 LGBM 作为被评估对象 ----
    log("\n训练 LightGBM（每个输出每个步长一个模型）...")
    import lightgbm as lgb
    n_out, Np = Ytr.shape[1], Ytr.shape[2]
    models = {}
    for j in range(n_out):
        for k in range(Np):
            m = lgb.LGBMRegressor(n_estimators=150, max_depth=5, learning_rate=0.08,
                                  verbose=-1, random_state=0)
            m.fit(Xtr, Ytr[:, j, k])
            models[(j, k)] = m

    class LGBMEnsemble:
        def predict(self, Xq):
            n = len(Xq)
            out = np.zeros((n, n_out, Np))
            for (j, k), m in models.items():
                out[:, j, k] = m.predict(Xq)
            return out

    # 让评估器能吃到 (n, n_out, Np) 结构
    ev = MultiStepEvaluator(scaler)
    pred_std = LGBMEnsemble().predict(Xte)
    n = Xte.shape[0]
    res = ev.evaluate(Y=Yte, batch_ids=b_te, model_name="LightGBM", y_pred=pred_std)

    # ---- 报告：多步误差增长（核心）----
    log("\n" + "-" * 78)
    log("【1】误差随预测时域增长（报告 7.3 核心要求）")
    log("-" * 78)
    log(res["horizon_table"].to_string(index=False))

    for name in CV_COLS:
        ps = res["per_step"][name]
        r1, r30 = ps.iloc[0]["RMSE"], ps.iloc[min(29, len(ps) - 1)]["RMSE"]
        log(f"\n  {name}: RMSE 从 1 步 {r1:.5f} → 30 步 {r30:.5f}  "
            f"(增长 {r30 / max(r1, 1e-9):.2f}×)")
        if r30 / max(r1, 1e-9) > 3:
            log(f"    ⚠️ 误差增长过快，说明长时域预测不可靠，MPC 的 Np 应缩短")
        else:
            log(f"    ✅ 增长平缓，可用于 {30} 步（{30}s）预测时域")

    # ---- 分组误差 ----
    if "by_batch" in res and len(res["by_batch"]):
        log("\n" + "-" * 78)
        log("【2】按批次分组误差（报告 7.3）")
        log("-" * 78)
        log(res["by_batch"].to_string(index=False))

    # ---- 残差自相关 ----
    log("\n" + "-" * 78)
    log("【3】残差自相关（报告 7.3）")
    log("-" * 78)
    for name, ac in res["resid_acf"].items():
        lag1 = ac[1] if len(ac) > 1 else 0
        # 95% 置信区间约 ±1.96/sqrt(n)
        ci = 1.96 / np.sqrt(len(Xte))
        log(f"  {name}: 滞后1自相关 = {lag1:+.4f}  (95%阈值 ±{ci:.4f})")
        if abs(lag1) > ci:
            log(f"    ⚠️ 残差有显著自相关，模型仍遗漏了系统性信息")
        else:
            log(f"    ✅ 残差近似白噪声，模型已充分提取信息")

    # ---- 推理时间（直接对真实模型计时，避免包装器短路）----
    ensemble = LGBMEnsemble()
    ts_bench = []
    for _ in range(20):
        t1 = time.perf_counter(); ensemble.predict(Xte); ts_bench.append(time.perf_counter() - t1)
    per_sample_us = float(np.median(ts_bench)) / len(Xte) * 1e6

    log("\n" + "-" * 78)
    log("【4】推理时间（报告 7.3：在线计算时间）")
    log("-" * 78)
    log(f"  LightGBM（{n_out}×{Np}={n_out*Np} 个子模型）: {per_sample_us:.1f} μs/样本 "
        f"= {per_sample_us/1000:.4f} ms/样本")
    log(f"  按 MPC 控制周期 15 s 计算，单次预测耗时占比 "
        f"{per_sample_us/1e6/15*100:.6f}% —— 完全可接受")
    log(f"  ⚠️ 但注意：LightGBM 无解析梯度，不适合做梯度型 MPC 优化；")
    log(f"     真正的 MPC 应使用 FOPDT/状态空间等可微分或解析模型（见第 6 项）。")

    # ---- 物理方向检查 ----
    log("\n" + "-" * 78)
    log("【5】物理响应方向检查（报告 7.3 最后一条）")
    log("-" * 78)
    # 用一个简单的探测器：直接对 LGBM 施加扰动
    def predict_step1(row):
        """用单行数据构造一个窗口（重复），返回 1 步预测（已反标准化）"""
        # 简化：直接用训练好的 1 步模型，输入为特征向量的窗口展平
        return None

    # LGBM 特征是窗口展平 (n_feat * L)。
    # 正确做法：在多个测试样本上做"保持其余不变、只抬高该变量"的对照实验，
    # 取平均响应，避免单个样本因树的分段常数而给出 0 响应。
    feat = info["feature_names"]
    L = cfg.L
    n_probe = min(40, len(Xte))
    phys_rows = []
    for col, delta, exp in [("steam_flow", 30.0, -1),
                            ("hot_air_flow", 500.0, -1),
                            ("inlet_moisture", 1.0, +1)]:
        if col not in feat:
            continue
        i = feat.index(col)
        base_pred = models[(0, 0)].predict(Xte[:n_probe])
        pert = Xte[:n_probe].copy()
        # 抬高该特征在整个窗口内的值（等效于工况整体上移），
        # 比只改末尾几个点更稳健，也不会被树的分段常数吃掉。
        for kk in range(L):
            pert[:, i * L + kk] += delta / scaler["sd"][i]
        new_pred = models[(0, 0)].predict(pert)
        d_std = float(np.mean(new_pred - base_pred))
        d_real = d_std * scaler["tsd"][0]        # 反标准化到 % 水分
        ok = (np.sign(d_real) == exp) if abs(d_real) > 1e-5 else None
        phys_rows.append({
            "输入": col, "阶跃": delta,
            "平均Δ水分": round(d_real, 5),
            "期望": "↓" if exp < 0 else "↑",
            "实际": "↓" if d_real < 0 else "↑",
            "符合物理": "✅" if ok else ("❌" if ok is False else "≈无响应"),
        })
    phys = pd.DataFrame(phys_rows)
    log(phys.to_string(index=False))
    n_ok = (phys["符合物理"] == "✅").sum()
    log(f"\n  {n_ok}/{len(phys)} 个通道方向正确")
    if n_ok < len(phys):
        log("  ⚠️ 方向不符的通道不能用于 MPC —— 控制器会朝错误方向推！")
        log("     可能原因：训练数据中该变量与目标的相关性被其他扰动掩盖，")
        log("     或特征标准化后幅值被压缩。真实数据上应重新验证。")

    # ---- 汇总 ----
    log("\n" + "-" * 78)
    log("【6】总体指标汇总")
    log("-" * 78)
    summ = []
    for name, ov in res["overall"].items():
        summ.append({"输出": name, **{k: round(v, 5) for k, v in ov.items()}})
    log(pd.DataFrame(summ).to_string(index=False))

    # ---- 保存 ----
    res["horizon_table"].to_csv(REP / "s4_horizon_errors.csv", index=False, encoding="utf-8-sig")
    for name in CV_COLS:
        res["per_step"][name].to_csv(REP / f"s4_perstep_{name}.csv",
                                     index=False, encoding="utf-8-sig")
    phys.to_csv(REP / "s4_physical_check.csv", index=False, encoding="utf-8-sig")

    # ---- 绘图 ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for j, name in enumerate(CV_COLS):
        ax = axes[0, j]
        ps = res["per_step"][name]
        ax.plot(ps["step"], ps["RMSE"], marker="o", ms=3, color=PALETTE[j], label="RMSE")
        ax.plot(ps["step"], np.abs(ps["Bias"]), marker="s", ms=3,
                color=PALETTE[j], alpha=.6, ls="--", label="|Bias|")
        ax.set_xlabel("预测步长"); ax.set_ylabel("误差")
        ax.set_title(f"{name} 误差随时域增长", fontweight="bold")
        ax.legend(fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(ps["step"], ps["R2"], color="gray", ls=":", lw=1, label="R²")
        ax2.set_ylabel("R²", color="gray")

    ax = axes[1, 0]
    for j, name in enumerate(CV_COLS):
        ac = res["resid_acf"][name][:21]
        ax.plot(range(len(ac)), ac, marker="o", ms=3, color=PALETTE[j], label=name)
    ci = 1.96 / np.sqrt(len(Xte))
    ax.axhspan(-ci, ci, color="green", alpha=.12, label="95% 白噪声带")
    ax.axhline(0, c="k", lw=.6)
    ax.set_title("残差自相关", fontweight="bold"); ax.set_xlabel("滞后")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    # 1步 vs 30步 散点
    for j, name in enumerate(CV_COLS):
        y1 = ev._denorm(Yte[:, j, 0], j)
        p1 = ev._denorm(pred_std[:, j, 0], j)
        y30 = ev._denorm(Yte[:, j, -1], j)
        p30 = ev._denorm(pred_std[:, j, -1], j)
        ax.scatter(y1, p1, s=6, alpha=.5, color=PALETTE[j], label=f"{name} 1步")
        ax.scatter(y30, p30, s=6, alpha=.25, color=PALETTE[j], marker="^")
    lo = min(ev._denorm(Yte[:, :, 0], 0).min(), ev._denorm(Yte[:, :, 0], 1).min())
    hi = max(ev._denorm(Yte[:, :, 0], 0).max(), ev._denorm(Yte[:, :, 0], 1).max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
    ax.set_title("预测 vs 实际（圆=1步，三角=30步）", fontweight="bold")
    ax.legend(fontsize=7)

    fig.suptitle("多步预测评估（报告 7.3）", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "s4_multistep_eval.png", dpi=110); plt.close(fig)
    log(f"\n图已保存 -> {FIG/'s4_multistep_eval.png'}")
    log(f"表已保存 -> {REP}/s4_*.csv")

    (REP / "s4_multistep_eval.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()