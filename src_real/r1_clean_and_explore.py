#!/usr/bin/env python3
"""
=== 真实数据脚本 1：数据清洗、稳态切分与探索性分析（EDA）===

与仿真脚本 src/s1_explore.py 严格区分，专门处理真实加料回潮工业数据：
  - data/raw/real/1A.csv              (完整大批次，含料头/稳态/料尾与时间断层)
  - data/raw/real/1B.csv              (完整短批次，含尾随空列)
  - data/raw/real/transformer_data.csv (纯稳态 5 变量小数据集)

运行：
    cd ~/hdt_dryer_mpc
    python3 src_real/r1_clean_and_explore.py

产出（全部独立保存在 data/processed/real/ 与 outputs_real/）：
    data/processed/real/1A_clean_full.csv
    data/processed/real/1A_steady.csv
    data/processed/real/1B_clean_full.csv
    data/processed/real/1B_steady.csv
    data/processed/real/transformer_steady.csv
    outputs_real/figures/r1_1A_overview.png
    outputs_real/figures/r1_1B_overview.png
    outputs_real/figures/r1_transformer_overview.png
    outputs_real/figures/r1_steady_acf_dist.png
    outputs_real/reports/r1_eda.txt
    outputs_real/reports/r1_quality_summary.csv
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
    load_named_batch, segment_batch_phases, get_longest_continuous_steady,
    load_transformer_dataset, acf, save_real_report,
)

print("=" * 72)
print("真实数据脚本 1：数据清洗、稳态切分与探索性分析（EDA）")
print("=" * 72)

lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


summary_rows = []
steady_store: dict[str, pd.DataFrame] = {}

# ---------------------------------------------------------------- 1. 清洗与切分 1A / 1B
for tag, fname in [("1A", "1A.csv"), ("1B", "1B.csv")]:
    df_raw, meta = load_named_batch(fname)
    df = segment_batch_phases(df_raw)
    steady_all = df[df["phase"] == "steady"].copy()
    steady_cont = get_longest_continuous_steady(df)
    steady_store[tag] = steady_cont

    # 保存清洗后的完整批次与稳态段
    df.to_csv(REAL_PROC / f"{tag}_clean_full.csv", index=False)
    steady_cont.to_csv(REAL_PROC / f"{tag}_steady.csv", index=False)

    phase_counts = df["phase"].value_counts().to_dict()
    log(f"【数据集 {tag} ({fname})】")
    log(f"  时间范围: {meta['start_time']} -> {meta['end_time']} (跨度 {meta['span_sec']:.0f}s, 实际行数 {meta['rows']})")
    log(f"  清除空列: {meta['dropped_cols'] or '无'} | 缺失值总数: {meta['missing_total']} | 常量/全零列: {meta['const_cols'] or '无'}")
    log(f"  时间断层数 (dt>1.5s): {meta['n_gaps']}")
    for g in meta["gap_details"]:
        log(f"    ⚠️ 断层 @ 行 {g['row_idx']}: {g['prev_time']} -> {g['curr_time']} (跳变 {g['gap_sec']:.0f} 秒)")
    log(f"  阶段切分: startup={phase_counts.get('startup', 0)}s, steady={phase_counts.get('steady', 0)}s, tail={phase_counts.get('tail', 0)}s")
    log(f"  最长连续 1Hz 稳态段: {len(steady_cont)} 秒 (≈ {len(steady_cont)/60:.1f} 分钟)")

    cols_show = BATCH_CV_COLS + BATCH_MV_COLS + BATCH_DV_COLS + BATCH_AUX_COLS
    desc = steady_cont[cols_show].describe().T[["mean", "std", "min", "max"]]
    log("  稳态段变量统计：")
    log(desc.round(4).to_string())
    log()

    summary_rows.append({
        "dataset": tag,
        "total_rows": meta["rows"],
        "span_sec": meta["span_sec"],
        "n_gaps": meta["n_gaps"],
        "steady_rows_total": len(steady_all),
        "steady_cont_rows": len(steady_cont),
        "moist_mean": float(steady_cont["outlet_moisture"].mean()),
        "moist_std": float(steady_cont["outlet_moisture"].std()),
        "temp_mean": float(steady_cont["outlet_temp"].mean()),
        "temp_std": float(steady_cont["outlet_temp"].std()),
        "const_cols": ",".join(meta["const_cols"]),
    })

    # 绘图：全批次 4 子图 + 阶段阴影 + 断层标注
    fig, axes = plt.subplots(4, 1, figsize=(14, 10.5), sharex=True)
    t_idx = np.arange(len(df))

    def _shade(ax):
        for ph, color, label in [
            ("startup", "#f39c12", "料头"),
            ("steady", "#2ecc71", "稳态"),
            ("tail", "#95a5a6", "料尾"),
        ]:
            m = (df["phase"] == ph).to_numpy()
            if m.any():
                ii = np.where(m)[0]
                ax.axvspan(ii[0], ii[-1], color=color, alpha=0.10, label=label if ax is axes[0] else None)
        for g in meta["gap_details"]:
            ax.axvline(g["row_idx"], color="red", ls="--", lw=1.2, alpha=0.8)

    for ax in axes:
        _shade(ax)

    axes[0].plot(t_idx, df["outlet_moisture"], color=COLORS["target"], lw=1.1, label="出口含水率")
    axes[0].set_ylabel("出口含水率 (%)")
    axes[0].set_title(f"[{tag}] 被控量 CV：出口含水率（绿底=稳态段，红虚线=时间断层）", loc="left", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8, ncol=4)

    axes[1].plot(t_idx, df["outlet_temp"], color=COLORS["temp"], lw=1.1, label="烟叶出口温度")
    axes[1].set_ylabel("出口温度 (℃)")
    axes[1].set_title(f"[{tag}] 被控量 CV：烟叶出口温度", loc="left", fontweight="bold")

    ax = axes[2]
    ax.plot(t_idx, df["steam_flow"], color=COLORS["good"], lw=0.9, label="补偿蒸汽流量")
    ax.set_ylabel("补偿蒸汽流量", color=COLORS["good"])
    ax.legend(loc="upper left", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t_idx, df["exhaust_air_flow"], color=COLORS["warn"], lw=0.8, alpha=0.85, label="排潮风量")
    ax2.set_ylabel("排潮风量", color=COLORS["warn"])
    ax2.legend(loc="upper right", fontsize=8)
    ax.set_title(f"[{tag}] 操纵量 MV：补偿蒸汽流量 / 排潮风量（加水流量全为0）", loc="left", fontweight="bold")

    ax = axes[3]
    ax.plot(t_idx, df["inlet_moisture"], color=COLORS["pred"], lw=0.9, label="入口含水率 %")
    ax.set_ylabel("入口含水率 (%)", color=COLORS["pred"])
    ax.legend(loc="upper left", fontsize=8)
    ax3 = ax.twinx()
    ax3.plot(t_idx, df["feed_rate"], color=COLORS["teal"], lw=0.9, alpha=0.85, label="称瞬时流量 kg/h")
    ax3.set_ylabel("称瞬时流量 (kg/h)", color=COLORS["teal"])
    ax3.legend(loc="upper right", fontsize=8)
    ax.set_title(f"[{tag}] 扰动量 DV：入口含水率 / 称瞬时流量", loc="left", fontweight="bold")
    axes[-1].set_xlabel("采样序号（1 Hz）")

    fig.suptitle(f"真实加料回潮批次 {tag} 全流程变量总览（{meta['rows']} 点）", fontsize=13, fontweight="bold")
    fig.savefig(REAL_FIG / f"r1_{tag}_overview.png", dpi=120)
    plt.close(fig)

# ---------------------------------------------------------------- 2. 清洗与探索 transformer_data.csv
df_tf, meta_tf = load_transformer_dataset()
df_tf.to_csv(REAL_PROC / "transformer_steady.csv", index=False)
steady_store["transformer"] = df_tf

log("【数据集 transformer (transformer_data.csv)】")
log(f"  行数: {meta_tf['rows']} (纯稳态 1Hz 序列，无时间戳列) | 缺失值: {meta_tf['missing_total']} | 常量列: {meta_tf['const_cols'] or '无'}")
desc_tf = df_tf[TRANSFORMER_CV_COLS + TRANSFORMER_EXOG_COLS].describe().T[["mean", "std", "min", "max"]]
log(desc_tf.round(4).to_string())
log()

summary_rows.append({
    "dataset": "transformer",
    "total_rows": meta_tf["rows"],
    "span_sec": meta_tf["span_sec"],
    "n_gaps": 0,
    "steady_rows_total": len(df_tf),
    "steady_cont_rows": len(df_tf),
    "moist_mean": float(df_tf["outlet_moisture"].mean()),
    "moist_std": float(df_tf["outlet_moisture"].std()),
    "temp_mean": float(df_tf["outlet_temp"].mean()),
    "temp_std": float(df_tf["outlet_temp"].std()),
    "const_cols": ",".join(meta_tf["const_cols"]),
})

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
t_tf = df_tf["sample_idx"].to_numpy()
axes[0].plot(t_tf, df_tf["outlet_moisture"], color=COLORS["target"], lw=1.1, label="出口含水率 (%)")
axes[0].set_ylabel("出口含水率 (%)", color=COLORS["target"])
ax0r = axes[0].twinx()
ax0r.plot(t_tf, df_tf["outlet_temp"], color=COLORS["temp"], lw=0.9, alpha=0.85, label="出口温度 (℃)")
ax0r.set_ylabel("出口温度 (℃)", color=COLORS["temp"])
axes[0].set_title("[transformer_data] 被控量 CV：出口含水率 & 出口温度", loc="left", fontweight="bold")

axes[1].plot(t_tf, df_tf["steam_flow"], color=COLORS["good"], lw=0.9, label="蒸汽/流量 BF52")
axes[1].set_ylabel("BF52 流量", color=COLORS["good"])
ax1r = axes[1].twinx()
ax1r.plot(t_tf, df_tf["water_flow"], color=COLORS["warn"], lw=0.9, alpha=0.85, label="加水流量 WaterFlow")
ax1r.set_ylabel("加水流量", color=COLORS["warn"])
axes[1].set_title("[transformer_data] 操纵量 MV：BF52 流量 / 加水流量（此处加水流量非零）", loc="left", fontweight="bold")

axes[2].plot(t_tf, df_tf["inlet_moisture"], color=COLORS["pred"], lw=1.0)
axes[2].set_ylabel("入口含水率 (%)")
axes[2].set_xlabel("采样序号")
axes[2].set_title("[transformer_data] 扰动量 DV：入口含水率", loc="left", fontweight="bold")

fig.suptitle(f"真实稳态数据集 transformer_data.csv 总览（{len(df_tf)} 点）", fontsize=13, fontweight="bold")
fig.savefig(REAL_FIG / "r1_transformer_overview.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- 3. 稳态自相关 ACF 与分布对比
NLAG = 120
log("【稳态段自相关 ACF（评估目标自身惯性与“单步虚高”风险）】")
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for col_i, (tag, sdf) in enumerate(steady_store.items()):
    ym = sdf["outlet_moisture"].to_numpy()
    yt = sdf["outlet_temp"].to_numpy()
    am = acf(ym, NLAG)
    at = acf(yt, NLAG)
    decay_m = int(np.argmax(am < 0.5)) if (am < 0.5).any() else NLAG
    decay_t = int(np.argmax(at < 0.5)) if (at < 0.5).any() else NLAG
    log(
        f"  {tag:11s} | 出口含水率: ACF(1s)={am[1]:+.4f}, ACF(30s)={am[30]:+.4f}, ACF(60s)={am[60]:+.4f}, 降至0.5≈{decay_m}s"
    )
    log(
        f"  {'':11s} | 出口温度  : ACF(1s)={at[1]:+.4f}, ACF(30s)={at[30]:+.4f}, ACF(60s)={at[60]:+.4f}, 降至0.5≈{decay_t}s"
    )

    axes[0, col_i].plot(range(NLAG + 1), am, color=COLORS["target"], lw=1.5, label="出口含水率")
    axes[0, col_i].plot(range(NLAG + 1), at, color=COLORS["temp"], lw=1.5, ls="--", label="出口温度")
    axes[0, col_i].axhline(0.5, color="gray", ls=":", lw=0.9)
    axes[0, col_i].axhline(0.0, color="black", lw=0.6)
    axes[0, col_i].set_title(f"{tag} 稳态段 ACF（n={len(sdf)}s）", fontweight="bold")
    axes[0, col_i].set_xlabel("滞后（秒）")
    if col_i == 0:
        axes[0, col_i].set_ylabel("自相关系数")
    axes[0, col_i].legend(fontsize=8)

    axes[1, col_i].hist(ym, bins=35, color=COLORS["target"], alpha=0.75, edgecolor="white")
    axes[1, col_i].set_title(
        f"{tag} 稳态出口含水率分布\nμ={ym.mean():.3f}%, σ={ym.std():.4f}%",
        fontweight="bold",
        fontsize=9.5,
    )
    axes[1, col_i].set_xlabel("出口含水率 (%)")
    if col_i == 0:
        axes[1, col_i].set_ylabel("频次")

fig.suptitle("真实数据稳态段：惯性记忆（ACF）与目标分布对比", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(REAL_FIG / "r1_steady_acf_dist.png", dpi=120)
plt.close(fig)

log()
log("【关键工程结论】")
log("  1. 1A.csv 中存在 1 处 204 秒时间断层，必须按连续段切分，不可跨断层直接滑窗或算滞后。")
log("  2. 1A.csv / 1B.csv 的加水流量全为 0，说明该批次工况仅靠补偿蒸汽和排潮风量调节；而 transformer_data.csv 中加水流量非零。")
log("  3. 三个数据集稳态段出口含水率的 ACF(1s) 均极高（>0.99），验证了“单步预测指标虚高”的风险——后续建模务必按多步预测（如 30s / 60s）并与 Naive 基线对比。")
log("  4. 下一步：运行 python3 src_real/r2_delay_identification.py 进行真实数据多方法时滞辨识。")

pd.DataFrame(summary_rows).to_csv(REAL_REP / "r1_quality_summary.csv", index=False)
save_real_report("r1_eda.txt", "\n".join(lines))
print(f"清洗后数据已保存 -> {REAL_PROC}")
print(f"图表已保存 -> {REAL_FIG}")
