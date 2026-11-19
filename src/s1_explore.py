#!/usr/bin/env python3
"""
=== 学习脚本 1 / 6：看懂你的数据（EDA） ===

在学习任何模型之前，先把数据摸清楚。这一步决定了后面所有建模决策：
  - 要预测多远？（预测步长 h）
  - 模型要看多久的历史？（回看窗口 input_size）
  - 哪些变量真的有用？
  - 工艺滞后有多长？

运行：
    cd ~/hdt_dryer_mpc
    /opt/conda_envs/dryer-ts/bin/python src/s1_explore.py

产出：
    outputs/figures/s1_overview.png     时序总览
    outputs/figures/s1_acf.png          自相关（看"记忆"）
    outputs/figures/s1_crosscorr.png    互相关（看"滞后"）
    outputs/reports/s1_eda.txt          文字结论
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import (simulate_hdt_batch, FIG, REP, PALETTE, COLORS,
                   EXOG_COLS, CV_COLS, save_report)

print("=" * 72)
print("学习脚本 1：数据探索")
print("=" * 72)

# ---------------------------------------------------------------- 1. 取数据
df = simulate_hdt_batch(n_steps=1200, seed=42)
df.to_csv(Path(REP).parent.parent / "data" / "processed" / "demo_batch.csv", index=False)

lines = []
def log(s=""):
    print(s); lines.append(s)

log(f"样本数: {len(df)} | 采样频率: 1 秒 | 单批时长: {len(df)} 秒 ≈ {len(df)/60:.0f} 分钟")
log(f"目标变量: {CV_COLS}")
log(f"输入变量: 外生 {EXOG_COLS}")
log()

# ---------------------------------------------------------------- 2. 基本统计
log("【1】变量统计")
desc = df[CV_COLS + EXOG_COLS].describe().T[["mean", "std", "min", "max"]]
log(desc.round(4).to_string())
log()

# ---------------------------------------------------------------- 3. 自相关
# ACF 回答："过去的值对现在还有多大影响？"
# 若 ACF 在 k 步后仍有 0.9，说明目标高度平滑（一阶惯性），模型可以靠"保持"蒙对
def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    denom = np.dot(x, x) + 1e-12
    return np.array([np.dot(x[:len(x) - k], x[k:]) / denom for k in range(nlags + 1)])

NLAG = 120
y = df["outlet_moisture"].to_numpy()
a = acf(y, NLAG)

log("【2】出口水分的自相关 ACF（看目标自身的「记忆」有多长）")
for k in [1, 5, 10, 20, 30, 60, 90, 120]:
    log(f"  ACF(k={k:3d}步 = {k:3d}秒) = {a[k]:+.4f}")
decay = int(np.argmax(a < 0.5)) if (a < 0.5).any() else NLAG
log(f"  → ACF 降到 0.5 约需 {decay} 秒，说明目标单靠自身历史就能预测一部分")
log("  ⚠️ 这正是「预测精度虚高」的陷阱：目标惯性大，简单「保持上一步」就有不错 RMSE。")
log("     所以评估时务必和 Naive（直接复制上一步）对比，否则会被误导。")
log()

# ---------------------------------------------------------------- 4. 互相关
# 互相关回答："输入 x 在多少步之前影响了输出 y？" → 峰值位置 ≈ 纯滞后
log("【3】输入 → 出口水分 的互相关（峰值位置 = 工艺纯滞后）")
best = {}
for c in EXOG_COLS:
    x = df[c].to_numpy()
    xc, yc = (x - x.mean()) / (x.std() + 1e-12), (y - y.mean()) / (y.std() + 1e-12)
    cc = np.array([np.corrcoef(xc[:len(xc) - k], yc[k:])[0, 1] for k in range(0, 61)])
    k = int(np.argmax(np.abs(cc)))
    best[c] = (k, cc[k])
    log(f"  {c:16s}: 同期 r={cc[0]:+.3f} | 峰值滞后 {k:2d} 秒, r={cc[k]:+.3f}")
log()
lags = [v[0] for v in best.values() if v[0] > 0]
log(f"  → 观测到的最大滞后约 {max(lags) if lags else 0} 秒。")
log("     这意味着模型回看窗口 input_size 至少要 ≥ 这个滞后，否则根本「看不到」激励。")
log()

# ---------------------------------------------------------------- 5. 画图
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
axes[0].plot(df.timestamp, df.outlet_moisture, color=COLORS["target"], lw=1.2)
axes[0].set_ylabel("出口含水率 (%)"); axes[0].set_title("被控量 CV：出口含水率", loc="left", fontweight="bold")
axes[1].plot(df.timestamp, df.outlet_temp, color="#8e44ad", lw=1.2)
axes[1].set_ylabel("出口温度 (℃)"); axes[1].set_title("被控量 CV：出口温度", loc="left", fontweight="bold")
ax = axes[2]
ax.plot(df.timestamp, df.steam_flow, color="#27ae60", lw=1.0, label="蒸汽流量 kg/h")
ax.set_ylabel("蒸汽流量", color="#27ae60"); ax.legend(loc="upper right", fontsize=8)
ax2 = ax.twinx(); ax2.plot(df.timestamp, df.hot_air_flow / 100, color="#e67e22", lw=1.0)
ax2.set_ylabel("热风流量 (÷100)", color="#e67e22")
ax.set_title("操纵量 MV：蒸汽 / 热风", loc="left", fontweight="bold")
ax = axes[3]
ax.plot(df.timestamp, df.inlet_moisture, color="#2980b9", lw=1.0, label="入口水分 %")
ax.set_ylabel("入口水分", color="#2980b9"); ax.legend(loc="upper right", fontsize=8)
ax3 = ax.twinx(); ax3.plot(df.timestamp, df.feed_rate / 100, color="#16a085", lw=1.0)
ax3.set_ylabel("物料流量 (÷100)", color="#16a085")
ax.set_title("扰动量 DV：入口水分 / 物料流量", loc="left", fontweight="bold")
axes[-1].set_xlabel("时间")
fig.suptitle("单批次 1200 秒 工艺变量总览", fontsize=13, fontweight="bold")
fig.savefig(FIG / "s1_overview.png", dpi=110); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
axes[0].bar(range(NLAG + 1), a, color="#2980b9")
axes[0].axhline(0, c="k", lw=.6); axes[0].axhline(0.5, ls="--", c="gray", lw=.8)
axes[0].set_title("出口水分 自相关 ACF", fontweight="bold"); axes[0].set_xlabel("滞后（秒）")
axes[1].hist(df.outlet_moisture, bins=40, color=COLORS["target"], alpha=.8, edgecolor="white")
axes[1].set_title("出口水分 分布", fontweight="bold"); axes[1].set_xlabel("%")
fig.savefig(FIG / "s1_acf.png", dpi=110); plt.close(fig)

fig, axes = plt.subplots(1, len(EXOG_COLS), figsize=(15, 3.2))
for ax, c in zip(axes, EXOG_COLS):
    x = df[c].to_numpy()
    xc, yc = (x - x.mean()) / (x.std() + 1e-12), (y - y.mean()) / (y.std() + 1e-12)
    cc = [np.corrcoef(xc[:len(xc) - k], yc[k:])[0, 1] for k in range(61)]
    ax.plot(range(61), cc, marker="o", ms=2.5, color=COLORS["target"])
    ax.axvline(best[c][0], ls="--", c="gray", lw=.9); ax.axhline(0, c="k", lw=.6)
    ax.set_title(f"{c}\n峰值滞后 {best[c][0]}s", fontweight="bold", fontsize=9)
    ax.set_xlabel("滞后（秒）")
axes[0].set_ylabel("互相关 r")
fig.suptitle("输入 → 出口水分 互相关（找纯滞后）", fontsize=12, fontweight="bold")
fig.savefig(FIG / "s1_crosscorr.png", dpi=110); plt.close(fig)

save_report("s1_eda.txt", "\n".join(lines))
print(f"\n图已保存 -> {FIG}")
log()
log("【下一步】运行 src/s2_baseline.py —— 先建立「最简单的预测能做到多好」的基线。")