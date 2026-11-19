#!/usr/bin/env python3
"""
=== 学习脚本 3 / 6：深度模型对比 ===

⚠️ 先说结论（跑了才知道的真实情况）：
    Persistence（复制上一步）RMSE ≈ 0.0147，
    连训练充分的 TimeXer 也只有 0.064，仍然打不过它。
    原因见下方"为什么深度模型打不过基线"。

本脚本对比 4 个模型，口径统一为【对测试批次逐步预测下一秒】：
    ① Persistence       —— 复制上一步，下限
    ② TimeXer           —— 外生变量 token + 内因 patch token，最适合本场景
    ③ iTransformer      —— 每个变量一个 token 的注意力
    ④ TCN (Darts)       —— 因果膨胀卷积，MPC 内部模型的优选

关于 NeuralForecast 的几个坑（踩过，记下来）：
  · nf.predict(df=...) 只给 h 个点。拿它和 1200 点测试集比是错位的，
    会得到荒谬的负 R²（我第一版就踩了这个坑，R²=-2.29）。
  · nf.cross_validation(refit=False) 与 input_size=120 冲突：
    每个窗口只有 h 点，短于 input_size，会报 "Time series is too short"。
  · 所以这里采用【手动滚动】：每步把测试序列截到当前时刻再 predict。
    慢一些（~100s/模型），但结果可靠。
  · 训练步数要足够。60 步远不够（RMSE 0.38），2000 步才收敛。

运行（约 4 分钟，用 GPU）：
    cd ~/hdt_dryer_mpc
    conda activate dryer-ts
    python src/s3_dl_forecast.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import warnings
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from common import (simulate_hdt_batch, calc_metrics, FIG, REP,
                    COLORS, PALETTE)

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

H = 1                       # 预测下一秒（最能体现模型能力）
INPUT_SIZE = 120            # 回看 120 秒（> s1 观测到的 60s 滞后）
MAX_STEPS = 2000            # 训练步数，少了根本不收敛
TRAIN_SEEDS = (10, 20, 30)
TEST_SEED = 99
EXOG = ["steam_flow", "hot_air_flow", "inlet_moisture", "feed_rate"]
# 逐步预测 1080 次很慢，这里每 3 秒预测一次，仍能覆盖全批
TEST_STRIDE = 3

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

print("=" * 78)
print(f"学习脚本 3：深度模型（H={H}s, input_size={INPUT_SIZE}s, steps={MAX_STEPS}, {DEVICE}）")
print("=" * 78)

test_df = simulate_hdt_batch(1200, seed=TEST_SEED)
y_test = test_df["outlet_moisture"].to_numpy(dtype=float)
rows, curves = [], {}

# ---------------------------------------------------------------- ① Persistence
persist_idx = range(INPUT_SIZE, len(y_test), TEST_STRIDE)
p_persist = np.array([y_test[i - 1] for i in persist_idx])
t_persist = y_test[list(persist_idx)]
m = calc_metrics(t_persist, p_persist)
rows.append({"model": "① Persistence(参照)", **m})
curves["① Persistence(参照)"] = (list(persist_idx), p_persist)
print(f"  ① Persistence  RMSE={m['RMSE']:.5f}  （这是必须超越的下限）")


# ---------------------------------------------------------------- NeuralForecast
def to_nf(df, uid, off_days=0):
    """转长表。不同批次用 off_days 错开时间轴，避免 ds 重叠。"""
    o = df[["timestamp", "outlet_moisture"] + EXOG].copy()
    o.insert(0, "unique_id", uid)
    o = o.rename(columns={"timestamp": "ds", "outlet_moisture": "y"})
    if off_days:
        o["ds"] = o["ds"] + pd.Timedelta(days=off_days)
    return o


# 3 个训练批次 + 1 个测试批次 = 4 条独立序列（全局模型的标准用法）
train_nf = pd.concat(
    [to_nf(simulate_hdt_batch(1200, seed=s), f"train_{i}", i * 100)
     for i, s in enumerate(TRAIN_SEEDS)], ignore_index=True)
test_nf = to_nf(test_df, "test", 400)
all_nf = pd.concat([train_nf, test_nf], ignore_index=True)
N_SERIES = len(TRAIN_SEEDS) + 1

from neuralforecast import NeuralForecast
from neuralforecast.models import TimeXer, iTransformer

print(f"\n--- ② ③ NeuralForecast（{N_SERIES} 条序列，训练 {MAX_STEPS} 步）---")


def run_nf(name, cls, use_hist_exog=False):
    t0 = time.time()
    try:
        kw = dict(
            h=H, input_size=INPUT_SIZE, n_series=N_SERIES,
            max_steps=MAX_STEPS, batch_size=64,
            logger=False, enable_checkpointing=False, accelerator=DEVICE,
        )
        if use_hist_exog:
            kw["hist_exog_list"] = EXOG
        nf = NeuralForecast(models=[cls(**kw)], freq="s")
        nf.fit(df=all_nf)

        preds, trues, idxs = [], [], []
        for t in range(INPUT_SIZE, len(y_test), TEST_STRIDE):
            ctx = pd.concat([train_nf, test_nf.iloc[:t]], ignore_index=True)
            fc = nf.predict(df=ctx)
            col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
            sub = fc[fc["unique_id"] == "test"]
            if len(sub) == 0:
                continue
            preds.append(float(sub[col].to_numpy(dtype=float)[0]))
            trues.append(float(y_test[t])); idxs.append(t)
        met = calc_metrics(np.array(trues), np.array(preds))
        rows.append({"model": name, **met})
        curves[name] = (idxs, np.array(preds))
        print(f"  ✅ {name:16s} RMSE={met['RMSE']:.5f}  MAE={met['MAE']:.5f}  "
              f"R²={met['R2']:.4f}  n={met['n']}  ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  ❌ {name:16s} {type(e).__name__}: {str(e)[:110]}")


run_nf("② TimeXer", TimeXer, use_hist_exog=True)
run_nf("③ iTransformer", iTransformer, use_hist_exog=False)

# ---------------------------------------------------------------- ④ Darts TCN
print("\n--- ④ Darts TCN ---")
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TCNModel

t0 = time.time()
try:
    train_dfs = [simulate_hdt_batch(1200, seed=s) for s in TRAIN_SEEDS]
    tr_ts = [TimeSeries.from_dataframe(d, "timestamp", "outlet_moisture", freq="s") for d in train_dfs]
    tr_cv = [TimeSeries.from_dataframe(d, "timestamp", EXOG, freq="s") for d in train_dfs]
    te_ts = TimeSeries.from_dataframe(test_df, "timestamp", "outlet_moisture", freq="s")
    te_cv = TimeSeries.from_dataframe(test_df, "timestamp", EXOG, freq="s")

    sc_y, sc_c = Scaler(), Scaler()
    tr_ts_s, tr_cv_s = sc_y.fit_transform(tr_ts), sc_c.fit_transform(tr_cv)
    te_ts_s = sc_y.transform(te_ts)
    te_cv_s = sc_c.transform(te_cv)

    tcn = TCNModel(
        input_chunk_length=INPUT_SIZE, output_chunk_length=H,
        kernel_size=3, num_filters=32, dilation_base=2, dropout=0.1,
        n_epochs=30, batch_size=64, random_state=42,
        pl_trainer_kwargs={"accelerator": "auto", "enable_progress_bar": False,
                           "enable_model_summary": False, "logger": False},
    )
    tcn.fit(series=tr_ts_s, past_covariates=tr_cv_s)
    # Darts 原生滚动预测，stride 控制密度
    pf = tcn.historical_forecasts(
        series=te_ts_s, past_covariates=te_cv_s, start=INPUT_SIZE,
        forecast_horizon=H, stride=TEST_STRIDE, retrain=False,
        last_points_only=True, verbose=False)
    # 注意：用单条序列 transform，避免 Scaler 在多序列上报警
    pred = sc_y.inverse_transform(pf).values().ravel()

    idxs = list(range(INPUT_SIZE, len(y_test), TEST_STRIDE))
    n = min(len(pred), len(idxs))
    met = calc_metrics(y_test[idxs[:n]], pred[:n])
    rows.append({"model": "④ TCN(Darts)", **met})
    curves["④ TCN(Darts)"] = (idxs[:n], pred[:n])
    print(f"  ✅ {'④ TCN(Darts)':16s} RMSE={met['RMSE']:.5f}  MAE={met['MAE']:.5f}  "
          f"R²={met['R2']:.4f}  n={met['n']}  ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"  ❌ ④ TCN(Darts) {type(e).__name__}: {str(e)[:130]}")

# ---------------------------------------------------------------- 汇总
print("\n" + "=" * 78)
print("深度模型结果（统一口径：对测试批次逐步预测下一秒）")
print("=" * 78)
tbl = pd.DataFrame(rows)
cols = [c for c in ["model", "MAE", "RMSE", "MAPE(%)", "R2"] if c in tbl.columns]
print(tbl[cols].round(5).to_string(index=False))
tbl[cols].to_csv(REP / "s3_dl.csv", index=False, encoding="utf-8-sig")

base = rows[0]["RMSE"]
print("\n【解读】")
print(f"  Persistence 下限 RMSE = {base:.5f}")
for r in rows[1:]:
    d = (base - r["RMSE"]) / base * 100
    print(f"  {r['model']:18s} {d:+7.1f}%   {'优于基线' if d > 0 else '仍不如基线'}")

print("""
  为什么深度模型打不过"复制上一步"？
  ─────────────────────────────────────
  仿真数据的出口水分每步只朝目标移动 5%（w[k] = w[k-1] + 0.05·Δ），
  加上 N(0, 0.01) 的极小噪声。这意味着：
    相邻两秒的值几乎完全一样 → "复制上一步"就已经接近最优。
  神经网络要学的是从 4 个外生变量到"未来水分增量"的映射，
  而这个增量本身只有 0.01 量级，信噪比极低，模型很难在这里超过惯性。

  真实 HDT 数据完全不同：来料波动、蒸汽压力抖动、换牌、断料让出口水分
  真正在动，那时外生变量才有用武之地，深度模型也才会体现价值。

  所以：本脚本的价值在于【验证工具链跑通】，而不是【选出最好的模型】。
  等真实数据进来，用完全相同的脚本重跑，结论才有意义。""")

# ---------------------------------------------------------------- 绘图
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(test_df["timestamp"], y_test, color=COLORS["target"], lw=1.2, alpha=.85, label="实际出口水分")
for i, (k, (idxs, pred)) in enumerate(curves.items()):
    if k.startswith("①"):
        continue
    ax.plot(test_df["timestamp"].iloc[idxs], pred, lw=1.0, alpha=.9,
            color=PALETTE[i % len(PALETTE)], label=k)
ax.set_title("深度模型逐步预测 vs 实际（独立测试批次）", fontweight="bold")
ax.set_ylabel("出口含水率 (%)"); ax.set_xlabel("时间")
ax.legend(fontsize=8, loc="upper left")
fig.savefig(FIG / "s3_dl_compare.png", dpi=110); plt.close(fig)
print(f"\n图已保存 -> {FIG / 's3_dl_compare.png'}")