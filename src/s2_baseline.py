#!/usr/bin/env python3
"""
=== 学习脚本 2 / 6：基线模型（Darts）===

为什么必须先做基线？
    出口水分惯性很大（ACF 降到 0.5 要 58 秒），意味着「直接复制上一个值」
    就能拿到不错的 RMSE。如果不先测这个下限，深度学习模型 RMSE=0.04
    会让你误以为它很神，其实可能只比"保持"好一点点。

本脚本对比 5 类模型，全部纯 CPU、秒级完成：

  ① Persistence（复制上一步）  —— 绝对下限，任何模型必须显著超过它才有价值
  ② 24 步均值平滑              —— 再简单不过的"平均"，验证噪声水平
  ③ ExponentialSmoothing       —— 水平+趋势+季节 三分解，经典统计方法
  ④ LightGBM                   —— 梯度提升树，把过去 60 步当特征做自回归
  ⑤ LightGBM + 外生变量         —— ★ 关键对比：加入蒸汽/热风/入口水分是否变好

评估口径（务必理解，这决定了你的结论可不可信）：
  - 用 3 个批次训练，1 个**独立**批次测试（绝不在同一批次内切分，否则会偷看未来）
  - 预测未来 H=15 秒
  - 与 benchmark_dryer.py 口径一致，可直接横向比较
  - 注意：Darts 的 LocalForecastingModel（如 Naive/ES）不支持 retrain=False，
    因此对它们采用 PyTorch 风格的滚动预测实现，见下面的 rolling_forecast()

运行：
    cd ~/hdt_dryer_mpc
    /opt/conda_envs/dryer-ts/bin/python src/s2_baseline.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ExponentialSmoothing, LightGBMModel, XGBModel
from darts.utils.utils import SeasonalityMode
from common import (simulate_hdt_batch, calc_metrics, print_metrics_table,
                    FIG, REP, EXOG_COLS, PALETTE, COLORS)

H = 15                 # 预测未来 15 秒
TRAIN_SEEDS = (10, 20, 30)
TEST_SEED = 99
START = 60             # 从测试批次第 60 秒开始滚动预测

print("=" * 78)
print(f"学习脚本 2：基线模型（预测未来 {H} 秒）")
print("=" * 78)

train_dfs = [simulate_hdt_batch(1200, seed=s) for s in TRAIN_SEEDS]
test_df = simulate_hdt_batch(1200, seed=TEST_SEED)
target = "outlet_moisture"

y_test = test_df[target].to_numpy(dtype=float)
y_train_list = [d[target].to_numpy(dtype=float) for d in train_dfs]
y_train_all = np.concatenate(y_train_list)

rows, preds = [], {}


def record(name, y_pred, t0, **kw):
    """y_pred 可与 y_test[START:] 长度不同；按尾部对齐（Darts 会丢掉开头不足 H 的部分）。"""
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = y_test[START:]
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[-n:], y_pred[-n:]
    m = calc_metrics(y_true, y_pred)
    m["model"] = name
    m["耗时(s)"] = round(time.time() - t0, 2)
    rows.append(m); preds[name] = (y_true, y_pred)
    print(f"  ✅ {name:32s} RMSE={m['RMSE']:.5f}  MAE={m['MAE']:.5f}  "
          f"R²={m['R2']:.4f}  ({m['耗时(s)']}s)")


# ================================================================
# ① Persistence：直接复制上一步。
# 这是任何预测模型必须超越的底线。真实工况下，很多"看起来很准"的
# 深度学习模型其实只是在复现这个基线。
# ================================================================
print("\n--- ① ② 简单参照 ---")
t0 = time.time()
persist = y_test[START - 1:-1]        # t-1 的值作为 t 的预测
record("① Persistence(复制上一步)", persist, t0)

# ================================================================
# ② 24 步滑动均值：用过去 24 秒均值预测下一步。
# 若它RMSE 很低，说明信号本身平滑、噪声小，任务偏简单。
# ================================================================
t0 = time.time()
win = 24
mov = np.array([y_test[i - win:i].mean() for i in range(START, len(y_test))])
record("② 24步滑动均值", mov, t0)


# ================================================================
# ★ 通用滚动预测器
# Darts 的 LocalForecastingModel（ExponentialSmoothing）不支持
# retrain=False 的 historical_forecasts，所以这里手写滚动：
#   对每个起点 t，用 [0, t) 的数据训练，预测 [t, t+H)
# 为控制耗时，统计模型用 stride=5（每 5 秒重训一次）。
# ================================================================
def rolling_stat_model(name, model_factory, stride=5, use_log=False):
    t0 = time.time()
    vals = y_test.copy()
    preds_arr = np.full(len(vals) - START, np.nan)
    ts_full = TimeSeries.from_values(vals.astype(np.float32))
    try:
        for t in range(START, len(vals), stride):
            model = model_factory()
            model.fit(ts_full[:t])
            p = model.predict(H).values().ravel()
            end = min(t + H, len(vals))
            preds_arr[t - START:t - START + (end - t)] = p[:end - t]
        # 填补 stride 间隔
        mask = np.isfinite(preds_arr)
        if mask.sum() > 2:
            idx = np.arange(len(preds_arr))
            preds_arr = np.interp(idx, idx[mask], preds_arr[mask])
        record(name, preds_arr, t0)
    except Exception as e:
        print(f"  ❌ {name:32s} {type(e).__name__}: {str(e)[:95]}")


# ================================================================
# ③ ExponentialSmoothing
# 原理：y_t = 水平 + 趋势 + 季节 + 误差。
# 适合有明显趋势/周期的平稳信号，参数少、可解释性强。
# ================================================================
print("\n--- ③ 经典统计 ---")
rolling_stat_model(
    "③ ExponentialSmoothing",
    lambda: ExponentialSmoothing(seasonal_periods=30, seasonal=SeasonalityMode.ADDITIVE),
    stride=5,
)


# ================================================================
# ④ ⑤ 树模型：Darts 的 LightGBMModel / XGBModel 是全局模型，
# 支持传多个序列训练，也支持 retrain=False 的快速滚动预测。
# ================================================================
def rolling_global_model(name, model, use_cov=False):
    t0 = time.time()
    try:
        train_ts = [TimeSeries.from_dataframe(d, "timestamp", target, freq="s") for d in train_dfs]
        test_ts = TimeSeries.from_dataframe(test_df, "timestamp", target, freq="s")
        if use_cov:
            train_cv = [TimeSeries.from_dataframe(d, "timestamp", EXOG_COLS, freq="s") for d in train_dfs]
            test_cv = TimeSeries.from_dataframe(test_df, "timestamp", EXOG_COLS, freq="s")
            model.fit(series=train_ts, past_covariates=train_cv)
            pf = model.historical_forecasts(
                series=test_ts, past_covariates=test_cv, start=START,
                forecast_horizon=H, stride=1, retrain=False,
                last_points_only=True, verbose=False)
        else:
            model.fit(series=train_ts)
            pf = model.historical_forecasts(
                series=test_ts, start=START, forecast_horizon=H, stride=1,
                retrain=False, last_points_only=True, verbose=False)
        record(name, pf.values().ravel(), t0)
    except Exception as e:
        print(f"  ❌ {name:32s} {type(e).__name__}: {str(e)[:95]}")


print("\n--- ④ ⑤ 树模型（自回归特征）---")
rolling_global_model("④ LightGBM(lags=60)",
                     LightGBMModel(lags=60, random_state=42, verbose=-1))

print("\n--- ★ 关键对比：加不加外生变量 ---")
rolling_global_model("⑤ LightGBM + 外生变量",
                     LightGBMModel(lags=60, lags_past_covariates=60,
                                   output_chunk_length=H, random_state=42, verbose=-1),
                     use_cov=True)

# ================================================================
# 汇总与解读
# ================================================================
tbl = print_metrics_table(rows, "\n基线结果（独立测试批次，预测 15 秒）")
if len(tbl):
    tbl.sort_values("RMSE").to_csv(REP / "s2_baseline.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存 -> {REP / 's2_baseline.csv'}")

    d = {r["model"]: r for r in rows}
    base = d.get("④ LightGBM(lags=60)")
    exo = d.get("⑤ LightGBM + 外生变量")
    per = d.get("① Persistence(复制上一步)")

    print("\n" + "=" * 78)
    print("【关键解读】—— 请特别关注这几点")
    print("=" * 78)
    if per:
        print(f"  ① Persistence 下限 RMSE = {per['RMSE']:.5f}")
        print(f"     含义：哪怕什么都不学，只复制上一步，误差就只有这么大。")
        print(f"     任何模型都必须显著优于 {per['RMSE']:.4f} 才算有真本事。")
    if base and per:
        gain = (per["RMSE"] - base["RMSE"]) / per["RMSE"] * 100
        print(f"  ④ LightGBM 相比 Persistence 提升 {gain:+.1f}%")
    if base and exo:
        imp = (base["RMSE"] - exo["RMSE"]) / base["RMSE"] * 100
        print(f"  ⑤ 加入外生变量后相比纯自回归提升 {imp:+.1f}%")
        if imp > 3:
            print("     → 外生变量有实质增益：工艺操作量确实驱动出口水分，")
            print("       MPC 必须把它们作为输入，否则控制没有抓手。")
        else:
            print("     → 增益有限：可能滞后窗口没对齐，或模型靠自回归已够用。")

# ================================================================
# 绘图
# ================================================================
if preds:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    t_axis = test_df["timestamp"].iloc[START:].to_numpy()
    ax.plot(t_axis, y_test[START:], color=COLORS["target"], lw=1.6, label="实际出口水分")
    for i, (k, (yt, yp)) in enumerate(preds.items()):
        ax.plot(t_axis[-len(yp):], yp, lw=1.0, alpha=.9,
                color=PALETTE[i % len(PALETTE)], label=k)
    ax.set_title(f"独立测试批次：各基线对比（预测 {H} 秒）", fontweight="bold")
    ax.set_ylabel("出口含水率 (%)"); ax.set_xlabel("时间")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.savefig(FIG / "s2_baseline.png", dpi=110); plt.close(fig)
    print(f"\n图已保存 -> {FIG / 's2_baseline.png'}")

print("\n【下一步】运行 src/s3_dl_forecast.py —— 深度模型（TCN / TimeXer）对比。")