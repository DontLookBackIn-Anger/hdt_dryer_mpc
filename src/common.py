"""
公共工具：中文字体、统一配色、评估指标、数据生成、训练/测试切分。

所有学习脚本都 import 本模块，保证风格一致、口径可比。
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW, DATA_PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
FIG, REP, OUT = ROOT / "outputs" / "figures", ROOT / "outputs" / "reports", ROOT / "outputs"
for _d in (DATA_RAW, DATA_PROC, FIG, REP, OUT):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 中文字体
for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
    if Path(_p).exists():
        try:
            fm.fontManager.addfont(_p)
        except Exception:
            pass
_avail = {f.name for f in fm.fontManager.ttflist}
for _n in ["Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
    if _n in _avail:
        plt.rcParams["font.sans-serif"] = [_n, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.bbox"] = "tight"

COLORS = {"target": "#c0392b", "pred": "#2980b9", "split": "#7f8c8d",
          "good": "#27ae60", "warn": "#e67e22"}
PALETTE = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#16a085"]


# ---------------------------------------------------------------- 数据生成
def simulate_hdt_batch(n_steps: int = 1200, seed: int = 42) -> pd.DataFrame:
    """模拟单个 HDT 气流烘丝机稳定生产批次（1 Hz 采样）。

    与 src/benchmark_dryer.py 保持一致，便于横向对比。
    工艺设定：
      - 操作量 MV: steam_flow(蒸汽流量), hot_air_flow(热风流量)
      - 扰动量 DV: inlet_moisture(入口水分), feed_rate(物料流量)
      - 被控量 CV: outlet_moisture(出口含水率), outlet_temp(出口温度)
      - 动态: 12~18 秒纯滞后 + 一阶惯性
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    steam_flow = 260.0 + 35.0 * np.sin(2 * np.pi * t / 300.0) + rng.normal(0, 2.0, n_steps)
    steam_flow[400:700] += 30.0

    hot_air_flow = 9500.0 + 500.0 * np.cos(2 * np.pi * t / 420.0) + rng.normal(0, 40.0, n_steps)
    hot_air_flow[750:1000] -= 450.0

    inlet_moisture = 20.0 + 0.8 * np.sin(2 * np.pi * t / 500.0) + rng.normal(0, 0.1, n_steps)
    feed_rate = 3800.0 + 200.0 * np.sin(2 * np.pi * t / 600.0) + rng.normal(0, 20.0, n_steps)

    w_out = np.full(n_steps, 12.5, dtype=np.float32)
    t_out = np.full(n_steps, 65.0, dtype=np.float32)

    for k in range(20, n_steps):
        u_steam = steam_flow[k - 14]
        u_air = hot_air_flow[k - 12]
        d_moist = inlet_moisture[k - 18]
        d_feed = feed_rate[k - 16]

        w_target = (12.5
                    + 0.05 * (d_moist - 20.0)
                    + 0.0008 * (d_feed - 3800.0) * 0.1
                    - 0.02 * (u_steam - 260.0)
                    - 0.0006 * (u_air - 9500.0))
        t_target = (65.0
                    + 0.8 * (u_air - 9500.0) * 0.1
                    + 0.05 * (u_steam - 260.0)
                    - 0.3 * (d_moist - 20.0))

        w_out[k] = w_out[k - 1] + 0.05 * (w_target - w_out[k - 1]) + rng.normal(0, 0.01)
        t_out[k] = t_out[k - 1] + 0.05 * (t_target - t_out[k - 1]) + rng.normal(0, 0.05)

    return pd.DataFrame({
        "timestamp": pd.date_range("2026-11-01", periods=n_steps, freq="1s"),
        "steam_flow": steam_flow.astype(np.float32),
        "hot_air_flow": hot_air_flow.astype(np.float32),
        "inlet_moisture": inlet_moisture.astype(np.float32),
        "feed_rate": feed_rate.astype(np.float32),
        "outlet_moisture": w_out,
        "outlet_temp": t_out,
    })


MV_COLS = ["steam_flow", "hot_air_flow"]
DV_COLS = ["inlet_moisture", "feed_rate"]
EXOG_COLS = MV_COLS + DV_COLS
CV_COLS = ["outlet_moisture", "outlet_temp"]


# ---------------------------------------------------------------- 评估
def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """回归评估指标。注意小量级目标（如水分 12.5%）要看绝对误差。"""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    err = y_pred - y_true
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE(%)": float(np.mean(np.abs(err) / (np.abs(y_true) + 1e-12)) * 100),
        "R2": float(1 - ss_res / ss_tot),
        "n": int(mask.sum()),
    }


def print_metrics_table(rows: list, title: str = "") -> pd.DataFrame:
    """rows: [{'model':..., 'MAE':..., ...}] -> 打印并返回 DataFrame。"""
    if not rows:
        print("(无结果)"); return pd.DataFrame()
    df = pd.DataFrame(rows)
    if title:
        print(f"\n{title}")
        print("-" * 72)
    cols = [c for c in ["model", "MAE", "RMSE", "MAPE(%)", "R2"] if c in df.columns]
    print(df[cols].round(4).to_string(index=False))
    return df[cols]


def save_report(name: str, text: str):
    (REP / name).write_text(text, encoding="utf-8")
    print(f"报告已保存 -> {REP / name}")