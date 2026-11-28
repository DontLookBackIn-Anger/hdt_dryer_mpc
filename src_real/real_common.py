"""
真实工业数据公共工具（与仿真代码 src/common.py 严格隔离）。

目录分工：
  - 仿真学习线（保持不动）：src/、data/processed/demo_batch.csv、outputs/
  - 真实数据线（独立留痕）：src_real/、data/raw/real/、data/processed/real/、outputs_real/

工艺说明：
  - 真实数据来自烟草制丝「加料回潮工序」（增湿升温：入口水分 ~16.5% -> 出口 ~20.8%，温度 ~50℃），
    与仿真线 HDT 烘丝机（脱水）工艺方向相反，但同属 1Hz、大滞后、强惯性批次过程。
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------------------------------------------------------- 独立路径
ROOT = Path(__file__).resolve().parent.parent
REAL_RAW = ROOT / "data" / "raw" / "real"
REAL_PROC = ROOT / "data" / "processed" / "real"
REAL_OUT = ROOT / "outputs_real"
REAL_FIG = REAL_OUT / "figures"
REAL_REP = REAL_OUT / "reports"

for _d in (REAL_RAW, REAL_PROC, REAL_OUT, REAL_FIG, REAL_REP):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 中文字体
for _p in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]:
    if Path(_p).exists():
        try:
            fm.fontManager.addfont(_p)
        except Exception:
            pass

_avail = {f.name for f in fm.fontManager.ttflist}
for _n in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "AR PL UMing CN"]:
    if _n in _avail:
        plt.rcParams["font.sans-serif"] = [_n, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.bbox"] = "tight"

COLORS = {
    "target": "#c0392b",
    "temp": "#8e44ad",
    "pred": "#2980b9",
    "good": "#27ae60",
    "warn": "#e67e22",
    "muted": "#7f8c8d",
    "teal": "#16a085",
}

# ---------------------------------------------------------------- 字段映射
BATCH_CN_TO_EN = {
    "日期时间": "timestamp",
    "称瞬时流量": "feed_rate",
    "补偿蒸汽流量": "steam_flow",
    "补偿蒸汽阀门开度反馈": "steam_valve",
    "加水流量": "water_flow",
    "回风温度PV": "return_air_temp",
    "入口含水率": "inlet_moisture",
    "加料累积量": "casing_cum",
    "补偿蒸汽压力": "steam_pressure",
    "烟叶出口温度": "outlet_temp",
    "出口含水率": "outlet_moisture",
    "排潮风量": "exhaust_air_flow",
}

TRANSFORMER_TO_EN = {
    "HHYP_ZX312_PID_TobTemp_PV": "outlet_temp",
    "HHYP_ZX315_Status_MoisPV": "outlet_moisture",
    "HHYP_ZX312_BF52_FlowRead_Mess": "steam_flow",
    "HHYP_ZX312_PID_WaterFlow_PV": "water_flow",
    "HHYP_ZX309_Status_MoisPV": "inlet_moisture",
}

EN_LABELS = {
    "feed_rate": "称瞬时流量 (kg/h)",
    "steam_flow": "补偿蒸汽流量",
    "steam_valve": "蒸汽阀门开度 (%)",
    "water_flow": "加水流量",
    "return_air_temp": "回风温度 (℃)",
    "inlet_moisture": "入口含水率 (%)",
    "casing_cum": "加料累积量",
    "steam_pressure": "补偿蒸汽压力",
    "outlet_temp": "烟叶出口温度 (℃)",
    "outlet_moisture": "出口含水率 (%)",
    "exhaust_air_flow": "排潮风量",
}

# 1A / 1B 中的有效外生输入（water_flow 在 1A/1B 中全为 0，在建模时自动剔除常量列）
BATCH_MV_COLS = ["steam_flow", "steam_valve", "exhaust_air_flow"]
BATCH_DV_COLS = ["inlet_moisture", "feed_rate"]
BATCH_AUX_COLS = ["return_air_temp", "steam_pressure"]
BATCH_CV_COLS = ["outlet_moisture", "outlet_temp"]

TRANSFORMER_EXOG_COLS = ["steam_flow", "water_flow", "inlet_moisture"]
TRANSFORMER_CV_COLS = ["outlet_moisture", "outlet_temp"]


# ---------------------------------------------------------------- 数据加载与清洗
def load_named_batch(filename: str) -> tuple[pd.DataFrame, dict]:
    """加载带中文表头的真实批次（1A.csv / 1B.csv），并执行基础清洗与时间断层检测。"""
    path = REAL_RAW / filename
    raw = pd.read_csv(path, encoding="utf-8-sig")

    # 清除尾随逗号导致的空列（如 1B.csv 的 Unnamed: 12）
    dropped_cols = [c for c in raw.columns if raw[c].isna().all() or str(c).startswith("Unnamed")]
    df = raw.drop(columns=dropped_cols, errors="ignore").copy()
    df = df.rename(columns=BATCH_CN_TO_EN)

    # 解析时间戳并按时间排序
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 时间步长与断层标记
    dt = df["timestamp"].diff().dt.total_seconds().fillna(1.0)
    df["dt_sec"] = dt
    gap_mask = dt > 1.5
    df["time_seg_id"] = gap_mask.cumsum().astype(int)

    # 数值列强制转浮点
    num_cols = [c for c in df.columns if c not in ("timestamp", "dt_sec", "time_seg_id")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 检测常量列（如 water_flow 全为 0）
    zero_or_const = [c for c in num_cols if df[c].nunique(dropna=True) <= 1]

    meta = {
        "file": filename,
        "rows": int(len(df)),
        "span_sec": float((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()),
        "start_time": str(df["timestamp"].iloc[0]),
        "end_time": str(df["timestamp"].iloc[-1]),
        "dropped_cols": dropped_cols,
        "missing_total": int(df[num_cols].isna().sum().sum()),
        "n_gaps": int(gap_mask.sum()),
        "gap_details": [
            {
                "row_idx": int(i),
                "prev_time": str(df["timestamp"].iloc[i - 1]),
                "curr_time": str(df["timestamp"].iloc[i]),
                "gap_sec": float(dt.iloc[i]),
            }
            for i in np.where(gap_mask)[0]
        ],
        "const_cols": zero_or_const,
    }
    return df, meta


def segment_batch_phases(
    df: pd.DataFrame,
    feed_thresh: float = 4000.0,
    moist_thresh: float = 20.2,
    temp_thresh: float = 47.0,
    margin_sec: int = 30,
) -> pd.DataFrame:
    """将完整生产批次划分为 startup（料头）/ steady（稳态）/ tail（料尾）。

    规则：
      1) 找到物料流量 > feed_thresh 且出口水分 > moist_thresh 且出口温度 > temp_thresh 的区间；
      2) 首尾各收缩 margin_sec 秒，避免把料头爬升末端或料尾下降前兆混入稳态。
    """
    out = df.copy()
    cond = (
        (out["feed_rate"] > feed_thresh)
        & (out["outlet_moisture"] > moist_thresh)
        & (out["outlet_temp"] > temp_thresh)
    )
    idx = np.where(cond)[0]
    phase = np.full(len(out), "other", dtype=object)

    if len(idx) > 2 * margin_sec + 60:
        s_start = int(idx[0]) + margin_sec
        s_end = int(idx[-1]) - margin_sec
        phase[:s_start] = "startup"
        phase[s_start : s_end + 1] = "steady"
        phase[s_end + 1 :] = "tail"
    else:
        phase[cond] = "steady"

    out["phase"] = phase
    return out


def get_longest_continuous_steady(df: pd.DataFrame) -> pd.DataFrame:
    """提取稳态阶段中「最长且无时间断层（严格 1 Hz 连续）」的子段，用于时滞辨识。"""
    steady = df[df["phase"] == "steady"].copy()
    if steady.empty:
        return steady
    # 找出每个连续段（按 time_seg_id 与行号连续性共同分组）
    idx_break = (steady.index.to_series().diff().fillna(1) != 1) | (steady["dt_sec"] > 1.5)
    steady["cont_id"] = idx_break.cumsum()
    best_id = steady["cont_id"].value_counts().idxmax()
    return steady[steady["cont_id"] == best_id].drop(columns=["cont_id"]).reset_index(drop=True)


def load_transformer_dataset(filename: str = "transformer_data.csv") -> tuple[pd.DataFrame, dict]:
    """加载对方提供的纯稳态小数据集 transformer/data.csv（自动去除 BOM）。"""
    path = REAL_RAW / filename
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 清理列名可能残留的空白或 BOM
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df = df.rename(columns=TRANSFORMER_TO_EN)
    df["sample_idx"] = np.arange(len(df), dtype=int)
    df["phase"] = "steady"
    df["dt_sec"] = 1.0
    df["time_seg_id"] = 0

    num_cols = [c for c in TRANSFORMER_CV_COLS + TRANSFORMER_EXOG_COLS if c in df.columns]
    meta = {
        "file": filename,
        "rows": int(len(df)),
        "span_sec": float(len(df) - 1),
        "missing_total": int(df[num_cols].isna().sum().sum()),
        "n_gaps": 0,
        "const_cols": [c for c in num_cols if df[c].nunique(dropna=True) <= 1],
    }
    return df, meta


# ---------------------------------------------------------------- 通用统计与报告
def acf(x: np.ndarray, nlags: int = 120) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    denom = float(np.dot(x, x)) + 1e-12
    nlags = min(nlags, len(x) - 2)
    return np.array([np.dot(x[: len(x) - k], x[k:]) / denom for k in range(nlags + 1)])


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_real_report(name: str, text: str) -> Path:
    p = REAL_REP / name
    p.write_text(text, encoding="utf-8")
    print(f"报告已保存 -> {p}")
    return p
