#!/usr/bin/env python3
"""
=== 真实数据脚本 4E：出口含水率 60s 专项攻坚（120s 长窗口 + 物料能量平衡物理特征 + 量化抗噪）===

针对痛点 2（出口含水率 60s Skill 仅 +2.42%）与痛点 3（1A/1B 0.01% 粗量化与缺失 valve_sp），
专门打造「出口含水率长程预测专精方案」：
  1. 剔除外部数据缺失的 steam_valve_sp，统一使用 8 个全域共有的标准物理列，实现 18 批次与 1A/1B 无缝兼容；
  2. 延长回看窗口 input_size = 120 秒（覆盖实测 40~80s 水分纯滞后与完整惯性周期）；
  3. 构造 4 维物料/能量平衡物理特征（全部严格因果）：
     - net_humidification = outlet_moisture - inlet_moisture（净增湿量）
     - specific_steam     = steam_flow / (feed_rate / 1000)（吨烟耗汽量）
     - specific_casing    = casing_flow / (feed_rate / 1000)（吨烟加料比）
     - cum_steam_60s      = 过去 60s 相对蒸汽流量积分（反映滚筒蓄热蓄湿能量）
  4. 15 秒平滑抗量化特征（专治 1A/1B 的 0.01% 锯齿台阶），在 4 月测试集 + 1 月 1A/1B 上同步验证！

运行：
    cd ~/hdt_dryer_mpc
    /opt/conda_envs/dryer-ts/bin/python src_real/r4e_moisture_specialist.py

产出：
    outputs_real/reports/r4e_moisture_specialist.csv
    outputs_real/reports/r4e_moisture_specialist.txt
    outputs_real/figures/r4e_moisture_60s_upgrade.png
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import torch
import torch.nn as nn

from real_common import REAL_PROC, REAL_OUT, REAL_FIG, REAL_REP, COLORS, calc_metrics, save_real_report
from r3_build_dataset import build_windows_for_batch
from r4_train_and_eval import predict_torch_model
from r4c_physics_guided_models import moving_avg_1d, PhysicsGRUSpecialist, train_specialist_with_physics
from r4d_external_test_1A_1B import align_1A_1B_to_feature_cols

# 全域统一 8 维基础列（不含 1A/1B 缺失的 steam_valve_sp，彻底消除跨数据源接口不一致）
UNIFIED_BASE_COLS = [
    "outlet_moisture",
    "outlet_temp",
    "steam_flow",
    "steam_valve_fb",
    "casing_flow",
    "inlet_moisture",
    "feed_rate",
    "return_air_temp",
]


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造严格因果的物料/能量平衡衍生特征（共 12 维）。"""
    out = df[UNIFIED_BASE_COLS].copy()
    feed_ton = np.maximum(out["feed_rate"].to_numpy(dtype=float) / 1000.0, 1.0)
    out["net_humid"] = out["outlet_moisture"] - out["inlet_moisture"]
    out["spec_steam"] = out["steam_flow"] / feed_ton
    out["spec_casing"] = out["casing_flow"] / feed_ton
    # 过去 60s 因果滑动蒸汽均值（反映进入滚筒的累积热湿能量）
    out["steam_ma60"] = out["steam_flow"].rolling(60, min_periods=1).mean()
    return out


PHYS_FEATURE_COLS = UNIFIED_BASE_COLS + ["net_humid", "spec_steam", "spec_casing", "steam_ma60"]


def to_denoised_revin_120s(X_seq: np.ndarray) -> np.ndarray:
    """将 120s 窗口变换为抗 0.01% 量化噪声的 2F 相对化序列：
    - [0:F]  : 15s 因果平滑相对位移 MA15(X)(τ) - MA15(X)(t)
    - [F:2F] : 15s 平滑中程斜率（抑制单秒量化跳变）
    """
    x_ma15 = moving_avg_1d(X_seq, k=15)
    x_rel = x_ma15 - x_ma15[:, -1:, :]
    slope15 = np.zeros_like(x_ma15)
    slope15[:, 15:, :] = (x_ma15[:, 15:, :] - x_ma15[:, :-15, :]) / 15.0
    return np.concatenate([x_rel, slope15], axis=-1).astype(np.float32)


def build_moisture_tabular_120s(X_seq: np.ndarray) -> np.ndarray:
    """面向 120s 长窗口的平滑滞后 + 多尺度积分表格特征（专攻 30s/60s 水分预测）。"""
    x_ma15 = moving_avg_1d(X_seq, k=15)
    x_rel = x_ma15 - x_ma15[:, -1:, :]
    N, T, F = x_rel.shape
    # 覆盖完整 120s 纯滞后与惯性：-10, -25, -45, -70, -95, -120s
    lag_offsets = [10, 25, 45, 70, 95, T]
    lag_idx = [max(0, T - o) for o in lag_offsets]
    lag_feats = x_rel[:, lag_idx, :].reshape(N, -1)

    m_30 = x_rel[:, -30:, :].mean(axis=1)
    m_60 = x_rel[:, -60:, :].mean(axis=1)
    m_120 = x_rel.mean(axis=1)
    std_60 = x_rel[:, -60:, :].std(axis=1)
    slope_short = x_ma15[:, -1, :] - x_ma15[:, -20, :]
    slope_mid = x_ma15[:, -1, :] - x_ma15[:, -60, :]
    slope_long = x_ma15[:, -1, :] - x_ma15[:, 0, :]
    return np.hstack([lag_feats, m_30, m_60, m_120, std_60, slope_short, slope_mid, slope_long]).astype(np.float32)


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_dir = REAL_PROC / "batches_named"
    csv_files = sorted(batch_dir.glob("*.csv"))
    train_files, val_files, test_files = csv_files[:12], csv_files[12:14], csv_files[14:]

    def _load(paths):
        out = []
        for p in paths:
            df = pd.read_csv(p, encoding="utf-8-sig").query("phase == 'steady'").reset_index(drop=True)
            out.append((p.stem, add_physics_features(df)))
        return out

    train_batches = _load(train_files)
    val_batches = _load(val_files)
    test_batches = _load(test_files)

    # 外部 1A / 1B 也走完全相同的 add_physics_features 接口（零缺失列！）
    ext_batches = []
    for tag in ["1A", "1B"]:
        df_ext = pd.read_csv(REAL_PROC / f"{tag}_steady.csv")
        df_ext_aligned = align_1A_1B_to_feature_cols(df_ext)
        ext_batches.append((tag, add_physics_features(df_ext_aligned)))

    train_concat = pd.concat([df[PHYS_FEATURE_COLS] for _, df in train_batches], ignore_index=True)
    clip_low = train_concat.quantile(0.001).to_numpy(dtype=np.float32)
    clip_high = train_concat.quantile(0.999).to_numpy(dtype=np.float32)
    scaler = RobustScaler().fit(np.clip(train_concat.to_numpy(dtype=np.float32), clip_low, clip_high))

    input_size = 120
    horizons = [15, 30, 60]
    n_h = len(horizons)

    def _build(batches, stride):
        xs, yfs, ycs = [], [], []
        for _, df in batches:
            rf = df[PHYS_FEATURE_COLS].to_numpy(dtype=np.float32)
            rt = df[["outlet_moisture"]].to_numpy(dtype=np.float32)
            fs = scaler.transform(np.clip(rf, clip_low, clip_high)).astype(np.float32)
            x, yf, yc = build_windows_for_batch(fs, rt, input_size, horizons, stride)
            if len(x) > 0:
                xs.append(x); yfs.append(yf[:, :, 0]); ycs.append(yc[:, 0])
        return np.concatenate(xs), np.concatenate(yfs), np.concatenate(ycs)

    X_tr, Y_tr, Yc_tr = _build(train_batches, stride=2)
    X_va, Y_va, Yc_va = _build(val_batches, stride=5)
    X_te, Y_te, Yc_te = _build(test_batches, stride=5)

    dY_tr = Y_tr - Yc_tr[:, None]
    dY_va = Y_va - Yc_va[:, None]
    dy_std = dY_tr.std(axis=0, keepdims=True) + 1e-6

    print("=" * 78)
    print(f"真实数据脚本 4E：出口含水率 120s 物理特征专精攻坚 | Device={device}")
    print(f"Train(120s, stride=2)={len(X_tr)}, Val={len(X_va)}, Test(4月)={len(X_te)}")
    print("=" * 78)

    X_tr_tab = build_moisture_tabular_120s(X_tr)
    X_va_tab = build_moisture_tabular_120s(X_va)
    X_te_tab = build_moisture_tabular_120s(X_te)

    # 1. Ridge_Phys120
    ridge = Ridge(alpha=200.0).fit(X_tr_tab, dY_tr / dy_std)

    # 2. LightGBM_Phys120
    lgb_models = []
    for hi in range(n_h):
        reg = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=23, min_child_samples=60,
            reg_alpha=2.0, reg_lambda=10.0, colsample_bytree=0.7, subsample=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        reg.fit(X_tr_tab, dY_tr[:, hi], eval_set=[(X_va_tab, dY_va[:, hi])], callbacks=[lgb.early_stopping(30, verbose=False)])
        lgb_models.append(reg)

    # 3. GRU_Phys120（带 15s 去噪 RevIN + 物理单调性正则）
    X_tr_seq = to_denoised_revin_120s(X_tr)
    X_va_seq = to_denoised_revin_120s(X_va)
    steam_idx = PHYS_FEATURE_COLS.index("steam_flow")
    spec_steam_idx = PHYS_FEATURE_COLS.index("spec_steam")
    gru = PhysicsGRUSpecialist(in_dim=X_tr_seq.shape[-1], hidden_dim=32, n_horizons=n_h, num_layers=2, dropout=0.30)
    ckpt = REAL_OUT / "checkpoints" / "r4e_moisture_gru120.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    train_specialist_with_physics(
        gru, X_tr_seq, (dY_tr / dy_std).astype(np.float32), X_va_seq, (dY_va / dy_std).astype(np.float32),
        steam_channels=[steam_idx, spec_steam_idx],
        horizon_weights=[1.0, 1.5, 2.0],
        phys_weight=0.30, phys_margin=0.04,
        epochs=30, batch_size=256, lr=8e-4, weight_decay=8e-3, noise_std=0.03,
        device=device, ckpt_path=ckpt, patience=8,
    )

    # 统一在 3 个测试域（4月4批 / 1月1A / 1月1B）评估
    eval_domains = [("April_4batches", X_te, Y_te, Yc_te)]
    for tag, df_e in ext_batches:
        xe, ye, yce = _build([(tag, df_e)], stride=1)
        eval_domains.append((f"{tag}_20250126", xe, ye, yce))

    rows = []
    april_preds_60 = {}
    for dname, Xd, Yd, Ycd in eval_domains:
        Xd_tab = build_moisture_tabular_120s(Xd)
        Xd_seq = to_denoised_revin_120s(Xd)

        p_naive = np.repeat(Ycd[:, None], n_h, axis=1)
        p_ridge = Ycd[:, None] + ridge.predict(Xd_tab) * dy_std
        p_lgb = Ycd[:, None] + np.column_stack([m.predict(Xd_tab) for m in lgb_models])
        p_gru = Ycd[:, None] + predict_torch_model(gru, Xd_seq, device) * dy_std
        p_blend = 0.4 * p_ridge + 0.3 * p_lgb + 0.3 * p_gru

        if dname == "April_4batches":
            april_preds_60 = {"Naive": p_naive[:, -1], "Ridge_Phys120": p_ridge[:, -1], "LightGBM_Phys120": p_lgb[:, -1], "GRU_Phys120": p_gru[:, -1], "Blend_Phys120": p_blend[:, -1], "True": Yd[:, -1]}

        for hi, h in enumerate(horizons):
            nv_rmse = calc_metrics(Yd[:, hi], p_naive[:, hi])["rmse"]
            for mname, phat in [
                ("Naive", p_naive),
                ("Ridge_Phys120", p_ridge),
                ("LightGBM_Phys120", p_lgb),
                ("GRU_Phys120", p_gru),
                ("Blend_Phys120", p_blend),
            ]:
                m = calc_metrics(Yd[:, hi], phat[:, hi])
                skill = (1.0 - m["rmse"] / (nv_rmse + 1e-12)) * 100.0
                rows.append({
                    "domain": dname, "model": mname, "horizon_s": h,
                    "rmse": round(m["rmse"], 5), "mae": round(m["mae"], 5),
                    "r2": round(m["r2"], 4), "skill_vs_naive_pct": round(skill, 2),
                })

    res_df = pd.DataFrame(rows)
    res_df.to_csv(REAL_REP / "r4e_moisture_specialist.csv", index=False, encoding="utf-8-sig")

    # 绘图：三测试域 60s 水分 Skill 柱状对比
    fig, ax = plt.subplots(figsize=(11, 4.8))
    domains = ["April_4batches", "1A_20250126", "1B_20250126"]
    m_list = ["Ridge_Phys120", "LightGBM_Phys120", "GRU_Phys120", "Blend_Phys120"]
    x_pos = np.arange(len(domains))
    w = 0.18
    for i, mname in enumerate(m_list):
        vals = [res_df[(res_df["domain"] == d) & (res_df["model"] == mname) & (res_df["horizon_s"] == 60)]["skill_vs_naive_pct"].iloc[0] for d in domains]
        ax.bar(x_pos + (i - 1.5) * w, vals, width=w, label=mname)
    ax.axhline(0, color="gray", ls="--", lw=1.0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["4月 4批次 (0.001%)", "1月 1A (0.01%粗量化)", "1月 1B (0.01%高波动)"])
    ax.set_ylabel("60s 出口含水率 Skill vs Naive (%)")
    ax.set_title("r4E 120s 物理特征 + 15s 抗量化平滑：跨域 60s 出口含水率 Skill 对比", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REAL_FIG / "r4e_moisture_60s_upgrade.png", dpi=120)
    plt.close(fig)

    rep_text = "\n".join([
        "=" * 82,
        "真实数据脚本 4E：出口含水率 120s 物理特征 + 15s 抗量化平滑专项报告",
        "=" * 82,
        res_df.to_string(index=False),
    ])
    print("\n" + rep_text)
    save_real_report("r4e_moisture_specialist.txt", rep_text)


if __name__ == "__main__":
    main()
