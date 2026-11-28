#!/usr/bin/env python3
"""
=== 真实数据脚本 4F：多执行器（蒸汽 + 加水）耦合工况建模验证（transformer_data.csv）===

针对痛点 4：18 批次中 water_flow 全为 0，仅 transformer_data.csv 包含非零加水流量（0~35.5）。
本脚本完成：
  1. 自动清洗 transformer_data.csv 末尾 156s 停机降温段（outlet_temp < 45℃ 或 steam_flow < 1）；
  2. 严格按时间顺序切分 70% 训练 / 15% 验证 / 15% 测试（零泄露，仅训练集 fit RobustScaler）；
  3. 统一对比 Naive / Ridge_MultiAct / LightGBM_MultiAct / GRU_MultiAct_Phys（同时受蒸汽+加水双执行器驱动）；
  4. 执行「双执行器物理方向检验」：分别微扰 steam_flow 与 water_flow，验证出口含水率响应方向！

运行：
    cd ~/hdt_dryer_mpc
    /opt/conda_envs/dryer-ts/bin/python src_real/r4f_multi_actuator_transformer.py

产出：
    outputs_real/reports/r4f_multi_actuator.csv
    outputs_real/reports/r4f_multi_actuator.txt
    outputs_real/figures/r4f_multi_actuator_60s.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import torch

from real_common import REAL_PROC, REAL_OUT, REAL_FIG, REAL_REP, COLORS, calc_metrics, save_real_report
from r3_build_dataset import build_windows_for_batch
from r4_train_and_eval import predict_torch_model
from r4b_revin_optimized import to_revin_sequence, build_revin_tabular
from r4c_physics_guided_models import PhysicsGRUSpecialist, train_specialist_with_physics

TF_TARGET_COLS = ["outlet_moisture", "outlet_temp"]
TF_FEATURE_COLS = ["outlet_moisture", "outlet_temp", "steam_flow", "water_flow", "inlet_moisture"]


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(REAL_PROC / "transformer_steady.csv")
    raw_len = len(df)
    # 剔除末尾停机降温段（索引 > 3000 后出现 outlet_temp < 45℃ 或 steam_flow < 1）
    bad_idx = np.where((df.index.to_numpy() > 3000) & ((df["outlet_temp"] < 45.0) | (df["steam_flow"] < 1.0)))[0]
    cut_end = int(bad_idx[0]) - 20 if len(bad_idx) > 0 else len(df)
    df_clean = df.iloc[:cut_end].reset_index(drop=True)

    # 严格按时间顺序切分 70% / 15% / 15%
    n = len(df_clean)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    tr_df, va_df, te_df = df_clean.iloc[:i1], df_clean.iloc[i1:i2], df_clean.iloc[i2:]

    clip_low = tr_df[TF_FEATURE_COLS].quantile(0.001).to_numpy(dtype=np.float32)
    clip_high = tr_df[TF_FEATURE_COLS].quantile(0.999).to_numpy(dtype=np.float32)
    scaler = RobustScaler().fit(np.clip(tr_df[TF_FEATURE_COLS].to_numpy(dtype=np.float32), clip_low, clip_high))

    input_size = 60
    horizons = [1, 15, 30, 60]
    n_h, n_t = len(horizons), len(TF_TARGET_COLS)

    def _win(sub_df, stride):
        rf = sub_df[TF_FEATURE_COLS].to_numpy(dtype=np.float32)
        rt = sub_df[TF_TARGET_COLS].to_numpy(dtype=np.float32)
        fs = scaler.transform(np.clip(rf, clip_low, clip_high)).astype(np.float32)
        return build_windows_for_batch(fs, rt, input_size, horizons, stride)

    X_tr, Y_tr, Yc_tr = _win(tr_df, stride=1)
    X_va, Y_va, Yc_va = _win(va_df, stride=1)
    X_te, Y_te, Yc_te = _win(te_df, stride=1)

    dY_tr = Y_tr - Yc_tr[:, None, :]
    dY_va = Y_va - Yc_va[:, None, :]
    dy_std = dY_tr.std(axis=0, keepdims=True) + 1e-6

    X_tr_tab, X_va_tab, X_te_tab = build_revin_tabular(X_tr), build_revin_tabular(X_va), build_revin_tabular(X_te)
    X_tr_seq, X_va_seq, X_te_seq = to_revin_sequence(X_tr), to_revin_sequence(X_va), to_revin_sequence(X_te)

    preds = {"Naive": np.repeat(Yc_te[:, None, :], n_h, axis=1)}

    # 1. Ridge_MultiAct
    ridge = Ridge(alpha=50.0).fit(X_tr_tab, (dY_tr / dy_std).reshape(len(X_tr), -1))
    preds["Ridge_MultiAct"] = Yc_te[:, None, :] + ridge.predict(X_te_tab).reshape(-1, n_h, n_t) * dy_std

    # 2. LightGBM_MultiAct
    lgb_d = np.zeros_like(Y_te)
    for hi in range(n_h):
        for ti in range(n_t):
            reg = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.04, num_leaves=23, min_child_samples=30,
                reg_alpha=1.0, reg_lambda=5.0, random_state=42, n_jobs=-1, verbose=-1,
            )
            reg.fit(X_tr_tab, dY_tr[:, hi, ti], eval_set=[(X_va_tab, dY_va[:, hi, ti])], callbacks=[lgb.early_stopping(25, verbose=False)])
            lgb_d[:, hi, ti] = reg.predict(X_te_tab)
    preds["LightGBM_MultiAct"] = Yc_te[:, None, :] + lgb_d

    # 3. GRU_MultiAct_Phys（双执行器物理单调性约束：蒸汽 + 加水 -> 水分正向）
    steam_idx = TF_FEATURE_COLS.index("steam_flow")
    water_idx = TF_FEATURE_COLS.index("water_flow")
    gru_pred = np.zeros_like(Y_te)
    gru_models = {}
    for ti, tname in enumerate(TF_TARGET_COLS):
        mdl = PhysicsGRUSpecialist(in_dim=X_tr_seq.shape[-1], hidden_dim=32, n_horizons=n_h, num_layers=2, dropout=0.25)
        ch_list = [steam_idx, water_idx] if tname == "outlet_moisture" else [steam_idx]
        ckpt = REAL_OUT / "checkpoints" / f"r4f_gru_multiact_{tname}.pt"
        train_specialist_with_physics(
            mdl, X_tr_seq, (dY_tr[:, :, ti] / dy_std[0, :, ti]).astype(np.float32),
            X_va_seq, (dY_va[:, :, ti] / dy_std[0, :, ti]).astype(np.float32),
            steam_channels=ch_list, horizon_weights=[0.5, 1.0, 1.5, 2.0],
            phys_weight=0.25, phys_margin=0.05,
            epochs=30, batch_size=128, lr=1e-3, weight_decay=5e-3, noise_std=0.02,
            device=device, ckpt_path=ckpt, patience=8,
        )
        gru_models[tname] = mdl
        gru_pred[:, :, ti] = Yc_te[:, ti : ti + 1] + predict_torch_model(mdl, X_te_seq, device) * dy_std[0, :, ti][None, :]
    preds["GRU_MultiAct_Phys"] = gru_pred

    # 评估
    rows = []
    for hi, h in enumerate(horizons):
        for ti, tname in enumerate(TF_TARGET_COLS):
            nv = calc_metrics(Y_te[:, hi, ti], preds["Naive"][:, hi, ti])["rmse"]
            for mname, phat in preds.items():
                m = calc_metrics(Y_te[:, hi, ti], phat[:, hi, ti])
                skill = (1.0 - m["rmse"] / (nv + 1e-12)) * 100.0
                rows.append({
                    "model": mname, "target": tname, "horizon_s": h,
                    "rmse": round(m["rmse"], 5), "mae": round(m["mae"], 5),
                    "r2": round(m["r2"], 4), "skill_vs_naive_pct": round(skill, 2),
                })
    res_df = pd.DataFrame(rows)
    res_df.to_csv(REAL_REP / "r4f_multi_actuator.csv", index=False, encoding="utf-8-sig")

    # 双执行器物理方向检验（steam_flow +1 IQR 与 water_flow +1 IQR）
    phys_lines = ["\n【双执行器物理方向检验（60s 处平均响应）】"]
    for act_name, idx in [("steam_flow", steam_idx), ("water_flow", water_idx)]:
        Xp = X_te[:300].copy()
        Xp[:, -20:, idx] += 1.0
        Xp_seq = to_revin_sequence(Xp)
        b_m = predict_torch_model(gru_models["outlet_moisture"], X_te_seq[:300], device)[:, -1] * dy_std[0, -1, 0]
        p_m = predict_torch_model(gru_models["outlet_moisture"], Xp_seq, device)[:, -1] * dy_std[0, -1, 0]
        b_t = predict_torch_model(gru_models["outlet_temp"], X_te_seq[:300], device)[:, -1] * dy_std[0, -1, 1]
        p_t = predict_torch_model(gru_models["outlet_temp"], Xp_seq, device)[:, -1] * dy_std[0, -1, 1]
        phys_lines.append(f"  {act_name:12s} +1 IQR -> Δ水分(60s)={(p_m - b_m).mean():+.5f}% | Δ温度(60s)={(p_t - b_t).mean():+.4f}℃")

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ti, tname in enumerate(TF_TARGET_COLS):
        ax = axes[ti]
        ax.plot(Y_te[:, -1, ti], color="black", lw=1.4, label="真实值")
        ax.plot(preds["Naive"][:, -1, ti], color="gray", ls=":", lw=1.0, label="Naive(60s前)")
        ax.plot(preds["Ridge_MultiAct"][:, -1, ti], color=COLORS["good"], lw=1.1, label="Ridge_MultiAct")
        ax.plot(preds["GRU_MultiAct_Phys"][:, -1, ti], color=COLORS["target"], ls="--", lw=1.2, label="GRU_MultiAct_Phys")
        ax.set_ylabel(tname)
        ax.set_title(f"多执行器 (蒸汽+加水) 清洗稳态测试集 — 提前 60 秒预测 ({tname})", loc="left", fontweight="bold")
        ax.legend(ncol=4, fontsize=8.5)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("测试集样本序号 (1 Hz)")
    fig.tight_layout()
    fig.savefig(REAL_FIG / "r4f_multi_actuator_60s.png", dpi=120)
    plt.close(fig)

    rep_text = "\n".join([
        "=" * 80,
        f"真实数据脚本 4F：多执行器（蒸汽+加水）耦合建模验证 (transformer_data.csv)",
        f"原始行数={raw_len} -> 剔除末尾停机降温段后稳态行数={len(df_clean)} | Test={len(Y_te)} 窗口",
        "=" * 80,
        res_df.to_string(index=False),
        *phys_lines,
    ])
    print("\n" + rep_text)
    save_real_report("r4f_multi_actuator.txt", rep_text)


if __name__ == "__main__":
    main()
