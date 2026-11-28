#!/usr/bin/env python3
"""
=== 真实数据脚本 4D：1A.csv / 1B.csv 外部跨日期零样本泛化测试 ===

严格使用在「2025年3月 12个批次」上训练并冻结的 r4c 模型权重（零微调、零重训、零泄露），
直接测试在「2025年1月26日」采集的两个完全独立外部批次（1A_steady.csv、1B_steady.csv）上的
多步预测泛化能力（1s / 15s / 30s / 60s）与物理一致性！

字段对齐说明（严格因果、无未来泄露）：
  - 1A/1B 中 steam_valve 直接对应 steam_valve_fb；
  - 1A/1B 无独立 steam_valve_sp 列，由于实测稳态 corr(fb, sp)=0.99915，以 steam_valve 因果代理；
  - 1A/1B 中 casing_cum 为加料累积量（kg），按过去 30s 因果差分换算为瞬时加料流量 (kg/h)。

运行：
    cd ~/hdt_dryer_mpc
    /opt/conda_envs/dryer-ts/bin/python src_real/r4d_external_test_1A_1B.py

产出（独立留痕于 outputs_real/）：
    outputs_real/reports/r4d_external_1A_1B.csv
    outputs_real/reports/r4d_external_1A_1B.txt
    outputs_real/figures/r4d_external_1A_60s.png
    outputs_real/figures/r4d_external_1B_60s.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import torch

from real_common import REAL_PROC, REAL_OUT, REAL_FIG, REAL_REP, COLORS, calc_metrics, save_real_report
from r3_build_dataset import TARGET_COLS, FEATURE_COLS, build_windows_for_batch
from r4_train_and_eval import predict_torch_model
from r4c_physics_guided_models import (
    to_multiscale_revin,
    DLinearSpecialist,
    PhysicsGRUSpecialist,
    PatchTCNSKNetTransformerSpecialist,
)


def align_1A_1B_to_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    """将 1A_steady / 1B_steady 的字段严格因果对齐到 9 维 FEATURE_COLS。"""
    out = pd.DataFrame(index=df.index)
    out["outlet_moisture"] = df["outlet_moisture"].astype(float)
    out["outlet_temp"] = df["outlet_temp"].astype(float)
    out["steam_flow"] = df["steam_flow"].astype(float)
    out["steam_valve_fb"] = df["steam_valve"].astype(float)
    # 因果代理（稳态 corr=0.99915，不使用未来值）
    out["steam_valve_sp"] = df["steam_valve"].astype(float)
    # 因果 30s 差分估算加料瞬时流量 (kg/h)：(cum(t) - cum(t-30)) / 30 * 3600
    dcum = df["casing_cum"].astype(float).diff(30) * (3600.0 / 30.0)
    out["casing_flow"] = dcum.bfill().clip(lower=0.0, upper=300.0)
    out["inlet_moisture"] = df["inlet_moisture"].astype(float)
    out["feed_rate"] = df["feed_rate"].astype(float)
    out["return_air_temp"] = df["return_air_temp"].astype(float)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = REAL_OUT / "checkpoints"
    horizons = [1, 15, 30, 60]
    input_size = 60
    n_h, n_t = len(horizons), len(TARGET_COLS)

    # 1. 复原 3 月训练集（前 12 批）的截断阈值、RobustScaler 与 dy_std（严格与 r4c 一致）
    batch_dir = REAL_PROC / "batches_named"
    train_files = sorted(batch_dir.glob("*.csv"))[:12]
    train_batches = [
        pd.read_csv(p, encoding="utf-8-sig").query("phase == 'steady'").reset_index(drop=True)
        for p in train_files
    ]
    train_concat = pd.concat([df[FEATURE_COLS] for df in train_batches], ignore_index=True)
    clip_low = train_concat.quantile(0.001).to_numpy(dtype=np.float32)
    clip_high = train_concat.quantile(0.999).to_numpy(dtype=np.float32)
    scaler = RobustScaler().fit(np.clip(train_concat.to_numpy(dtype=np.float32), clip_low, clip_high))

    # 复原训练集 dy_std
    dy_list = []
    for df in train_batches:
        raw_feat = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        raw_targ = df[TARGET_COLS].to_numpy(dtype=np.float32)
        feat_sc = scaler.transform(np.clip(raw_feat, clip_low, clip_high)).astype(np.float32)
        _, yf, yc = build_windows_for_batch(feat_sc, raw_targ, input_size, horizons, stride=1)
        if len(yf) > 0:
            dy_list.append(yf - yc[:, None, :])
    dY_tr = np.concatenate(dy_list, axis=0)
    dy_std = dY_tr.std(axis=0, keepdims=True) + 1e-6  # (1, H, C)

    # 2. 加载冻结的 r4c 模型权重
    in_dim_ms = len(FEATURE_COLS) * 3
    model_specs = [
        ("DLinear_Phys", lambda: DLinearSpecialist(input_size, in_dim_ms, n_h, kernel_size=15)),
        ("GRU_Phys", lambda: PhysicsGRUSpecialist(in_dim_ms, hidden_dim=32, n_horizons=n_h, num_layers=2, dropout=0.25)),
        ("Patch_TCN_SKNet_Trans_Phys", lambda: PatchTCNSKNetTransformerSpecialist(
            in_dim_ms, n_horizons=n_h, d_model=32, nhead=4, num_layers=2, patch_size=5, dropout=0.30
        )),
    ]
    models = {}
    for mname, ctor in model_specs:
        for tname in TARGET_COLS:
            ckpt = ckpt_dir / f"r4c_{mname}_{tname}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"未找到权重 {ckpt}，请先运行 src_real/r4c_physics_guided_models.py")
            m = ctor().to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            m.eval()
            models[(mname, tname)] = m

    # 与 r4c 一致的验证集融合权重
    ens_weights = {"DLinear_Phys": 0.33, "GRU_Phys": 0.35, "Patch_TCN_SKNet_Trans_Phys": 0.32}

    # 3. 逐个测试 1A_steady.csv 和 1B_steady.csv（跨 2 个月零样本外部测试！）
    all_rows = []
    report_lines = [
        "=" * 82,
        "真实数据脚本 4D：1A.csv / 1B.csv（2025-01-26）跨日期零样本外部泛化测试",
        "（所有模型仅在 2025-03 12 批次上训练并冻结，零微调直接推理 1 月外部批次）",
        "=" * 82,
    ]

    for tag in ["1A", "1B"]:
        df_raw = pd.read_csv(REAL_PROC / f"{tag}_steady.csv")
        df_aligned = align_1A_1B_to_feature_cols(df_raw)
        raw_feat = df_aligned[FEATURE_COLS].to_numpy(dtype=np.float32)
        raw_targ = df_aligned[TARGET_COLS].to_numpy(dtype=np.float32)
        feat_sc = scaler.transform(np.clip(raw_feat, clip_low, clip_high)).astype(np.float32)

        # 使用 stride=1 逐秒全量评估！
        X_ext_raw, Y_ext, Yc_ext = build_windows_for_batch(feat_sc, raw_targ, input_size, horizons, stride=1)
        X_ext_ms = to_multiscale_revin(X_ext_raw)

        preds = {"Naive": np.repeat(Yc_ext[:, None, :], n_h, axis=1)}
        for mname, _ in model_specs:
            p = np.zeros_like(Y_ext)
            for ti, tname in enumerate(TARGET_COLS):
                delta = predict_torch_model(models[(mname, tname)], X_ext_ms, device) * dy_std[0, :, ti][None, :]
                p[:, :, ti] = Yc_ext[:, ti : ti + 1] + delta
            preds[mname] = p

        ens = np.zeros_like(Y_ext)
        for mname, w in ens_weights.items():
            ens += w * preds[mname]
        preds["Ensemble_Phys"] = ens

        naive_rmse = {
            (h, tname): calc_metrics(Y_ext[:, hi, ti], preds["Naive"][:, hi, ti])["rmse"]
            for hi, h in enumerate(horizons)
            for ti, tname in enumerate(TARGET_COLS)
        }

        sub_rows = []
        for mname, y_hat in preds.items():
            for hi, h in enumerate(horizons):
                for ti, tname in enumerate(TARGET_COLS):
                    m = calc_metrics(Y_ext[:, hi, ti], y_hat[:, hi, ti])
                    skill = (1.0 - m["rmse"] / (naive_rmse[(h, tname)] + 1e-12)) * 100.0
                    row = {
                        "dataset": f"{tag}_20250126",
                        "model": mname,
                        "target": tname,
                        "horizon_s": h,
                        "rmse": round(m["rmse"], 5),
                        "mae": round(m["mae"], 5),
                        "r2": round(m["r2"], 4),
                        "skill_vs_naive_pct": round(skill, 2),
                    }
                    sub_rows.append(row)
                    all_rows.append(row)

        report_lines.append(
            f"\n>>> 外部批次 {tag} (2025-01-26 连续稳态 {len(df_raw)}s, 逐秒测试窗口 {len(Y_ext)} 个)"
        )
        report_lines.append(
            f"    稳态统计: 出口含水率 μ={df_aligned['outlet_moisture'].mean():.4f}%(σ={df_aligned['outlet_moisture'].std():.4f}%), "
            f"出口温度 μ={df_aligned['outlet_temp'].mean():.3f}℃(σ={df_aligned['outlet_temp'].std():.3f}℃), "
            f"估算加料流量 μ={df_aligned['casing_flow'].mean():.1f} kg/h"
        )
        report_lines.append(pd.DataFrame(sub_rows).to_string(index=False))

        # 绘制 60s 提前预测图（聚焦 GRU_Phys 与 Ensemble_Phys vs Naive）
        h_idx = len(horizons) - 1
        fig, axes = plt.subplots(2, 1, figsize=(14, 7.2), sharex=True)
        for ti, tname in enumerate(TARGET_COLS):
            ax = axes[ti]
            ax.plot(Y_ext[:, h_idx, ti], color="black", lw=1.4, label="真实值")
            ax.plot(preds["Naive"][:, h_idx, ti], color="gray", ls=":", lw=1.0, label="Naive(60s前)")
            ax.plot(preds["GRU_Phys"][:, h_idx, ti], color=COLORS["pred"], lw=1.2, label="GRU_Phys (零样本)")
            ax.plot(preds["Ensemble_Phys"][:, h_idx, ti], color=COLORS["target"], ls="--", lw=1.2, label="Ensemble_Phys (零样本)")
            ax.set_ylabel(tname)
            g_skill = [r["skill_vs_naive_pct"] for r in sub_rows if r["model"] == "GRU_Phys" and r["target"] == tname and r["horizon_s"] == 60][0]
            e_skill = [r["skill_vs_naive_pct"] for r in sub_rows if r["model"] == "Ensemble_Phys" and r["target"] == tname and r["horizon_s"] == 60][0]
            ax.set_title(
                f"外部跨日期批次 {tag} (2025-01-26) — 提前 60 秒零样本预测 ({tname}) | "
                f"GRU_Phys Skill={g_skill:+.2f}%, Ensemble Skill={e_skill:+.2f}%",
                loc="left",
                fontweight="bold",
                fontsize=10,
            )
            ax.legend(loc="best", fontsize=8.5, ncol=4)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("逐秒测试样本序号 (1 Hz)")
        fig.tight_layout()
        fig.savefig(REAL_FIG / f"r4d_external_{tag}_60s.png", dpi=120)
        plt.close(fig)

    res_df = pd.DataFrame(all_rows)
    res_df.to_csv(REAL_REP / "r4d_external_1A_1B.csv", index=False, encoding="utf-8-sig")
    rep_text = "\n".join(report_lines)
    print(rep_text)
    save_real_report("r4d_external_1A_1B.txt", rep_text)
    print(f"外部测试图已保存 -> {REAL_FIG}")


if __name__ == "__main__":
    main()
