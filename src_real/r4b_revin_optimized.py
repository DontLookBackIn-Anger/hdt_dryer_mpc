#!/usr/bin/env python3
"""
=== 真实数据脚本 4B：RevIN 窗口相对化 + 正则化升级对比实验 ===

针对 r4_train_and_eval.py 中诊断出的两大根因对症下药，保留 r4 结果作为优化前基线：
  根因 1：输入窗口含全局绝对水平值，导致 4 月跨月测试批次（均值高 +0.8℃）被深度模型系统性下拉。
  → 解法 1：RevIN 锚点相对化输入 X_rel = X - X[:, -1:, :]，拼接一阶变化率 dX，
           让模型只学习过去 60 秒的「相对变化形态与速率」，100% 免疫跨批次/跨月基准漂移！

  根因 2：stride=5 样本仅 7,157 条，64 维深度网络容量偏大导致跨批次过拟合。
  → 解法 2：训练集改用最密滑窗 stride=1（约 3.5 万条样本，与 r4 相同的 12 批训练/2 批验证），
           验证/测试集保持与 r4 完全一致的 4 个 4 月批次（同口径横向对比！）；
           收紧网络维度 d_model=32、加大 dropout=0.25 与 weight_decay=5e-3、加入训练期微小高斯噪声，
           并采用步长加权损失 [0.5, 1.0, 1.5, 2.0] 聚焦 30s/60s 中长步 MPC 预测。

运行：
    cd ~/hdt_dryer_mpc
    /opt/conda_envs/dryer-ts/bin/python src_real/r4b_revin_optimized.py

产出（独立留痕，不覆盖 r4）：
    outputs_real/checkpoints/r4b_gru_best.pt
    outputs_real/checkpoints/r4b_tcn_sknet_transformer_best.pt
    outputs_real/reports/r4b_model_comparison.csv
    outputs_real/reports/r4b_before_after_compare.csv
    outputs_real/reports/r4b_train_report.txt
    outputs_real/figures/r4b_horizon_rmse_compare.png
    outputs_real/figures/r4b_test_predictions_60s.png
    outputs_real/figures/r4b_dl_loss_curves.png
"""
import sys
import time
import argparse
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
from torch.utils.data import TensorDataset, DataLoader

from real_common import REAL_PROC, REAL_OUT, REAL_FIG, REAL_REP, COLORS, calc_metrics, save_real_report
from r3_build_dataset import TARGET_COLS, FEATURE_COLS, build_windows_for_batch
from r4_train_and_eval import ResidualGRU, ImprovedTCNSKNetTransformer, predict_torch_model


# ---------------------------------------------------------------- 1. RevIN 锚点相对化特征变换
def to_revin_sequence(X_seq: np.ndarray) -> np.ndarray:
    """将 (N, T, F) 转换为跨批次基准漂移不变的序列表示 (N, T, 2F)：
    - 前 F 维：窗口锚点相对位移 X(τ) - X(t)（彻底消除跨月均值截距偏移！）
    - 后 F 维：一阶变化速率 X(τ) - X(τ-1)（强化瞬态动态与阀门动作特征）
    """
    x_rel = X_seq - X_seq[:, -1:, :]
    dx = np.zeros_like(X_seq)
    dx[:, 1:, :] = X_seq[:, 1:, :] - X_seq[:, :-1, :]
    return np.concatenate([x_rel, dx], axis=-1).astype(np.float32)


def build_revin_tabular(X_seq: np.ndarray) -> np.ndarray:
    """供 Ridge / LightGBM 使用的无绝对偏移相对化表格特征。"""
    x_rel = X_seq - X_seq[:, -1:, :]
    N, T, F = x_rel.shape
    lag_offsets = [2, 6, 16, 31, 46, T]  # 相对 t 的滞后位移
    lag_idx = [max(0, T - o) for o in lag_offsets]
    lag_feats = x_rel[:, lag_idx, :].reshape(N, -1)

    mean_rel = x_rel.mean(axis=1)
    std_all = x_rel.std(axis=1)
    short_diff = X_seq[:, -10:, :].mean(axis=1) - X_seq[:, -20:-10, :].mean(axis=1)
    mid_diff = X_seq[:, -20:, :].mean(axis=1) - X_seq[:, -40:-20, :].mean(axis=1)
    long_diff = X_seq[:, -15:, :].mean(axis=1) - X_seq[:, :15, :].mean(axis=1)
    return np.hstack([lag_feats, mean_rel, std_all, short_diff, mid_diff, long_diff]).astype(np.float32)


# ---------------------------------------------------------------- 2. 步长加权训练循环
def train_torch_weighted(
    model: nn.Module,
    X_tr: np.ndarray,
    dY_tr_norm: np.ndarray,
    X_va: np.ndarray,
    dY_va_norm: np.ndarray,
    horizon_weights: list[float],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    noise_std: float,
    device: torch.device,
    ckpt_path: Path,
    patience: int = 10,
) -> dict:
    model = model.to(device)
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(dY_tr_norm)),
        batch_size=batch_size,
        shuffle=True,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(dY_va_norm)),
        batch_size=batch_size * 2,
        shuffle=False,
    )

    h_w = torch.tensor(horizon_weights, dtype=torch.float32, device=device).view(1, -1, 1)
    h_w = h_w / h_w.mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05)
    huber = nn.SmoothL1Loss(beta=0.5, reduction="none")

    best_val = float("inf")
    wait = 0
    tr_hist, va_hist = [], []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            if noise_std > 0:
                bx = bx + torch.randn_like(bx) * noise_std
            optimizer.zero_grad()
            pred = model(bx)
            loss = (huber(pred, by) * h_w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(bx)
        tr_loss /= len(X_tr)
        scheduler.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                va_loss += (huber(model(bx), by) * h_w).mean().item() * len(bx)
        va_loss /= len(X_va)

        tr_hist.append(tr_loss)
        va_hist.append(va_loss)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"    Epoch {ep:3d}/{epochs} | train_loss={tr_loss:.5f} | val_loss={va_loss:.5f} | best={best_val:.5f}")
        if wait >= patience:
            print(f"    Early stopping @ epoch {ep} (连续 {patience} 轮验证集未改善)")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return {"train_loss": tr_hist, "val_loss": va_hist, "best_val": best_val}


# ---------------------------------------------------------------- 3. 主流程
def main():
    parser = argparse.ArgumentParser(description="RevIN 窗口相对化 + 正则化升级实验")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--train-stride", type=int, default=1, help="训练集滑窗步长，默认 1（最密滑窗）")
    parser.add_argument("--eval-stride", type=int, default=5, help="验证/测试滑窗步长，默认 5（与 r4 完全同口径）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = REAL_OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取 18 批次（严格沿用 r3 的 12 训练 / 2 验证 / 4 测试划分）
    batch_dir = REAL_PROC / "batches_named"
    csv_files = sorted(batch_dir.glob("*.csv"))
    train_files, val_files, test_files = csv_files[:12], csv_files[12:14], csv_files[14:]

    def _load_steady(paths):
        out = []
        for p in paths:
            df = pd.read_csv(p, encoding="utf-8-sig")
            out.append((p.stem, df[df["phase"] == "steady"].reset_index(drop=True)))
        return out

    train_batches = _load_steady(train_files)
    val_batches = _load_steady(val_files)
    test_batches = _load_steady(test_files)

    # 仅在训练集上 fit 截断与 RobustScaler
    train_concat = pd.concat([df[FEATURE_COLS] for _, df in train_batches], ignore_index=True)
    clip_low = train_concat.quantile(0.001).to_numpy(dtype=np.float32)
    clip_high = train_concat.quantile(0.999).to_numpy(dtype=np.float32)
    scaler = RobustScaler().fit(np.clip(train_concat.to_numpy(dtype=np.float32), clip_low, clip_high))

    input_size = 60
    horizons = [1, 15, 30, 60]
    horizon_weights = [0.5, 1.0, 1.5, 2.0]

    def _build_split(batches, stride):
        X_list, Yf_list, Yc_list, bid_list = [], [], [], []
        for bid, df in batches:
            raw_feat = df[FEATURE_COLS].to_numpy(dtype=np.float32)
            raw_targ = df[TARGET_COLS].to_numpy(dtype=np.float32)
            feat_scaled = scaler.transform(np.clip(raw_feat, clip_low, clip_high)).astype(np.float32)
            xs, yf, yc = build_windows_for_batch(feat_scaled, raw_targ, input_size, horizons, stride)
            if len(xs) > 0:
                X_list.append(xs)
                Yf_list.append(yf)
                Yc_list.append(yc)
                bid_list.extend([bid] * len(xs))
        return (
            np.concatenate(X_list, axis=0),
            np.concatenate(Yf_list, axis=0),
            np.concatenate(Yc_list, axis=0),
            np.asarray(bid_list),
        )

    X_tr_raw, Y_tr, Yc_tr, _ = _build_split(train_batches, args.train_stride)
    X_va_raw, Y_va, Yc_va, _ = _build_split(val_batches, args.eval_stride)
    X_te_raw, Y_te, Yc_te, B_te = _build_split(test_batches, args.eval_stride)

    # RevIN 锚点相对化变换（彻底消除跨月绝对水平漂移）
    X_tr_seq = to_revin_sequence(X_tr_raw)
    X_va_seq = to_revin_sequence(X_va_raw)
    X_te_seq = to_revin_sequence(X_te_raw)

    n_h, n_t = len(horizons), len(TARGET_COLS)
    dY_tr = Y_tr - Yc_tr[:, None, :]
    dY_va = Y_va - Yc_va[:, None, :]
    # 针对每个 (horizon, target) 独立标准化 ΔY，平衡 1s 与 60s 以及水分与温度的量级
    dy_std = dY_tr.std(axis=0, keepdims=True) + 1e-6  # (1, H, C)
    dY_tr_norm = (dY_tr / dy_std).astype(np.float32)
    dY_va_norm = (dY_va / dy_std).astype(np.float32)

    print("=" * 74)
    print(f"真实数据脚本 4B：RevIN 相对化 + 正则化升级实验 | Device: {device}")
    print(f"Train(stride={args.train_stride})={len(X_tr_seq)}, Val={len(X_va_seq)}, Test={len(X_te_seq)} | SeqFeat={X_tr_seq.shape[-1]}")
    print("=" * 74)

    preds_test: dict[str, np.ndarray] = {}
    train_times: dict[str, float] = {}
    dl_histories: dict[str, dict] = {}

    # 1. Naive
    preds_test["Naive"] = np.repeat(Yc_te[:, None, :], n_h, axis=1)
    train_times["Naive"] = 0.0

    # 相对化表格特征（为了速度，表格模型可用 stride=2 子采样训练集，已含 ~1.8 万样本）
    tab_step = 2 if args.train_stride == 1 else 1
    X_tr_tab = build_revin_tabular(X_tr_raw[::tab_step])
    dY_tr_tab = dY_tr[::tab_step]
    dY_tr_tab_norm = dY_tr_norm[::tab_step]
    X_va_tab = build_revin_tabular(X_va_raw)
    X_te_tab = build_revin_tabular(X_te_raw)

    # 2. Ridge_ARX (RevIN)
    print("\n[1/4] 训练 Ridge_ARX (RevIN 相对化)...")
    t0 = time.time()
    ridge = Ridge(alpha=50.0)
    ridge.fit(X_tr_tab, dY_tr_tab_norm.reshape(len(X_tr_tab), -1))
    dY_pred_ridge = ridge.predict(X_te_tab).reshape(-1, n_h, n_t) * dy_std
    preds_test["Ridge_ARX_RevIN"] = Yc_te[:, None, :] + dY_pred_ridge
    train_times["Ridge_ARX_RevIN"] = time.time() - t0
    print(f"    完成，耗时 {train_times['Ridge_ARX_RevIN']:.2f}s")

    # 3. LightGBM (RevIN)
    print("\n[2/4] 训练 LightGBM (RevIN 相对化)...")
    t0 = time.time()
    lgb_pred_delta = np.zeros((len(X_te_tab), n_h, n_t), dtype=np.float32)
    for hi, h in enumerate(horizons):
        for ti, tname in enumerate(target_cols := TARGET_COLS):
            reg = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=40,
                reg_alpha=1.0,
                reg_lambda=5.0,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=args.seed,
                n_jobs=-1,
                verbose=-1,
            )
            reg.fit(
                X_tr_tab,
                dY_tr_tab[:, hi, ti],
                eval_set=[(X_va_tab, dY_va[:, hi, ti])],
                callbacks=[lgb.early_stopping(35, verbose=False)],
            )
            lgb_pred_delta[:, hi, ti] = reg.predict(X_te_tab)
    preds_test["LightGBM_RevIN"] = Yc_te[:, None, :] + lgb_pred_delta
    train_times["LightGBM_RevIN"] = time.time() - t0
    print(f"    完成，耗时 {train_times['LightGBM_RevIN']:.2f}s")

    # 4. Residual GRU (RevIN + 轻量正则化)
    print("\n[3/4] 训练 Residual_GRU_RevIN (d=32, dropout=0.25, stride=1)...")
    t0 = time.time()
    gru = ResidualGRU(
        in_dim=X_tr_seq.shape[-1], hidden_dim=32, n_horizons=n_h, n_targets=n_t,
        num_layers=2, dropout=0.25,
    )
    dl_histories["Residual_GRU_RevIN"] = train_torch_weighted(
        gru, X_tr_seq, dY_tr_norm, X_va_seq, dY_va_norm,
        horizon_weights=horizon_weights, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=5e-3, noise_std=0.02, device=device,
        ckpt_path=ckpt_dir / "r4b_gru_best.pt", patience=10,
    )
    dY_pred_gru = predict_torch_model(gru, X_te_seq, device) * dy_std
    preds_test["Residual_GRU_RevIN"] = Yc_te[:, None, :] + dY_pred_gru
    train_times["Residual_GRU_RevIN"] = time.time() - t0
    print(f"    完成，耗时 {train_times['Residual_GRU_RevIN']:.2f}s")

    # 5. Improved TCN-SKNet-Transformer (RevIN + 轻量正则化)
    print("\n[4/4] 训练 TCN_SKNet_Trans_RevIN (d=32, dropout=0.25, stride=1)...")
    t0 = time.time()
    hybrid = ImprovedTCNSKNetTransformer(
        in_dim=X_tr_seq.shape[-1], n_horizons=n_h, n_targets=n_t,
        d_model=32, nhead=4, num_layers=2, dropout=0.25,
    )
    dl_histories["TCN_SKNet_Trans_RevIN"] = train_torch_weighted(
        hybrid, X_tr_seq, dY_tr_norm, X_va_seq, dY_va_norm,
        horizon_weights=horizon_weights, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=5e-3, noise_std=0.02, device=device,
        ckpt_path=ckpt_dir / "r4b_tcn_sknet_transformer_best.pt", patience=10,
    )
    dY_pred_hyb = predict_torch_model(hybrid, X_te_seq, device) * dy_std
    preds_test["TCN_SKNet_Trans_RevIN"] = Yc_te[:, None, :] + dY_pred_hyb
    train_times["TCN_SKNet_Trans_RevIN"] = time.time() - t0
    print(f"    完成，耗时 {train_times['TCN_SKNet_Trans_RevIN']:.2f}s")

    # ---------------------------------------------------------------- 4. 统一口径评估
    rows = []
    naive_rmse = {}
    for hi, h in enumerate(horizons):
        for ti, tname in enumerate(TARGET_COLS):
            naive_rmse[(h, tname)] = calc_metrics(Y_te[:, hi, ti], preds_test["Naive"][:, hi, ti])["rmse"]

    for mname, y_hat in preds_test.items():
        for hi, h in enumerate(horizons):
            for ti, tname in enumerate(TARGET_COLS):
                m = calc_metrics(Y_te[:, hi, ti], y_hat[:, hi, ti])
                skill = (1.0 - m["rmse"] / (naive_rmse[(h, tname)] + 1e-12)) * 100.0
                rows.append({
                    "model": mname,
                    "target": tname,
                    "horizon_s": h,
                    "rmse": round(m["rmse"], 5),
                    "mae": round(m["mae"], 5),
                    "r2": round(m["r2"], 4),
                    "skill_vs_naive_pct": round(skill, 2),
                    "train_sec": round(train_times[mname], 2),
                })

    res_df = pd.DataFrame(rows)
    res_df.to_csv(REAL_REP / "r4b_model_comparison.csv", index=False, encoding="utf-8-sig")

    # 与优化前 r4 做直接对比表（若 r4_model_comparison.csv 存在）
    compare_lines = []
    r4_csv = REAL_REP / "r4_model_comparison.csv"
    if r4_csv.exists():
        old_df = pd.read_csv(r4_csv)
        pair_map = [
            ("Ridge_ARX", "Ridge_ARX_RevIN"),
            ("LightGBM", "LightGBM_RevIN"),
            ("Residual_GRU", "Residual_GRU_RevIN"),
            ("TCN_SKNet_Trans", "TCN_SKNet_Trans_RevIN"),
        ]
        comp_rows = []
        for old_m, new_m in pair_map:
            for tname in TARGET_COLS:
                for h in [30, 60]:
                    o = old_df[(old_df["model"] == old_m) & (old_df["target"] == tname) & (old_df["horizon_s"] == h)].iloc[0]
                    n = res_df[(res_df["model"] == new_m) & (res_df["target"] == tname) & (res_df["horizon_s"] == h)].iloc[0]
                    comp_rows.append({
                        "model": old_m,
                        "target": tname,
                        "horizon_s": h,
                        "r4_rmse": o["rmse"],
                        "r4b_rmse": n["rmse"],
                        "r4_skill_pct": o["skill_vs_naive_pct"],
                        "r4b_skill_pct": n["skill_vs_naive_pct"],
                        "skill_gain_pct": round(n["skill_vs_naive_pct"] - o["skill_vs_naive_pct"], 2),
                    })
        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(REAL_REP / "r4b_before_after_compare.csv", index=False, encoding="utf-8-sig")
        compare_lines.append("\n【优化前 (r4) vs 优化后 (r4b RevIN) 关键步长 (30s / 60s) 对比】")
        compare_lines.append(comp_df.to_string(index=False))

    # 物理方向一致性检验（对过去 20s 蒸汽施加 +1 IQR 阶跃，检查 60s 预测响应）
    steam_idx = FEATURE_COLS.index("steam_flow")
    X_pert_raw = X_te_raw[:500].copy()
    X_pert_raw[:, -20:, steam_idx] += 1.0
    X_pert_seq = to_revin_sequence(X_pert_raw)
    phys_lines = ["\n【物理方向一致性检验：过去 20s 补偿蒸汽 +1 IQR 对未来 60s 预测的平均响应】"]
    for mname, mdl in [("Residual_GRU_RevIN", gru), ("TCN_SKNet_Trans_RevIN", hybrid)]:
        base_p = predict_torch_model(mdl, X_te_seq[:500], device) * dy_std
        pert_p = predict_torch_model(mdl, X_pert_seq, device) * dy_std
        diff_60 = (pert_p - base_p)[:, -1, :].mean(axis=0)
        phys_lines.append(
            f"  {mname:22s} | Δ出口含水率(60s) = {diff_60[0]:+.5f}% | Δ出口温度(60s) = {diff_60[1]:+.4f}℃"
        )

    # ---------------------------------------------------------------- 5. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ti, tname in enumerate(TARGET_COLS):
        ax = axes[ti]
        for mname in preds_test.keys():
            sub = res_df[(res_df["model"] == mname) & (res_df["target"] == tname)]
            ax.plot(sub["horizon_s"], sub["rmse"], marker="o", lw=1.8, label=mname)
        ax.set_title(f"{tname}：RevIN 优化后各步长 RMSE 对比", fontweight="bold")
        ax.set_xlabel("预测步长 Horizon (秒)")
        ax.set_ylabel("RMSE")
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("真实 18 批次跨月泛化测试（r4b RevIN + 正则化优化版）", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REAL_FIG / "r4b_horizon_rmse_compare.png", dpi=120)
    plt.close(fig)

    uniq_b, counts_b = np.unique(B_te, return_counts=True)
    show_bid = uniq_b[int(np.argmax(counts_b))]
    mask_b = B_te == show_bid
    h_idx = len(horizons) - 1
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    for ti, tname in enumerate(TARGET_COLS):
        ax = axes[ti]
        ax.plot(Y_te[mask_b, h_idx, ti], color="black", lw=1.4, label="真实值")
        ax.plot(preds_test["Naive"][mask_b, h_idx, ti], color="gray", ls=":", lw=1.0, label="Naive(60s前)")
        ax.plot(preds_test["LightGBM_RevIN"][mask_b, h_idx, ti], color=COLORS["good"], lw=1.1, alpha=0.9, label="LightGBM_RevIN")
        ax.plot(preds_test["Residual_GRU_RevIN"][mask_b, h_idx, ti], color=COLORS["pred"], lw=1.1, alpha=0.9, label="GRU_RevIN")
        ax.plot(preds_test["TCN_SKNet_Trans_RevIN"][mask_b, h_idx, ti], color=COLORS["target"], ls="--", lw=1.2, label="TCN_SKNet_Trans_RevIN")
        ax.set_ylabel(tname)
        ax.set_title(f"测试批次 {show_bid} — RevIN 优化后提前 60 秒预测 ({tname})", loc="left", fontweight="bold")
        ax.legend(loc="best", fontsize=8.5, ncol=5)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("滑窗样本序号")
    fig.tight_layout()
    fig.savefig(REAL_FIG / "r4b_test_predictions_60s.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (mname, hist) in zip(axes, dl_histories.items()):
        ax.plot(hist["train_loss"], color=COLORS["target"], lw=1.5, label="Train Loss")
        ax.plot(hist["val_loss"], color=COLORS["good"], lw=1.5, label="Val Loss")
        ax.set_title(f"{mname} 训练与验证损失曲线", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted SmoothL1 Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(REAL_FIG / "r4b_dl_loss_curves.png", dpi=120)
    plt.close(fig)

    rep_lines = [
        "=" * 80,
        f"真实数据脚本 4B：RevIN 窗口相对化 + 正则化升级评估报告 (Device={device})",
        "=" * 80,
        res_df.to_string(index=False),
        *compare_lines,
        *phys_lines,
        "",
        f"详细指标 CSV -> {REAL_REP / 'r4b_model_comparison.csv'}",
        f"前后对比 CSV -> {REAL_REP / 'r4b_before_after_compare.csv'}",
        f"图表已保存   -> {REAL_FIG}",
    ]
    rep_text = "\n".join(rep_lines)
    print("\n" + rep_text)
    save_real_report("r4b_train_report.txt", rep_text)


if __name__ == "__main__":
    main()
