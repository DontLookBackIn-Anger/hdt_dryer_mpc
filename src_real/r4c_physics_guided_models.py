#!/usr/bin/env python3
"""
=== 真实数据脚本 4C：物理单调性约束 + 分目标专精 + Patch-TCN-SKNet-Transformer ===

在 r4b（RevIN 消除跨月漂移）基础上，针对模型预测精度与物理一致性进行 4 项深度打磨：
  1. 分目标专精建模（Decoupled Specialists）：
     出口含水率（低信噪比、0.01% 量化台阶）与出口温度（高信噪比、快动态）分开训练，
     各自独立早停与调优，避免共享主干时互相牵制。
  2. 多尺度平滑 RevIN 特征（专治 0.01% 量化噪声）：
     输入拼接 [锚点相对位移 X(τ)-X(t), 5秒滑动平滑相对位移, 5秒平滑一阶趋势]，
     显著提高水分通道信噪比。
  3. 物理单调性正则化（PINN-style Monotonicity Loss）：
     训练期对蒸汽通道施加可微正向微扰，惩罚负响应 ReLU(margin - Δŷ_steam+)，
     彻底根治闭环 PID 伪相关导致的 GRU 水分负响应问题！
  4. Patch 化 TCN-SKNet-Transformer + DLinear 强基线：
     - 将 TCN+SKNet 输出按 5s 步长 Patch 压缩（60 步 → 12 个 Patch Token），
       注意力矩阵缩小 25 倍，根治原 60-token Transformer 过拟合；
     - 引入工业时序强基线 DLinear（趋势/季节分解线性模型）；
     - 验证加权融合（Ensemble）能否进一步突破单模型上限。

评估口径：与 r4、r4b 完全相同的 4 个 4 月跨月测试批次（2,242 窗口），严格零泄露。

运行：
    cd ~/hdt_dryer_mpc
    /opt/conda_envs/dryer-ts/bin/python src_real/r4c_physics_guided_models.py

产出（独立留痕，不覆盖 r4 / r4b）：
    outputs_real/checkpoints/r4c_*.pt
    outputs_real/reports/r4c_model_comparison.csv
    outputs_real/reports/r4c_evolution_r4_r4b_r4c.csv
    outputs_real/reports/r4c_train_report.txt
    outputs_real/figures/r4c_horizon_skill_compare.png
    outputs_real/figures/r4c_test_predictions_60s.png
    outputs_real/figures/r4c_dl_loss_curves.png
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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from real_common import REAL_PROC, REAL_OUT, REAL_FIG, REAL_REP, COLORS, calc_metrics, save_real_report
from r3_build_dataset import TARGET_COLS, FEATURE_COLS, build_windows_for_batch
from r4_train_and_eval import CausalConv1dBlock, SKNet1D, predict_torch_model


# ---------------------------------------------------------------- 1. 多尺度平滑 RevIN 特征
def moving_avg_1d(arr: np.ndarray, k: int = 5) -> np.ndarray:
    """沿时间轴 axis=1 做因果滑动平均（保持长度不变，去 0.01% 量化毛刺）。"""
    N, T, F = arr.shape
    pad = np.repeat(arr[:, :1, :], k - 1, axis=1)
    padded = np.concatenate([pad, arr], axis=1)
    csum = np.cumsum(padded, axis=1)
    csum_shift = np.concatenate([np.zeros((N, 1, F), dtype=arr.dtype), csum[:, :-k, :]], axis=1)
    return (csum[:, k - 1 :, :] - csum_shift) / float(k)


def to_multiscale_revin(X_seq: np.ndarray) -> np.ndarray:
    """将 (N, T, F) 变换为 (N, T, 3F) 多尺度无漂移特征：
    - [0:F]   : 原始锚点相对位移 X(τ) - X(t)
    - [F:2F]  : 5秒因果平滑相对位移 MA5(X)(τ) - MA5(X)(t)
    - [2F:3F] : 5秒平滑变化趋势（当前5s均值 - 5s前均值），抑制单秒量化噪声
    """
    x_rel = X_seq - X_seq[:, -1:, :]
    x_ma5 = moving_avg_1d(X_seq, k=5)
    x_ma5_rel = x_ma5 - x_ma5[:, -1:, :]
    slope5 = np.zeros_like(x_ma5)
    slope5[:, 5:, :] = (x_ma5[:, 5:, :] - x_ma5[:, :-5, :]) / 5.0
    return np.concatenate([x_rel, x_ma5_rel, slope5], axis=-1).astype(np.float32)


# ---------------------------------------------------------------- 2. 模型定义（单目标专精）
class DLinearSpecialist(nn.Module):
    """DLinear（趋势 + 残差分解线性模型），工业时序强基线，极抗过拟合。"""

    def __init__(self, seq_len: int, in_dim: int, n_horizons: int, kernel_size: int = 15):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.linear_trend = nn.Linear(seq_len * in_dim, n_horizons)
        self.linear_season = nn.Linear(seq_len * in_dim, n_horizons)
        self.drop = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        xt = x.transpose(1, 2)
        trend = nn.functional.avg_pool1d(xt, kernel_size=self.kernel_size, stride=1, padding=self.pad)
        if trend.size(-1) > xt.size(-1):
            trend = trend[:, :, : xt.size(-1)]
        season = xt - trend
        out = self.linear_trend(self.drop(trend.reshape(x.size(0), -1))) + \
              self.linear_season(self.drop(season.reshape(x.size(0), -1)))
        return out  # (B, H)


class PhysicsGRUSpecialist(nn.Module):
    """单目标专精轻量残差 GRU。"""

    def __init__(self, in_dim: int, hidden_dim: int, n_horizons: int, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        h_last = self.norm(out[:, -1, :])
        h_mean = self.norm(out.mean(dim=1))
        return self.head(torch.cat([h_last, h_mean], dim=-1))


class PatchTCNSKNetTransformerSpecialist(nn.Module):
    """Patch 化升级版 TCN + SK-Net + 2层 Transformer（专治 60-token 过拟合）。

    改进要点：
      1. 因果膨胀 TCN (d=1,2,4,8) 提取 60s 多尺度滞后特征；
      2. 真多尺度 SK-Net (k=3/5) 自适应选择时间感受野；
      3. Patch 压缩层（AvgPool1d k=5, s=5）：将 60 个 1s 噪声 token 压缩为 12 个 5s 语义 Patch，
         注意力矩阵从 3600 降至 144（缩小 25 倍！），大幅抑噪并消除过拟合；
      4. 2 层 TransformerEncoder + 正弦位置编码建模 12 个 Patch 的全局依赖。
    """

    def __init__(
        self,
        in_dim: int,
        n_horizons: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        patch_size: int = 5,
        dropout: float = 0.30,
        max_patches: int = 64,
    ):
        super().__init__()
        self.tcn = nn.Sequential(
            CausalConv1dBlock(in_dim, d_model, kernel_size=3, dilation=1, dropout=dropout),
            CausalConv1dBlock(d_model, d_model, kernel_size=3, dilation=2, dropout=dropout),
            CausalConv1dBlock(d_model, d_model, kernel_size=3, dilation=4, dropout=dropout),
            CausalConv1dBlock(d_model, d_model, kernel_size=3, dilation=8, dropout=dropout),
        )
        self.sknet = SKNet1D(d_model, reduction=4)
        self.patch_pool = nn.AvgPool1d(kernel_size=patch_size, stride=patch_size)

        pe = torch.zeros(max_patches, d_model)
        pos = torch.arange(0, max_patches, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x.transpose(1, 2))      # (B, C, 60)
        h = self.sknet(h)                     # (B, C, 60)
        h = self.patch_pool(h).transpose(1, 2)  # (B, 12, C)
        h = h + self.pe[:, : h.size(1), :]
        h = self.norm(self.transformer(h))
        feat = torch.cat([h[:, -1, :], h.mean(dim=1)], dim=-1)
        return self.head(feat)


# ---------------------------------------------------------------- 3. 物理单调性正则训练循环
def train_specialist_with_physics(
    model: nn.Module,
    X_tr: np.ndarray,
    dY_tr_norm: np.ndarray,
    X_va: np.ndarray,
    dY_va_norm: np.ndarray,
    steam_channels: list[int],
    horizon_weights: list[float],
    phys_weight: float,
    phys_margin: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    noise_std: float,
    device: torch.device,
    ckpt_path: Path,
    patience: int = 10,
) -> dict:
    """带物理单调性惩罚的专精训练：
    对蒸汽通道在过去 20s 施加 +0.5 单位正向微扰（体现在锚点相对位移上为前 40s 下降、后 20s 上升），
    强制模型在 30s / 60s 处输出非负响应，根治闭环 PID 伪相关！
    """
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

    h_w = torch.tensor(horizon_weights, dtype=torch.float32, device=device).view(1, -1)
    h_w = h_w / h_w.mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05)
    huber = nn.SmoothL1Loss(beta=0.5, reduction="none")

    best_val = float("inf")
    wait = 0
    tr_hist, va_hist = [], []

    # 预构造蒸汽阶跃微扰掩码（对前 40s 的相对位移减去 0.5，等价于后 20s 蒸汽上升 +0.5）
    pert_mask = torch.zeros((1, X_tr.shape[1], X_tr.shape[2]), dtype=torch.float32, device=device)
    for ch in steam_channels:
        pert_mask[:, :-20, ch] = -0.5

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            if noise_std > 0:
                bx_in = bx + torch.randn_like(bx) * noise_std
            else:
                bx_in = bx

            optimizer.zero_grad()
            pred = model(bx_in)
            data_loss = (huber(pred, by) * h_w).mean()

            # 物理单调性正则：蒸汽正向阶跃 -> 30s/60s 预测必须 >= phys_margin
            if phys_weight > 0:
                pred_pert = model(bx_in + pert_mask)
                delta_phys = (pred_pert - pred)[:, -2:]  # 30s & 60s
                phys_loss = torch.relu(phys_margin - delta_phys).mean()
                loss = data_loss + phys_weight * phys_loss
            else:
                loss = data_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += data_loss.item() * len(bx)
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
            print(f"      Epoch {ep:3d}/{epochs} | train={tr_loss:.5f} | val={va_loss:.5f} | best={best_val:.5f}")
        if wait >= patience:
            print(f"      Early stopping @ epoch {ep}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return {"train_loss": tr_hist, "val_loss": va_hist, "best_val": best_val}


# ---------------------------------------------------------------- 4. 主流程
def main():
    parser = argparse.ArgumentParser(description="物理单调性约束 + 分目标专精 + Patch-TCN-SKNet-Transformer")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = REAL_OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. 严格沿用 r3/r4/r4b 的 12 训练 / 2 验证 / 4 测试批次划分
    batch_dir = REAL_PROC / "batches_named"
    csv_files = sorted(batch_dir.glob("*.csv"))
    train_files, val_files, test_files = csv_files[:12], csv_files[12:14], csv_files[14:]

    def _load_steady(paths):
        return [(p.stem, pd.read_csv(p, encoding="utf-8-sig").query("phase == 'steady'").reset_index(drop=True)) for p in paths]

    train_batches = _load_steady(train_files)
    val_batches = _load_steady(val_files)
    test_batches = _load_steady(test_files)

    train_concat = pd.concat([df[FEATURE_COLS] for _, df in train_batches], ignore_index=True)
    clip_low = train_concat.quantile(0.001).to_numpy(dtype=np.float32)
    clip_high = train_concat.quantile(0.999).to_numpy(dtype=np.float32)
    scaler = RobustScaler().fit(np.clip(train_concat.to_numpy(dtype=np.float32), clip_low, clip_high))

    input_size = 60
    horizons = [1, 15, 30, 60]
    horizon_weights = [0.5, 1.0, 1.5, 2.0]
    n_h, n_t = len(horizons), len(TARGET_COLS)

    def _build_split(batches, stride):
        X_list, Yf_list, Yc_list, bid_list = [], [], [], []
        for bid, df in batches:
            raw_feat = df[FEATURE_COLS].to_numpy(dtype=np.float32)
            raw_targ = df[TARGET_COLS].to_numpy(dtype=np.float32)
            feat_scaled = scaler.transform(np.clip(raw_feat, clip_low, clip_high)).astype(np.float32)
            xs, yf, yc = build_windows_for_batch(feat_scaled, raw_targ, input_size, horizons, stride)
            if len(xs) > 0:
                X_list.append(xs); Yf_list.append(yf); Yc_list.append(yc)
                bid_list.extend([bid] * len(xs))
        return np.concatenate(X_list), np.concatenate(Yf_list), np.concatenate(Yc_list), np.asarray(bid_list)

    X_tr_raw, Y_tr, Yc_tr, _ = _build_split(train_batches, stride=1)
    X_va_raw, Y_va, Yc_va, _ = _build_split(val_batches, stride=5)
    X_te_raw, Y_te, Yc_te, B_te = _build_split(test_batches, stride=5)

    X_tr_ms = to_multiscale_revin(X_tr_raw)
    X_va_ms = to_multiscale_revin(X_va_raw)
    X_te_ms = to_multiscale_revin(X_te_raw)

    dY_tr = Y_tr - Yc_tr[:, None, :]
    dY_va = Y_va - Yc_va[:, None, :]
    dy_std = dY_tr.std(axis=0, keepdims=True) + 1e-6  # (1, H, C)
    dY_tr_norm = (dY_tr / dy_std).astype(np.float32)
    dY_va_norm = (dY_va / dy_std).astype(np.float32)

    F_base = len(FEATURE_COLS)
    steam_idx = FEATURE_COLS.index("steam_flow")
    # 在 3F 特征中，原始相对位移与平滑相对位移的 steam_flow 通道
    steam_channels = [steam_idx, F_base + steam_idx]

    print("=" * 78)
    print(f"真实数据脚本 4C：物理单调性约束 + 分目标专精 + Patch-TCN-SKNet-Transformer | {device}")
    print(f"Train(stride=1)={len(X_tr_ms)}, Val={len(X_va_ms)}, Test={len(X_te_ms)} | FeatDim={X_tr_ms.shape[-1]}")
    print("=" * 78)

    preds_test = {
        "Naive": np.repeat(Yc_te[:, None, :], n_h, axis=1),
        "DLinear_Phys": np.zeros_like(Y_te),
        "GRU_Phys": np.zeros_like(Y_te),
        "Patch_TCN_SKNet_Trans_Phys": np.zeros_like(Y_te),
    }
    preds_val = {
        "DLinear_Phys": np.zeros_like(Y_va),
        "GRU_Phys": np.zeros_like(Y_va),
        "Patch_TCN_SKNet_Trans_Phys": np.zeros_like(Y_va),
    }
    train_times = {k: 0.0 for k in preds_test}
    dl_histories = {}
    trained_models = {}

    # 针对每个目标分别训练专精模型
    for ti, tname in enumerate(TARGET_COLS):
        print(f"\n>>> 专精训练目标 [{ti+1}/{n_t}]: {tname}")
        y_tr_t = dY_tr_norm[:, :, ti]
        y_va_t = dY_va_norm[:, :, ti]
        std_t = dy_std[0, :, ti]  # (H,)
        # 物理正则强度：水分通道受闭环 PID 伪相关影响更重，给予更强单调性约束
        p_w = 0.35 if tname == "outlet_moisture" else 0.15
        p_m = 0.05 if tname == "outlet_moisture" else 0.08

        specs = [
            ("DLinear_Phys", lambda: DLinearSpecialist(input_size, X_tr_ms.shape[-1], n_h, kernel_size=15), 1e-3, 1e-2),
            ("GRU_Phys", lambda: PhysicsGRUSpecialist(X_tr_ms.shape[-1], hidden_dim=32, n_horizons=n_h, num_layers=2, dropout=0.25), args.lr, 5e-3),
            ("Patch_TCN_SKNet_Trans_Phys", lambda: PatchTCNSKNetTransformerSpecialist(
                X_tr_ms.shape[-1], n_horizons=n_h, d_model=32, nhead=4, num_layers=2, patch_size=5, dropout=0.30
            ), args.lr, 8e-3),
        ]

        for mname, ctor, lr_m, wd_m in specs:
            print(f"  [训练] {mname} ({tname})...")
            t0 = time.time()
            mdl = ctor()
            ckpt = ckpt_dir / f"r4c_{mname}_{tname}.pt"
            hist = train_specialist_with_physics(
                mdl, X_tr_ms, y_tr_t, X_va_ms, y_va_t,
                steam_channels=steam_channels,
                horizon_weights=horizon_weights,
                phys_weight=p_w,
                phys_margin=p_m,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=lr_m,
                weight_decay=wd_m,
                noise_std=0.02,
                device=device,
                ckpt_path=ckpt,
                patience=10,
            )
            train_times[mname] += time.time() - t0
            dl_histories[f"{mname}_{tname}"] = hist
            trained_models[(mname, tname)] = mdl

            va_delta = predict_torch_model(mdl, X_va_ms, device) * std_t[None, :]
            te_delta = predict_torch_model(mdl, X_te_ms, device) * std_t[None, :]
            preds_val[mname][:, :, ti] = Yc_va[:, ti : ti + 1] + va_delta
            preds_test[mname][:, :, ti] = Yc_te[:, ti : ti + 1] + te_delta

    # 构造验证集加权融合模型（在验证集上寻优非负权重，绝不碰测试集）
    print("\n>>> 在验证集上构造物理一致集成模型 (Ensemble_Phys)...")
    ens_test = np.zeros_like(Y_te)
    base_names = ["DLinear_Phys", "GRU_Phys", "Patch_TCN_SKNet_Trans_Phys"]
    weight_notes = []
    for ti, tname in enumerate(TARGET_COLS):
        # 在验证集 30s & 60s 上按反误差平方自动定权
        inv_mse = []
        for mname in base_names:
            err = preds_val[mname][:, -2:, ti] - Y_va[:, -2:, ti]
            inv_mse.append(1.0 / (float(np.mean(err ** 2)) + 1e-12))
        w = np.array(inv_mse) / np.sum(inv_mse)
        weight_notes.append(f"  {tname:16s} 融合权重: " + ", ".join(f"{m}={wi:.2f}" for m, wi in zip(base_names, w)))
        for mname, wi in zip(base_names, w):
            ens_test[:, :, ti] += wi * preds_test[mname][:, :, ti]
    preds_test["Ensemble_Phys"] = ens_test
    train_times["Ensemble_Phys"] = sum(train_times[m] for m in base_names)

    # ---------------------------------------------------------------- 5. 统一口径评估
    rows = []
    naive_rmse = {
        (h, tname): calc_metrics(Y_te[:, hi, ti], preds_test["Naive"][:, hi, ti])["rmse"]
        for hi, h in enumerate(horizons)
        for ti, tname in enumerate(TARGET_COLS)
    }

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
    res_df.to_csv(REAL_REP / "r4c_model_comparison.csv", index=False, encoding="utf-8-sig")

    # 物理方向一致性检验（测试集前 500 窗口，过去 20s 蒸汽 +1 IQR）
    X_pert_raw = X_te_raw[:500].copy()
    X_pert_raw[:, -20:, steam_idx] += 1.0
    X_pert_ms = to_multiscale_revin(X_pert_raw)
    phys_lines = ["\n【物理方向一致性检验：过去 20s 补偿蒸汽 +1 IQR 对未来 60s 预测的平均响应】"]
    for mname in base_names:
        resp = []
        for ti, tname in enumerate(TARGET_COLS):
            mdl = trained_models[(mname, tname)]
            std_60 = dy_std[0, -1, ti]
            b_p = predict_torch_model(mdl, X_te_ms[:500], device)[:, -1] * std_60
            p_p = predict_torch_model(mdl, X_pert_ms, device)[:, -1] * std_60
            resp.append(float((p_p - b_p).mean()))
        status = "✅ 双正通过" if (resp[0] > 0 and resp[1] > 0) else "⚠️ 需关注"
        phys_lines.append(
            f"  {mname:28s} | Δ水分(60s)={resp[0]:+.5f}% | Δ温度(60s)={resp[1]:+.4f}℃ | {status}"
        )

    # 三代演进总表（r4 -> r4b -> r4c）
    evo_lines = []
    r4_p, r4b_p = REAL_REP / "r4_model_comparison.csv", REAL_REP / "r4b_model_comparison.csv"
    if r4_p.exists() and r4b_p.exists():
        d4 = pd.read_csv(r4_p)
        d4b = pd.read_csv(r4b_p)
        evo_rows = []
        pairs = [
            ("GRU", "Residual_GRU", "Residual_GRU_RevIN", "GRU_Phys"),
            ("TCN_SKNet_Trans", "TCN_SKNet_Trans", "TCN_SKNet_Trans_RevIN", "Patch_TCN_SKNet_Trans_Phys"),
        ]
        for label, m4, m4b, m4c in pairs:
            for tname in TARGET_COLS:
                for h in [15, 30, 60]:
                    s4 = d4[(d4["model"] == m4) & (d4["target"] == tname) & (d4["horizon_s"] == h)]["skill_vs_naive_pct"].iloc[0]
                    s4b = d4b[(d4b["model"] == m4b) & (d4b["target"] == tname) & (d4b["horizon_s"] == h)]["skill_vs_naive_pct"].iloc[0]
                    s4c = res_df[(res_df["model"] == m4c) & (res_df["target"] == tname) & (res_df["horizon_s"] == h)]["skill_vs_naive_pct"].iloc[0]
                    evo_rows.append({
                        "family": label,
                        "target": tname,
                        "horizon_s": h,
                        "r4_v1_skill": s4,
                        "r4b_revin_skill": s4b,
                        "r4c_phys_skill": s4c,
                        "total_gain": round(s4c - s4, 2),
                    })
        evo_df = pd.DataFrame(evo_rows)
        evo_df.to_csv(REAL_REP / "r4c_evolution_r4_r4b_r4c.csv", index=False, encoding="utf-8-sig")
        evo_lines.append("\n【三代模型迭代留痕：r4 (初版) -> r4b (RevIN) -> r4c (物理约束+专精+Patch) Skill%】")
        evo_lines.append(evo_df.to_string(index=False))

    # ---------------------------------------------------------------- 6. 绘图
    # 图 1：Skill vs Naive 柱状/折线对比
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ti, tname in enumerate(TARGET_COLS):
        ax = axes[ti]
        for mname in ["DLinear_Phys", "GRU_Phys", "Patch_TCN_SKNet_Trans_Phys", "Ensemble_Phys"]:
            sub = res_df[(res_df["model"] == mname) & (res_df["target"] == tname)]
            ax.plot(sub["horizon_s"], sub["skill_vs_naive_pct"], marker="o", lw=2.0, label=mname)
        ax.axhline(0.0, color="gray", ls="--", lw=1.0, label="Naive 基线 (0%)")
        ax.set_title(f"{tname}：相对 Naive 提升率 Skill (%)（越高越好）", fontweight="bold")
        ax.set_xlabel("预测步长 Horizon (秒)")
        ax.set_ylabel("Skill vs Naive (%)")
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5)
    fig.suptitle("r4c 物理引导 + 分目标专精模型：跨月 4 批次多步预测 Skill 对比", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REAL_FIG / "r4c_horizon_skill_compare.png", dpi=120)
    plt.close(fig)

    # 图 2：测试批次 60s 预测曲线
    uniq_b, counts_b = np.unique(B_te, return_counts=True)
    show_bid = uniq_b[int(np.argmax(counts_b))]
    mask_b = B_te == show_bid
    h_idx = len(horizons) - 1
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    for ti, tname in enumerate(TARGET_COLS):
        ax = axes[ti]
        ax.plot(Y_te[mask_b, h_idx, ti], color="black", lw=1.4, label="真实值")
        ax.plot(preds_test["DLinear_Phys"][mask_b, h_idx, ti], color=COLORS["good"], lw=1.1, alpha=0.9, label="DLinear_Phys")
        ax.plot(preds_test["GRU_Phys"][mask_b, h_idx, ti], color=COLORS["pred"], lw=1.1, alpha=0.9, label="GRU_Phys")
        ax.plot(preds_test["Patch_TCN_SKNet_Trans_Phys"][mask_b, h_idx, ti], color=COLORS["warn"], lw=1.1, alpha=0.9, label="Patch_TCN_SKNet_Trans")
        ax.plot(preds_test["Ensemble_Phys"][mask_b, h_idx, ti], color=COLORS["target"], ls="--", lw=1.3, label="Ensemble_Phys")
        ax.set_ylabel(tname)
        ax.set_title(f"测试批次 {show_bid} — r4c 物理引导模型提前 60 秒预测 ({tname})", loc="left", fontweight="bold")
        ax.legend(loc="best", fontsize=8.5, ncol=5)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("滑窗样本序号")
    fig.tight_layout()
    fig.savefig(REAL_FIG / "r4c_test_predictions_60s.png", dpi=120)
    plt.close(fig)

    # 图 3：Patch-TCN-SKNet-Transformer 与 GRU 的分目标损失曲线
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2))
    plot_keys = [
        ("GRU_Phys_outlet_moisture", "GRU_Phys (出口含水率)"),
        ("GRU_Phys_outlet_temp", "GRU_Phys (出口温度)"),
        ("Patch_TCN_SKNet_Trans_Phys_outlet_moisture", "Patch_TCN_SKNet_Trans (出口含水率)"),
        ("Patch_TCN_SKNet_Trans_Phys_outlet_temp", "Patch_TCN_SKNet_Trans (出口温度)"),
    ]
    for ax, (k, title) in zip(axes.ravel(), plot_keys):
        h = dl_histories[k]
        ax.plot(h["train_loss"], color=COLORS["target"], lw=1.5, label="Train Loss")
        ax.plot(h["val_loss"], color=COLORS["good"], lw=1.5, label="Val Loss")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted SmoothL1 Loss")
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)
    fig.suptitle("r4c 分目标专精训练/验证损失曲线（观察 Patch 压缩抑制过拟合效果）", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REAL_FIG / "r4c_dl_loss_curves.png", dpi=120)
    plt.close(fig)

    rep_lines = [
        "=" * 82,
        f"真实数据脚本 4C：物理单调性约束 + 分目标专精 + Patch-TCN-SKNet-Transformer 报告 ({device})",
        "=" * 82,
        res_df.to_string(index=False),
        "\n【验证集集成权重】",
        *weight_notes,
        *phys_lines,
        *evo_lines,
        "",
        f"详细指标 CSV -> {REAL_REP / 'r4c_model_comparison.csv'}",
        f"三代演进 CSV -> {REAL_REP / 'r4c_evolution_r4_r4b_r4c.csv'}",
        f"图表已保存   -> {REAL_FIG}",
    ]
    rep_text = "\n".join(rep_lines)
    print("\n" + rep_text)
    save_real_report("r4c_train_report.txt", rep_text)


if __name__ == "__main__":
    main()
