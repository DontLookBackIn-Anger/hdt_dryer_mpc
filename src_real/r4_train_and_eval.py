#!/usr/bin/env python3
"""
=== 真实数据脚本 4：多模型训练、跨批次多步评估与物性校验 ===

与仿真脚本 src/s3_dl_forecast.py、src/s4_multistep_eval.py 严格隔离，在真实 18 批次
无泄露滑窗数据集（data/processed/real/multibatch_windows.npz）上统一口径训练并对比 5 类模型：

  1. Naive (Persistence)          —— 保持当前值 ŷ(t+h) = y(t)，打破单步虚高陷阱的下限基线
  2. Ridge (多输出线性 ARX)        —— 线性基线，对应线性 MPC 内部预测模型
  3. LightGBM (直接多步 GBDT)      —— 强非线性表格特征模型（滞后采样 + 差分 + 窗口统计）
  4. Residual GRU (PyTorch GPU)   —— 轻量循环时序网络，预测增量 Δy 抵抗跨批次漂移
  5. Improved TCN-SKNet-Transformer (PyTorch GPU)
     —— 升级复现原报告混合架构：因果膨胀 TCN + 真多尺度 SK-Net (k=3/5) + 2层 Transformer + 残差多步头，
        彻底修复原 Keras 代码中的缩放泄露、单步冒充多步、漏特征与层数未生效问题。

运行（由用户自行执行，推荐在项目根目录运行）：
    cd ~/hdt_dryer_mpc
    /opt/conda_envs/dryer-ts/bin/python src_real/r4_train_and_eval.py
    # 可选快速试跑（10轮验证流程）：
    /opt/conda_envs/dryer-ts/bin/python src_real/r4_train_and_eval.py --epochs 10

产出（全部独立保存在 outputs_real/）：
    outputs_real/checkpoints/gru_best.pt
    outputs_real/checkpoints/tcn_sknet_transformer_best.pt
    outputs_real/reports/r4_model_comparison.csv
    outputs_real/reports/r4_train_report.txt
    outputs_real/figures/r4_horizon_rmse_compare.png
    outputs_real/figures/r4_test_predictions_60s.png
    outputs_real/figures/r4_dl_loss_curves.png
"""
import sys
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from real_common import REAL_PROC, REAL_OUT, REAL_FIG, REAL_REP, COLORS, calc_metrics, save_real_report


# ---------------------------------------------------------------- 1. 表格特征提取（供 Ridge / LightGBM）
def build_tabular_features(X_seq: np.ndarray) -> np.ndarray:
    """将 (N, T=60, F=9) 滑窗压缩为工业控制高信息量表格特征：
    - 关键滞后点采样（t=0, -5, -15, -30, -45, -59 秒）
    - 窗口均值、标准差
    - 短期变化率（最后 10s - 前 10s）与长期趋势（最后 15s - 最前 15s）
    """
    N, T, F = X_seq.shape
    lag_offsets = [1, 6, 16, 31, 46, T]  # 对应倒数第 1, 6, 16, 31, 46, 60 个点
    lag_idx = [max(0, T - o) for o in lag_offsets]
    lag_feats = X_seq[:, lag_idx, :].reshape(N, -1)

    mean_all = X_seq.mean(axis=1)
    std_all = X_seq.std(axis=1)
    short_diff = X_seq[:, -10:, :].mean(axis=1) - X_seq[:, -20:-10, :].mean(axis=1)
    long_diff = X_seq[:, -15:, :].mean(axis=1) - X_seq[:, :15, :].mean(axis=1)
    return np.hstack([lag_feats, mean_all, std_all, short_diff, long_diff]).astype(np.float32)


# ---------------------------------------------------------------- 2. PyTorch 深度学习模型定义
class ResidualGRU(nn.Module):
    """轻量残差 GRU：输入 (B, T, F) -> 输出多步残差增量 Δy (B, H, C)。"""

    def __init__(self, in_dim: int, hidden_dim: int, n_horizons: int, n_targets: int, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_targets = n_targets
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_horizons * n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        h_last = self.norm(out[:, -1, :])
        delta = self.head(h_last)
        return delta.view(-1, self.n_horizons, self.n_targets)


class CausalConv1dBlock(nn.Module):
    """因果膨胀残差卷积块（严格左侧补零，杜绝未来信息泄露）。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _causal_conv(self, x: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        x_pad = nn.functional.pad(x, (self.pad, 0))
        return conv(x_pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self._causal_conv(x, self.conv1)))
        h = self.drop(self.act(self._causal_conv(h, self.conv2)))
        return h + self.residual(x)


class SKNet1D(nn.Module):
    """真正的 1D 多尺度选择性卷积核注意力模块（Split k=3/5 -> Fuse -> Softmax Select）。"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.branch3 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.fc_reduce = nn.Sequential(nn.Linear(channels, mid), nn.GELU())
        self.fc_a = nn.Linear(mid, channels)
        self.fc_b = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        u1 = self.branch3(x)
        u2 = self.branch5(x)
        u = u1 + u2
        s = u.mean(dim=-1)  # 全局时间平均池化 (B, C)
        z = self.fc_reduce(s)
        logits = torch.stack([self.fc_a(z), self.fc_b(z)], dim=1)  # (B, 2, C)
        weights = torch.softmax(logits, dim=1)
        w1 = weights[:, 0, :].unsqueeze(-1)
        w2 = weights[:, 1, :].unsqueeze(-1)
        return w1 * u1 + w2 * u2


class ImprovedTCNSKNetTransformer(nn.Module):
    """升级版 TCN + SK-Net + 2层 Transformer 混合架构（残差多步预测头）。

    相比原报告 Keras 代码的 4 项核心修复：
      1. 因果膨胀率 [1, 2, 4, 8]（感受野 > 60s，覆盖实测 32~53s 工艺纯滞后）
      2. 实现真正的多尺度分支 SK-Net（k=3 与 k=5 Softmax 自适应选择）
      3. 真正堆叠 num_layers=2 层 TransformerEncoder（加正弦位置编码）
      4. 预测多步增量 Δy(t+h) 而非单步绝对值，彻底消除跨批次均值漂移与单步虚高
    """

    def __init__(
        self,
        in_dim: int,
        n_horizons: int,
        n_targets: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
        max_len: int = 256,
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_targets = n_targets

        # 1. 多层因果膨胀 TCN（d=1,2,4,8，感受野 = 1 + 2*(2)*(1+2+4+8) = 61s）
        self.tcn = nn.Sequential(
            CausalConv1dBlock(in_dim, d_model, kernel_size=3, dilation=1, dropout=dropout),
            CausalConv1dBlock(d_model, d_model, kernel_size=3, dilation=2, dropout=dropout),
            CausalConv1dBlock(d_model, d_model, kernel_size=3, dilation=4, dropout=dropout),
            CausalConv1dBlock(d_model, d_model, kernel_size=3, dilation=8, dropout=dropout),
        )
        # 2. 多尺度 SK-Net
        self.sknet = SKNet1D(d_model, reduction=4)

        # 3. 位置编码 + 2 层 TransformerEncoder
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

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

        # 4. 融合末尾时间步与全局平均池化 -> 多步残差输出头
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_horizons * n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        h = self.tcn(x.transpose(1, 2))
        h = self.sknet(h).transpose(1, 2)  # (B, T, d_model)
        h = h + self.pe[:, : h.size(1), :]
        h = self.norm(self.transformer(h))
        feat = torch.cat([h[:, -1, :], h.mean(dim=1)], dim=-1)
        delta = self.head(feat)
        return delta.view(-1, self.n_horizons, self.n_targets)


# ---------------------------------------------------------------- 3. PyTorch 通用训练循环
def train_torch_model(
    model: nn.Module,
    X_tr: np.ndarray,
    dY_tr_norm: np.ndarray,
    X_va: np.ndarray,
    dY_va_norm: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    ckpt_path: Path,
    patience: int = 12,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05)
    criterion = nn.SmoothL1Loss(beta=0.5)  # Huber 损失，对工业传感器残余毛刺更鲁棒

    best_val = float("inf")
    wait = 0
    tr_hist, va_hist = [], []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
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
                va_loss += criterion(model(bx), by).item() * len(bx)
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


@torch.no_grad()
def predict_torch_model(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        bx = torch.from_numpy(X[i : i + batch_size]).to(device)
        preds.append(model(bx).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------- 4. 主流程
def main():
    parser = argparse.ArgumentParser(description="真实 18 批次多模型训练与跨批次多步评估")
    parser.add_argument("--epochs", type=int, default=40, help="深度学习模型最大训练轮数，默认 40")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size，默认 128")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率，默认 1e-3")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    ckpt_dir = REAL_OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    npz_path = REAL_PROC / "multibatch_windows.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"未找到数据集 {npz_path}，请先运行 src_real/r3_build_dataset.py")

    data = np.load(npz_path, allow_pickle=True)
    X_tr, Y_tr, Yc_tr = data["X_train"], data["Y_train"], data["Y_curr_train"]
    X_va, Y_va, Yc_va = data["X_val"], data["Y_val"], data["Y_curr_val"]
    X_te, Y_te, Yc_te = data["X_test"], data["Y_test"], data["Y_curr_test"]
    B_te = data["batch_test"]
    horizons = data["horizons"].tolist()
    target_cols = data["target_cols"].tolist()
    feature_cols = data["feature_cols"].tolist()

    n_h, n_t = len(horizons), len(target_cols)

    # 构造残差增量目标 ΔY = Y_future - Y_curr（抵抗跨批次基准漂移！）
    dY_tr = Y_tr - Yc_tr[:, None, :]
    dY_va = Y_va - Yc_va[:, None, :]
    # 仅用训练集的 ΔY 标准差做归一化，平衡水分（~0.04%）与温度（~0.3℃）的梯度量级
    dy_std = dY_tr.std(axis=(0, 1), keepdims=True) + 1e-6
    dY_tr_norm = (dY_tr / dy_std).astype(np.float32)
    dY_va_norm = (dY_va / dy_std).astype(np.float32)

    print("=" * 72)
    print(f"真实数据多模型训练与跨批次测试 | Device: {device}")
    print(f"Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)} | Horizons={horizons}s")
    print("=" * 72)

    preds_test: dict[str, np.ndarray] = {}
    train_times: dict[str, float] = {}
    dl_histories: dict[str, dict] = {}

    # ---- 模型 1: Naive (Persistence) ----
    print("\n[1/5] 评估 Naive Persistence 基线...")
    preds_test["Naive"] = np.repeat(Yc_te[:, None, :], n_h, axis=1)
    train_times["Naive"] = 0.0

    # 预先提取表格特征供 Ridge & LightGBM 共用
    X_tr_tab = build_tabular_features(X_tr)
    X_va_tab = build_tabular_features(X_va)
    X_te_tab = build_tabular_features(X_te)

    # ---- 模型 2: Multi-Output Ridge (线性 ARX) ----
    print("\n[2/5] 训练 Multi-Output Ridge (线性 ARX 基线)...")
    t0 = time.time()
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr_tab, dY_tr_norm.reshape(len(X_tr), -1))
    dY_pred_ridge = ridge.predict(X_te_tab).reshape(-1, n_h, n_t) * dy_std
    preds_test["Ridge_ARX"] = Yc_te[:, None, :] + dY_pred_ridge
    train_times["Ridge_ARX"] = time.time() - t0
    print(f"    完成，耗时 {train_times['Ridge_ARX']:.2f}s")

    # ---- 模型 3: LightGBM (直接多步 GBDT) ----
    print("\n[3/5] 训练 LightGBM (直接多步 GBDT)...")
    t0 = time.time()
    lgb_pred_delta = np.zeros((len(X_te), n_h, n_t), dtype=np.float32)
    for hi, h in enumerate(horizons):
        for ti, tname in enumerate(target_cols):
            reg = lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                subsample_for_bin=50000,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=args.seed,
                n_jobs=-1,
                verbose=-1,
            )
            reg.fit(
                X_tr_tab,
                dY_tr[:, hi, ti],
                eval_set=[(X_va_tab, dY_va[:, hi, ti])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            lgb_pred_delta[:, hi, ti] = reg.predict(X_te_tab)
    preds_test["LightGBM"] = Yc_te[:, None, :] + lgb_pred_delta
    train_times["LightGBM"] = time.time() - t0
    print(f"    完成，耗时 {train_times['LightGBM']:.2f}s")

    # ---- 模型 4: Residual GRU ----
    print("\n[4/5] 训练 Residual GRU (PyTorch)...")
    t0 = time.time()
    gru = ResidualGRU(in_dim=X_tr.shape[-1], hidden_dim=64, n_horizons=n_h, n_targets=n_t, num_layers=2, dropout=0.15)
    dl_histories["Residual_GRU"] = train_torch_model(
        gru, X_tr, dY_tr_norm, X_va, dY_va_norm,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=device, ckpt_path=ckpt_dir / "gru_best.pt",
    )
    dY_pred_gru = predict_torch_model(gru, X_te, device) * dy_std
    preds_test["Residual_GRU"] = Yc_te[:, None, :] + dY_pred_gru
    train_times["Residual_GRU"] = time.time() - t0
    print(f"    完成，耗时 {train_times['Residual_GRU']:.2f}s")

    # ---- 模型 5: Improved TCN-SKNet-Transformer ----
    print("\n[5/5] 训练 Improved TCN-SKNet-Transformer (PyTorch)...")
    t0 = time.time()
    hybrid = ImprovedTCNSKNetTransformer(
        in_dim=X_tr.shape[-1], n_horizons=n_h, n_targets=n_t,
        d_model=64, nhead=4, num_layers=2, dropout=0.15,
    )
    dl_histories["TCN_SKNet_Trans"] = train_torch_model(
        hybrid, X_tr, dY_tr_norm, X_va, dY_va_norm,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=device, ckpt_path=ckpt_dir / "tcn_sknet_transformer_best.pt",
    )
    dY_pred_hyb = predict_torch_model(hybrid, X_te, device) * dy_std
    preds_test["TCN_SKNet_Trans"] = Yc_te[:, None, :] + dY_pred_hyb
    train_times["TCN_SKNet_Trans"] = time.time() - t0
    print(f"    完成，耗时 {train_times['TCN_SKNet_Trans']:.2f}s")

    # ---------------------------------------------------------------- 5. 统一口径评估
    rows = []
    naive_rmse = {}
    for hi, h in enumerate(horizons):
        for ti, tname in enumerate(target_cols):
            m_nv = calc_metrics(Y_te[:, hi, ti], preds_test["Naive"][:, hi, ti])
            naive_rmse[(h, tname)] = m_nv["rmse"]

    for mname, y_hat in preds_test.items():
        for hi, h in enumerate(horizons):
            for ti, tname in enumerate(target_cols):
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
    res_df.to_csv(REAL_REP / "r4_model_comparison.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- 6. 物理方向一致性检验（蒸汽 +1σ 扰动）
    steam_idx = feature_cols.index("steam_flow")
    X_te_pert = X_te[:500].copy()
    X_te_pert[:, -20:, steam_idx] += 1.0  # 过去 20s 蒸汽增加 1 个 IQR 单位
    phys_lines = ["\n【物理方向一致性检验：过去 20s 补偿蒸汽 +1 IQR 对未来 60s 预测的平均响应】"]
    for mname, mdl in [("Residual_GRU", gru), ("TCN_SKNet_Trans", hybrid)]:
        base_p = predict_torch_model(mdl, X_te[:500], device) * dy_std
        pert_p = predict_torch_model(mdl, X_te_pert, device) * dy_std
        diff_60 = (pert_p - base_p)[:, -1, :].mean(axis=0)  # 60s 处平均变化
        phys_lines.append(
            f"  {mname:18s} | Δ出口含水率(60s) = {diff_60[0]:+.5f}% | Δ出口温度(60s) = {diff_60[1]:+.4f}℃"
        )
    phys_lines.append("  （物理预期：增加蒸汽应使出口含水率与出口温度呈正响应；若为负值，说明模型仍受闭环 PID 伪相关影响）")

    # ---------------------------------------------------------------- 7. 绘图
    # 图 1：各模型分步 RMSE 对比
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    model_names = list(preds_test.keys())
    for ti, tname in enumerate(target_cols):
        ax = axes[ti]
        for mname in model_names:
            sub = res_df[(res_df["model"] == mname) & (res_df["target"] == tname)]
            ax.plot(sub["horizon_s"], sub["rmse"], marker="o", lw=1.8, label=mname)
        ax.set_title(f"{tname}：不同预测步长 RMSE 对比（测试集 4 批）", fontweight="bold")
        ax.set_xlabel("预测步长 Horizon (秒)")
        ax.set_ylabel("RMSE")
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5)
    fig.suptitle("真实 18 批次跨月泛化测试：5 类模型多步预测误差对比", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REAL_FIG / "r4_horizon_rmse_compare.png", dpi=120)
    plt.close(fig)

    # 图 2：选取测试集中最长的一批画 60s 预测曲线对比
    uniq_b, counts_b = np.unique(B_te, return_counts=True)
    show_bid = uniq_b[int(np.argmax(counts_b))]
    mask_b = B_te == show_bid
    h_idx = len(horizons) - 1  # 60s
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    for ti, tname in enumerate(target_cols):
        ax = axes[ti]
        ax.plot(Y_te[mask_b, h_idx, ti], color="black", lw=1.4, label="真实值")
        ax.plot(preds_test["Naive"][mask_b, h_idx, ti], color="gray", ls=":", lw=1.0, label="Naive(60s前)")
        ax.plot(preds_test["LightGBM"][mask_b, h_idx, ti], color=COLORS["good"], lw=1.1, alpha=0.9, label="LightGBM")
        ax.plot(preds_test["Residual_GRU"][mask_b, h_idx, ti], color=COLORS["pred"], lw=1.1, alpha=0.9, label="Residual_GRU")
        ax.plot(preds_test["TCN_SKNet_Trans"][mask_b, h_idx, ti], color=COLORS["target"], ls="--", lw=1.2, label="TCN_SKNet_Trans")
        ax.set_ylabel(tname)
        ax.set_title(f"测试批次 {show_bid} — 提前 {horizons[h_idx]} 秒预测效果对比 ({tname})", loc="left", fontweight="bold")
        ax.legend(loc="best", fontsize=8.5, ncol=5)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("滑窗样本序号")
    fig.tight_layout()
    fig.savefig(REAL_FIG / "r4_test_predictions_60s.png", dpi=120)
    plt.close(fig)

    # 图 3：深度学习训练/验证损失曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (mname, hist) in zip(axes, dl_histories.items()):
        ax.plot(hist["train_loss"], color=COLORS["target"], lw=1.5, label="Train Loss")
        ax.plot(hist["val_loss"], color=COLORS["good"], lw=1.5, label="Val Loss")
        ax.set_title(f"{mname} 训练与验证损失曲线", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("SmoothL1 Loss (Normalized Δy)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(REAL_FIG / "r4_dl_loss_curves.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- 8. 保存文字报告
    rep_lines = [
        "=" * 78,
        f"真实数据脚本 4：多模型跨批次多步预测评估报告 (Device={device})",
        "=" * 78,
        res_df.to_string(index=False),
        *phys_lines,
        "",
        f"详细指标 CSV -> {REAL_REP / 'r4_model_comparison.csv'}",
        f"对比图已保存 -> {REAL_FIG}",
    ]
    rep_text = "\n".join(rep_lines)
    print("\n" + rep_text)
    save_real_report("r4_train_report.txt", rep_text)


if __name__ == "__main__":
    main()
