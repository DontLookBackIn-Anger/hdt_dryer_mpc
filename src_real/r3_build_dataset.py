#!/usr/bin/env python3
"""
=== 真实数据脚本 3：跨 18 批次无泄露多步滑窗数据集构建 ===

与仿真脚本 src/s3_window_builder.py 严格隔离，专门处理 data/processed/real/batches_named/
中的 18 个已核对表头的真实生产批次：

核心防泄露与工程设计：
  1. 严格特征筛选：仅保留 9 个已 100% 核对物理意义的干净列（2 目标 + 7 外生），
     彻底剔除全零列（water_flow）、待确认列、以及含目标泄露/9966毛刺的 30s 控制衍生列（col12~15）。
  2. 严格按时间顺序「按完整批次」切分：
       - 训练集：前 12 个批次（2025-03-18 全部 6 批 + 2025-03-19 前 6 批）
       - 验证集：中间 2 个批次（2025-03-19 最后 2 批）
       - 测试集：最后 4 个批次（2025-04-09 / 04-10 跨月未来批次，检验真实泛化！）
  3. 严禁跨批次滑窗：每个批次仅在稳态段（phase == 'steady'）内部独立构造滑窗。
  4. 零数据泄露缩放：分位数截断上下界 [p0.1, p99.9] 与 RobustScaler 仅在训练集上 fit。
  5. 同时保存当前时刻 y_curr 与未来多步 y_future（默认预测未来 [1, 15, 30, 60] 秒），
     支持所有模型采用「残差增量 Δy = y(t+h) - y(t)」预测范式，抵抗跨批次均值漂移。

运行（由用户自行执行）：
    cd ~/hdt_dryer_mpc
    /opt/conda_envs/dryer-ts/bin/python src_real/r3_build_dataset.py

产出：
    data/processed/real/multibatch_windows.npz
    data/processed/real/robust_scaler.joblib
    outputs_real/reports/r3_dataset_summary.txt
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from real_common import REAL_PROC, REAL_REP, save_real_report

# 严格锁定的 9 个干净物理列（绝不包含 col12~15 衍生泄露列）
TARGET_COLS = ["outlet_moisture", "outlet_temp"]
EXOG_COLS = [
    "steam_flow",       # 补偿蒸汽流量 (MV)
    "steam_valve_fb",   # 蒸汽阀门开度反馈 (MV fb)
    "steam_valve_sp",   # 蒸汽阀门开度给定 (MV sp)
    "casing_flow",      # 加料瞬时流量 (MV/配比)
    "inlet_moisture",   # 入口含水率 (DV)
    "feed_rate",        # 称瞬时流量 (DV)
    "return_air_temp",  # 回风温度 (状态)
]
FEATURE_COLS = TARGET_COLS + EXOG_COLS


def build_windows_for_batch(
    X_scaled: np.ndarray,
    Y_raw: np.ndarray,
    input_size: int,
    horizons: list[int],
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在单个连续稳态批次内部构造滑窗样本，绝不跨批次拼接。"""
    max_h = max(horizons)
    n = len(X_scaled)
    if n < input_size + max_h + 1:
        return (
            np.empty((0, input_size, X_scaled.shape[1]), dtype=np.float32),
            np.empty((0, len(horizons), Y_raw.shape[1]), dtype=np.float32),
            np.empty((0, Y_raw.shape[1]), dtype=np.float32),
        )

    xs, ys_fut, ys_cur = [], [], []
    # t 为当前时刻索引：输入使用 [t - input_size + 1 .. t]，预测 [t + h for h in horizons]
    for t in range(input_size - 1, n - max_h, stride):
        xs.append(X_scaled[t - input_size + 1 : t + 1])
        ys_cur.append(Y_raw[t])
        ys_fut.append(Y_raw[[t + h for h in horizons]])

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys_fut, dtype=np.float32),
        np.asarray(ys_cur, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description="构建真实 18 批次无泄露多步预测滑窗数据集")
    parser.add_argument("--input-size", type=int, default=60, help="回看历史窗口长度（秒），默认 60")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 15, 30, 60], help="多步预测步长（秒）")
    parser.add_argument("--stride", type=int, default=5, help="滑窗步长（秒），默认 5（设为 1 可获得全量稠密样本）")
    parser.add_argument("--n-train", type=int, default=12, help="按时间顺序取前 n 个批次作训练集，默认 12")
    parser.add_argument("--n-val", type=int, default=2, help="紧接其后取 n 个批次作验证集，默认 2（其余作测试集）")
    args = parser.parse_args()

    batch_dir = REAL_PROC / "batches_named"
    csv_files = sorted(batch_dir.glob("*.csv"))
    if len(csv_files) < args.n_train + args.n_val + 1:
        raise FileNotFoundError(
            f"在 {batch_dir} 中仅找到 {len(csv_files)} 个批次文件，请先运行 src_real/r1b_add_batch_headers.py"
        )

    train_files = csv_files[: args.n_train]
    val_files = csv_files[args.n_train : args.n_train + args.n_val]
    test_files = csv_files[args.n_train + args.n_val :]

    # 1. 读取各批次稳态段
    def _load_steady(paths: list[Path]) -> list[tuple[str, pd.DataFrame]]:
        out = []
        for p in paths:
            df = pd.read_csv(p, encoding="utf-8-sig")
            st = df[df["phase"] == "steady"].reset_index(drop=True)
            out.append((p.stem, st))
        return out

    train_batches = _load_steady(train_files)
    val_batches = _load_steady(val_files)
    test_batches = _load_steady(test_files)

    # 2. 仅在训练集上计算 [p0.1, p99.9] 截断阈值与 RobustScaler（零数据泄露）
    train_concat = pd.concat([df[FEATURE_COLS] for _, df in train_batches], ignore_index=True)
    clip_low = train_concat.quantile(0.001).to_numpy(dtype=np.float32)
    clip_high = train_concat.quantile(0.999).to_numpy(dtype=np.float32)

    train_clipped = np.clip(train_concat.to_numpy(dtype=np.float32), clip_low, clip_high)
    scaler = RobustScaler()
    scaler.fit(train_clipped)

    # 3. 逐批次内部构造滑窗
    def _build_split(batches: list[tuple[str, pd.DataFrame]]):
        X_list, Yf_list, Yc_list, bid_list = [], [], [], []
        for bid, df in batches:
            raw_feat = df[FEATURE_COLS].to_numpy(dtype=np.float32)
            raw_targ = df[TARGET_COLS].to_numpy(dtype=np.float32)
            feat_scaled = scaler.transform(np.clip(raw_feat, clip_low, clip_high)).astype(np.float32)
            xs, yf, yc = build_windows_for_batch(
                feat_scaled, raw_targ, args.input_size, args.horizons, args.stride
            )
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

    X_tr, Yf_tr, Yc_tr, B_tr = _build_split(train_batches)
    X_va, Yf_va, Yc_va, B_va = _build_split(val_batches)
    X_te, Yf_te, Yc_te, B_te = _build_split(test_batches)

    # 4. 保存数据包与缩放器
    npz_path = REAL_PROC / "multibatch_windows.npz"
    scaler_path = REAL_PROC / "robust_scaler.joblib"
    np.savez_compressed(
        npz_path,
        X_train=X_tr, Y_train=Yf_tr, Y_curr_train=Yc_tr, batch_train=B_tr,
        X_val=X_va, Y_val=Yf_va, Y_curr_val=Yc_va, batch_val=B_va,
        X_test=X_te, Y_test=Yf_te, Y_curr_test=Yc_te, batch_test=B_te,
        feature_cols=np.asarray(FEATURE_COLS),
        target_cols=np.asarray(TARGET_COLS),
        horizons=np.asarray(args.horizons, dtype=np.int32),
        input_size=np.asarray(args.input_size, dtype=np.int32),
        clip_low=clip_low,
        clip_high=clip_high,
    )
    joblib.dump(
        {"scaler": scaler, "clip_low": clip_low, "clip_high": clip_high, "feature_cols": FEATURE_COLS},
        scaler_path,
    )

    # 5. 输出留痕报告
    lines = [
        "=" * 72,
        "真实数据脚本 3：跨 18 批次无泄露多步滑窗数据集构建报告",
        "=" * 72,
        f"输入特征 ({len(FEATURE_COLS)} 维): {FEATURE_COLS}",
        f"预测目标 ({len(TARGET_COLS)} 维): {TARGET_COLS}",
        f"回看窗口 input_size = {args.input_size} 秒 | 预测步长 horizons = {args.horizons} 秒 | 步长 stride = {args.stride} 秒",
        "",
        "【严格按时间顺序的批次划分（零跨批次泄露）】",
        f"  训练集 ({len(train_files)} 批): {[p.stem for p in train_files]}",
        f"  验证集 ({len(val_files)} 批): {[p.stem for p in val_files]}",
        f"  测试集 ({len(test_files)} 批): {[p.stem for p in test_files]}",
        "",
        "【滑窗张量维度】",
        f"  X_train: {X_tr.shape} | Y_train: {Yf_tr.shape}",
        f"  X_val  : {X_va.shape} | Y_val  : {Yf_va.shape}",
        f"  X_test : {X_te.shape} | Y_test : {Yf_te.shape}",
        "",
        "【测试集跨月目标分布对比（检验漂移程度）】",
        f"  训练集出口含水率: μ={Yc_tr[:, 0].mean():.4f}%, σ={Yc_tr[:, 0].std():.4f}%",
        f"  测试集出口含水率: μ={Yc_te[:, 0].mean():.4f}%, σ={Yc_te[:, 0].std():.4f}%",
        f"  训练集出口温度  : μ={Yc_tr[:, 1].mean():.3f}℃, σ={Yc_tr[:, 1].std():.3f}℃",
        f"  测试集出口温度  : μ={Yc_te[:, 1].mean():.3f}℃, σ={Yc_te[:, 1].std():.3f}℃",
        "",
        f"数据集已保存 -> {npz_path}",
        f"缩放器已保存 -> {scaler_path}",
        "下一步：运行 /opt/conda_envs/dryer-ts/bin/python src_real/r4_train_and_eval.py",
    ]
    text = "\n".join(lines)
    print(text)
    save_real_report("r3_dataset_summary.txt", text)


if __name__ == "__main__":
    main()
