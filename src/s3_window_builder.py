#!/usr/bin/env python3
"""
=== 第 3 项：滑动窗口样本构造器（报告 6.3）===

报告 6.3 给出的核心公式：

    X(k) = [u(k−L+1 : k),  d(k−L+1 : k),  y(k−L+1 : k)]
    Y(k) = [Wout(k+1 : k+Np),  Tout(k+1 : k+Np)]

    其中 L  = 回看窗口长度（历史）
         Np = 预测步长（多步 MIMO 输出）
         u  = 操纵量 MV（蒸汽、热风）
         d  = 可测扰动 DV（入口水分、入口温度、进料量）+ 过程状态 PV
         y  = 历史输出（出口水分、出口温度）

**这是整个项目最核心的一段代码。** 真实数据来了以后，所有模型都从这里取样本。

关键设计与坑：
  1. 标签必须双输出（MIMO），不能只预测水分
  2. 必须按时间顺序，禁止打乱
  3. 断点检测：跨越"通讯中断"或"状态切换"的窗口必须丢弃，
     否则会用断裂的数据训练（报告 5.4）
  4. 标准化参数只能在训练集拟合
  5. 必须按批次划分，不能随机划分（报告 7.1）

运行：
    cd ~/hdt_dryer_mpc
    python src/s3_window_builder.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from common import FIG, REP, OUT, COLORS, PALETTE
from dryer_sim import (simulate_hdt_full, MV_COLS, DV_COLS, PV_COLS, CV_COLS,
                       STATE_CODES, STATE_CN, data_quality_summary)


# ================================================================
# 配置
# ================================================================
@dataclass
class WindowConfig:
    """滑动窗口配置。

    L 和 Np 的确定依据（报告 6.3）：
      L  ≥ 最大滞后 θ + 3τ（要看到完整的响应过程）
      Np ≥ 控制周期 × 需要的预测能力，且 ≥ θ + τ（覆盖一次完整动作）
    """
    L: int = 120              # 回看窗口（秒）。延迟辨识 θ=17s, τ=90s → 至少 120s
    Np: int = 30              # 预测步长（秒）。MPC 控制周期通常 15~30s
    stride: int = 1           # 滑动步长。1=每个点一个样本（训练）；评估时可加大
    # 输入配置
    use_mv: bool = True
    use_dv: bool = True
    use_pv: bool = True
    use_y_history: bool = True
    # 质量过滤
    drop_quality: tuple = (1, 2, 3, 4)   # 丢弃含这些质量码的窗口
    require_stable: bool = True           # 只保留稳定生产段
    # 归一化
    standardize: bool = True


# ================================================================
# 核心：样本构造
# ================================================================
def build_windows(df: pd.DataFrame, cfg: WindowConfig | None = None,
                  scaler: dict | None = None, return_info: bool = True):
    """按报告 6.3 构造滑动窗口样本。

    Returns
    -------
    X : (n_samples, n_features)  展平后的特征矩阵，便于树模型/MLP
    Y : (n_samples, n_out, Np)   多步双输出标签
    meta : DataFrame             每个样本的起点、批次、时间戳（用于划分）
    scaler : dict                归一化参数（仅在训练集拟合时返回）
    X_seq : (n_samples, L, n_feat) 时序形式，便于 LSTM/TCN/Transformer

    注意 X 和 X_seq 是同一数据的两种形状：
        X 用于树模型（需要展平）
        X_seq 用于深度时序模型
    """
    cfg = cfg or WindowConfig()

    in_cols = []
    if cfg.use_mv: in_cols += MV_COLS
    if cfg.use_dv: in_cols += DV_COLS
    if cfg.use_pv: in_cols += PV_COLS
    if cfg.use_y_history: in_cols += CV_COLS

    # 有效行掩码
    ok = pd.Series(True, index=df.index)
    if cfg.require_stable:
        ok &= df["machine_state"] == STATE_CODES["stable"]
    for q in cfg.drop_quality:
        ok &= df["quality_flag"] != q
    # 输入输出都不能缺失
    ok &= df[in_cols + CV_COLS].notna().all(axis=1)
    ok &= df["machine_state"].notna()

    ok_arr = ok.values

    # ---- 归一化参数 ----
    if cfg.standardize:
        if scaler is None:
            mu = df.loc[ok, in_cols].mean().values
            sd = df.loc[ok, in_cols].std().replace(0, 1).values
            tmu = df.loc[ok, CV_COLS].mean().values
            tsd = df.loc[ok, CV_COLS].std().replace(0, 1).values
            scaler = dict(mu=mu, sd=sd, tmu=tmu, tsd=tsd, cols=in_cols)
        data_in = (df[in_cols].values - scaler["mu"]) / scaler["sd"]
        data_out = (df[CV_COLS].values - scaler["tmu"]) / scaler["tsd"]
    else:
        data_in = df[in_cols].values
        data_out = df[CV_COLS].values
        scaler = dict(mu=np.zeros(len(in_cols)), sd=np.ones(len(in_cols)),
                      tmu=np.zeros(len(CV_COLS)), tsd=np.ones(len(CV_COLS)), cols=in_cols)

    # 填补 NaN（仅限输入；输出 NaN 的窗口会被丢弃）
    data_in = np.nan_to_num(data_in, nan=0.0)

    L, Np, stride = cfg.L, cfg.Np, cfg.stride
    n = len(df)

    # 样本起点：至少要有 L 点历史，且未来 Np 点标签可用
    starts = np.arange(L - 1, n - Np, stride)

    # 窗口内所有点必须有效（质量+状态）—— 这是最关键的一步
    # 用累计和快速判断窗口内是否有无效点
    invalid = (~ok_arr).astype(np.int32)
    cs = np.r_[0, np.cumsum(invalid)]

    X_list, Y_list, meta_rows = [], [], []

    for k in starts:
        # 历史窗口取 [k-L+1, k]，共 L 个点（含当前时刻 k）
        # 注意切片上界不含：h1 必须等于 k+1 才能得到长度 L
        h0, h1 = k - L + 1, k + 1
        f0, f1 = k + 1, k + 1 + Np         # 未来标签 [k+1, k+1+Np)
        if f1 > n:
            continue
        # 整个 [h0, f1) 区间内不能有无效点
        if cs[f1] - cs[h0] > 0:
            continue

        X_list.append(data_in[h0:h1].T)        # (n_feat, L)
        Y_list.append(data_out[f0:f1].T)       # (n_out, Np)
        meta_rows.append({
            "k": k,
            "ds": df["timestamp"].iloc[k],
            "machine_state": int(df["machine_state"].iloc[k]),
        })

    if not X_list:
        raise RuntimeError("没有构造出任何有效样本！检查 L、Np、质量过滤条件。")

    X_seq = np.array(X_list)        # (n_samples, n_feat, L)
    Y = np.array(Y_list)            # (n_samples, n_out, Np)
    meta = pd.DataFrame(meta_rows)

    # 展平给树模型：(n_samples, n_feat*L)
    X_flat = X_seq.reshape(len(X_seq), -1)

    if return_info:
        info = dict(n_features=len(in_cols), feature_names=in_cols,
                    L=L, Np=Np, n_out=len(CV_COLS), out_names=CV_COLS)
        return X_flat, Y, meta, scaler, X_seq, info
    return X_flat, Y, meta, scaler


# ================================================================
# 批次级划分（报告 7.1）
# ================================================================
def split_by_batch(X, Y, meta, batch_ids, train_ratio=0.7, val_ratio=0.15):
    """按批次划分训练/验证/测试。

    报告 7.1："优先采用较早批次训练、后续批次验证/测试"。
    **绝不能随机打乱时间序列**，否则相邻点同时出现在训练和测试里，
    指标会虚高得离谱。
    """
    uniq = sorted(set(batch_ids))
    n_b = len(uniq)
    if n_b < 3:
        # 批次太少时无法三分，退化为时间顺序切分并告警
        import warnings
        warnings.warn(f"只有 {n_b} 个批次，无法做批次级三分；退化为时间顺序切分")
        n = len(X)
        i1, i2 = int(n * train_ratio), int(n * (train_ratio + val_ratio))
        sl = slice
        return (X[sl(0, i1)], Y[sl(0, i1)]), (X[sl(i1, i2)], Y[sl(i1, i2)]), \
               (X[sl(i2, n)], Y[sl(i2, n)]), dict(train=["<early>"], val=["<mid>"], test=["<late>"])
    # 保证三个集合互斥且都不为空
    n_tr = max(1, int(round(n_b * train_ratio)))
    n_va = max(1, int(round(n_b * val_ratio)))
    if n_tr + n_va >= n_b:
        n_va = 1
        n_tr = n_b - 2
    tr_b = set(uniq[:n_tr])
    va_b = set(uniq[n_tr:n_tr + n_va])
    te_b = set(uniq[n_tr + n_va:])
    assert not (tr_b & va_b) and not (tr_b & te_b) and not (va_b & te_b), "划分集合重叠！"

    bids = np.array(batch_ids)
    m_tr = np.array([b in tr_b for b in bids])
    m_va = np.array([b in va_b for b in bids])
    m_te = np.array([b in te_b for b in bids])
    return (X[m_tr], Y[m_tr]), (X[m_va], Y[m_va]), (X[m_te], Y[m_te]), \
           dict(train=sorted(tr_b), val=sorted(va_b), test=sorted(te_b))


# ================================================================
# 演示
# ================================================================
def main():
    print("=" * 76)
    print("第 3 项：滑动窗口样本构造器（报告 6.3）")
    print("=" * 76)

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    # 生成 3 个批次模拟不同生产日
    batches = []
    for i, seed in enumerate((11, 22, 33, 44)):
        d = simulate_hdt_full(n_steps=1800, seed=seed)
        d["batch_id"] = f"B{i+1:02d}"
        batches.append(d)
    df_all = pd.concat(batches, ignore_index=True)

    log(f"数据：{len(batches)} 个批次，共 {len(df_all)} 点")
    log("\n数据质量审查（报告 5.4）")
    qs = data_quality_summary(df_all)
    log(qs.to_string(index=False))

    # ---- 构造窗口 ----
    cfg = WindowConfig(L=120, Np=30, stride=1, require_stable=True)
    log(f"\n窗口配置：L={cfg.L}s (回看), Np={cfg.Np}s (预测), stride={cfg.stride}")
    log(f"          输入: MV={cfg.use_mv}, DV={cfg.use_dv}, PV={cfg.use_pv}, "
        f"y历史={cfg.use_y_history}")

    X, Y, meta, scaler, X_seq, info = build_windows(df_all, cfg)
    log(f"\n构造结果：")
    log(f"  X (展平)  : {X.shape}      ← 给树模型")
    log(f"  X_seq     : {X_seq.shape}  ← 给 TCN/Transformer")
    log(f"  Y         : {Y.shape}      ← (样本, 双输出, Np)")
    log(f"  特征名     : {info['feature_names']}")
    log(f"  输出名     : {info['out_names']}")

    # ---- 检查标签对齐（关键验证）----
    log("\n【对齐验证】检查样本 0 的标签是否确实对应未来的真值")
    s0 = meta.iloc[0]
    k = int(s0["k"])
    # 标签是 outlet_moisture 的标准化值，反标准化后对比
    y0_std = Y[0, 0, :]                                  # 第 0 个样本的水分标签
    y0_real = y0_std * scaler["tsd"][0] + scaler["tmu"][0]
    actual = df_all["outlet_moisture"].values[k + 1:k + 1 + cfg.Np]
    err = np.abs(y0_real - actual).max()
    log(f"  样本起点 k={k}（{s0['ds']}）")
    log(f"  标签反标准化后 vs 原始数据 最大差 = {err:.6f}")
    log(f"  {'✅ 对齐正确' if err < 1e-6 else '❌ 对齐错误！'}")
    log(f"  历史末尾 y[k] 应在窗口内：X_seq 最后一行包含 y 在索引 {info['n_features']-2} 和 {info['n_features']-1}")

    # ---- 批次划分 ----
    # ⚠️ 坑：4 个批次的时间戳完全相同，不能用 timestamp 反查 batch_id
    # （会永远返回第一条）。应直接用行索引映射。
    batch_ids = df_all["batch_id"].values[meta["k"].values]
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte), split = split_by_batch(X, Y, meta, batch_ids)
    log(f"\n【批次级划分】（报告 7.1：禁止随机打乱）")
    log(f"  训练: {Xtr.shape[0]:6d} 样本  批次 {split['train']}")
    log(f"  验证: {Xva.shape[0]:6d} 样本  批次 {split['val']}")
    log(f"  测试: {Xte.shape[0]:6d} 样本  批次 {split['test']}")

    # ---- 对比：随机划分会造成什么后果 ----
    log("\n【反例演示】时间序列随机划分的泄漏问题")
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    # 正确做法：训练用前 70% 时间，测试用后 30%（严格时间顺序）
    n_tr_seq = int(len(X) * 0.7)
    m_ok = Ridge(alpha=1.0).fit(X[:n_tr_seq], Y[:n_tr_seq, 0, 0])
    r2_ok = r2_score(Y[n_tr_seq:, 0, 0], m_ok.predict(X[n_tr_seq:]))
    # 错误做法：随机打乱
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    m_bad = Ridge(alpha=1.0).fit(X[idx[:n_tr_seq]], Y[idx[:n_tr_seq], 0, 0])
    r2_bad = r2_score(Y[idx[n_tr_seq:], 0, 0], m_bad.predict(X[idx[n_tr_seq:]]))
    log(f"  按时间顺序划分  R² = {r2_ok:.4f}")
    log(f"  随机打乱划分    R² = {r2_bad:.4f}")
    if r2_bad > r2_ok + 0.005:
        log(f"  → 随机划分虚高 {(r2_bad - r2_ok):+.4f}：相邻样本（相隔 1 秒）几乎相同，")
        log(f"    同时出现在训练和测试中，等于把答案抄给模型。")
        log(f"    真实部署时性能会明显下降，这就是典型的时间序列数据泄漏。")
    else:
        log(f"  → 本次两者接近（差 {r2_bad - r2_ok:+.4f}）。原因：该仿真信号本身很平滑，")
        log(f"    Ridge 用「上一秒的值」就几乎能预测，泄漏带来的额外收益不明显。")
        log(f"    **但随机划分仍然是错的做法**：真实数据波动更大、有工况切换时，")
        log(f"    泄漏会让指标虚高好几倍。报告 7.1 要求按时间/批次划分，必须遵守。")

    # ---- 多种 Np 的样本量对比 ----
    log("\n【窗口长度敏感性】")
    log(f"  {'L':>5} {'Np':>5} {'样本数':>8} {'每样本特征':>10}")
    for L_ in (60, 120, 240):
        for Np_ in (15, 30, 60):
            try:
                Xa, Ya, ma, _, _, inf = build_windows(df_all, WindowConfig(L=L_, Np=Np_))
                log(f"  {L_:>5} {Np_:>5} {Xa.shape[0]:>8} {Xa.shape[1]:>10}")
            except RuntimeError:
                log(f"  {L_:>5} {Np_:>5} {'无样本':>8} {'-':>10}   (稳定段不够长)")

    # ---- 保存样例数据 ----
    np.savez_compressed(
        OUT / "window_dataset.npz",
        X_tr=Xtr, Y_tr=Ytr, X_va=Xva, Y_va=Yva, X_te=Xte, Y_te=Yte,
        scaler_mu=scaler["mu"], scaler_sd=scaler["sd"],
        scaler_tmu=scaler["tmu"], scaler_tsd=scaler["tsd"],
        feature_names=np.array(info["feature_names"]),
        out_names=np.array(info["out_names"]),
        L=cfg.L, Np=cfg.Np,
    )
    log(f"\n数据集已保存 -> {OUT/'window_dataset.npz'}")
    log(f"（真实数据来了以后，只需替换 df_all，其余代码不用改）")

    # ---- 绘图 ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    ax = axes[0, 0]
    df1 = batches[0]
    ax.plot(df1["timestamp"], df1["outlet_moisture"], color=COLORS["target"], lw=1.0, label="出口水分")
    st = df1["machine_state"].values
    for name, code in [("稳定", "stable"), ("收尾", "shutdown")]:
        m = st == STATE_CODES[code]
        if m.any():
            ax.fill_between(df1["timestamp"], 0, 1, where=m, alpha=.12,
                            transform=ax.get_xaxis_transform(), label=name)
    ax.set_title("批次 1 出口水分与工况", fontweight="bold"); ax.legend(fontsize=8)
    ax.set_ylabel("水分 (%)")

    ax = axes[0, 1]
    # 展示一个窗口的结构
    k = len(df1) // 2
    w_idx = np.arange(k - cfg.L + 1, k + 1)
    f_idx = np.arange(k + 1, k + 1 + cfg.Np)
    ax.plot(df1["timestamp"].iloc[w_idx], df1["outlet_moisture"].values[w_idx],
            color=COLORS["pred"], lw=1.5, label="历史 L（输入）")
    ax.plot(df1["timestamp"].iloc[f_idx], df1["outlet_moisture"].values[f_idx],
            color=COLORS["target"], lw=2.5, label="未来 Np（标签）")
    ax.axvline(df1["timestamp"].iloc[k], ls="--", c="k", lw=1, label="当前时刻 k")
    ax.set_title(f"单个样本结构：L={cfg.L} → Np={cfg.Np}", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(["训练", "验证", "测试"], [len(Xtr), len(Xva), len(Xte)],
           color=[PALETTE[0], PALETTE[1], PALETTE[2]])
    ax.set_title("批次级划分样本数（报告 7.1）", fontweight="bold")
    ax.set_ylabel("样本数")
    for i, v in enumerate([len(Xtr), len(Xva), len(Xte)]):
        ax.text(i, v, str(v), ha="center", va="bottom")

    ax = axes[1, 1]
    # 标签分布（多步）
    ax.hist(Y[:, 0, 0] * scaler["tsd"][0] + scaler["tmu"][0], bins=40,
            color=COLORS["target"], alpha=.7, label="1 步后水分")
    ax.hist(Y[:, 0, -1] * scaler["tsd"][0] + scaler["tmu"][0], bins=40,
            color=COLORS["pred"], alpha=.6, label=f"{cfg.Np} 步后水分")
    ax.set_title("标签分布（不同预测步长）", fontweight="bold")
    ax.set_xlabel("出口水分 (%)"); ax.legend(fontsize=8)

    fig.suptitle("滑动窗口样本构造（报告 6.3 / 7.1）", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "s3_window_builder.png", dpi=110); plt.close(fig)
    log(f"图已保存 -> {FIG/'s3_window_builder.png'}")

    (REP / "s3_window_builder.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()