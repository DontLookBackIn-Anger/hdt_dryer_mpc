import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import LightGBMModel, TCNModel
from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import LeastSquares


def simulate_hdt_batch(n_steps: int = 1200, seed: int = 42) -> pd.DataFrame:
    """模拟单个 HDT 气流烘丝机稳定生产批次（1 Hz 采样）"""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    # 1. 操纵量 (MV) 与入口扰动 (DV)：包含阶跃与慢变，确保独立激励
    steam_flow = 260.0 + 35.0 * np.sin(2 * np.pi * t / 300.0) + rng.normal(0, 2.0, n_steps)
    steam_flow[400:700] += 30.0  # 模拟蒸汽阶跃调节

    hot_air_flow = 9500.0 + 500.0 * np.cos(2 * np.pi * t / 420.0) + rng.normal(0, 40.0, n_steps)
    hot_air_flow[750:1000] -= 450.0  # 模拟热风独立变化

    inlet_moisture = 20.0 + 0.8 * np.sin(2 * np.pi * t / 500.0) + rng.normal(0, 0.1, n_steps)
    feed_rate = 3800.0 + 200.0 * np.sin(2 * np.pi * t / 600.0) + rng.normal(0, 20.0, n_steps)

    # 2. 带有物料输送滞后 + 一阶惯性的双输出动态响应
    w_out = np.full(n_steps, 12.5, dtype=np.float32)
    t_out = np.full(n_steps, 65.0, dtype=np.float32)

    for k in range(20, n_steps):
        # 模拟 12~18 秒工艺与输送延迟
        u_steam = steam_flow[k - 14]
        u_air = hot_air_flow[k - 12]
        d_moist = inlet_moisture[k - 18]
        d_feed = feed_rate[k - 16]

        # 稳态目标近似映射
        w_eq = (
            12.5
            + 0.65 * (d_moist - 20.0)
            + 0.0012 * (d_feed - 3800.0)
            - 0.015 * (u_steam - 260.0)
            - 0.0008 * (u_air - 9500.0)
        )
        t_eq = (
            65.0
            + 0.08 * (u_steam - 260.0)
            + 0.0025 * (u_air - 9500.0)
            - 0.40 * (d_moist - 20.0)
            - 0.0020 * (d_feed - 3800.0)
        )

        # 一阶惯性滤波 (tau ≈ 15s)
        alpha = 1.0 / 15.0
        w_out[k] = (1 - alpha) * w_out[k - 1] + alpha * w_eq + rng.normal(0, 0.03)
        t_out[k] = (1 - alpha) * t_out[k - 1] + alpha * t_eq + rng.normal(0, 0.10)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-11-01", periods=n_steps, freq="1s"),
            "steam_flow": steam_flow.astype(np.float32),
            "hot_air_flow": hot_air_flow.astype(np.float32),
            "inlet_moisture": inlet_moisture.astype(np.float32),
            "feed_rate": feed_rate.astype(np.float32),
            "outlet_moisture": w_out,
            "outlet_temp": t_out,
        }
    )


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    input_cols = ["steam_flow", "hot_air_flow", "inlet_moisture", "feed_rate"]
    target_cols = ["outlet_moisture", "outlet_temp"]
    horizon = 15  # 预测未来 15 秒

    # 生成 3 个训练批次 + 1 个独立测试批次
    train_dfs = [simulate_hdt_batch(1200, seed=s) for s in (10, 20, 30)]
    test_df = simulate_hdt_batch(1200, seed=99)

    results = {}

    # ==========================================================
    # 1. SysIdentPy NARX 基线（分别对水分和温度建立 MISO 模型）
    # ==========================================================
    print("正在训练 SysIdentPy NARX ...")
    X_train_narx = np.vstack([df[input_cols].values for df in train_dfs])
    X_test_narx = test_df[input_cols].values

    # 简单按训练集做标准化，提升多项式回归数值稳定性
    x_mean, x_std = X_train_narx.mean(axis=0), X_train_narx.std(axis=0) + 1e-8
    X_train_norm = (X_train_narx - x_mean) / x_std
    X_test_norm = (X_test_narx - x_mean) / x_std

    narx_rmses = []
    for col in target_cols:
        y_train = np.concatenate([df[col].values for df in train_dfs]).reshape(-1, 1)
        y_test = test_df[col].values.reshape(-1, 1)

        model_narx = FROLS(
            order_selection=True,
            n_terms=12,
            ylag=5,
            xlag=[list(range(1, 19))] * len(input_cols),
            basis_function=Polynomial(degree=2),
            estimator=LeastSquares(),
        )
        model_narx.fit(X=X_train_norm, y=y_train)
        yhat = model_narx.predict(X=X_test_norm, y=y_test, steps_ahead=horizon)
        narx_rmses.append(calc_rmse(y_test[18:].ravel(), yhat[18:].ravel()))

    results["SysIdentPy NARX"] = narx_rmses

    # ==========================================================
    # 构造 Darts 多批次 TimeSeries（仅在训练集 fit Scaler）
    # ==========================================================
    train_targets = [
        TimeSeries.from_dataframe(df, "timestamp", target_cols).astype(np.float32)
        for df in train_dfs
    ]
    train_covs = [
        TimeSeries.from_dataframe(df, "timestamp", input_cols).astype(np.float32)
        for df in train_dfs
    ]
    test_target = TimeSeries.from_dataframe(test_df, "timestamp", target_cols).astype(np.float32)
    test_cov = TimeSeries.from_dataframe(test_df, "timestamp", input_cols).astype(np.float32)

    scaler_y = Scaler()
    scaler_x = Scaler()
    train_targets_s = scaler_y.fit_transform(train_targets)
    train_covs_s = scaler_x.fit_transform(train_covs)
    test_target_s = scaler_y.transform(test_target)
    test_cov_s = scaler_x.transform(test_cov)

    # ==========================================================
    # 2. Darts LightGBM（动态滞后表格基线）
    # ==========================================================
    print("正在训练 Darts LightGBM ...")
    lgbm = LightGBMModel(
        lags=30,
        lags_past_covariates=30,
        output_chunk_length=horizon,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(series=train_targets_s, past_covariates=train_covs_s)

    pred_lgbm_s = lgbm.historical_forecasts(
        series=test_target_s,
        past_covariates=test_cov_s,
        start=60,
        forecast_horizon=horizon,
        stride=1,
        retrain=False,
        last_points_only=True,
        verbose=False,
    )
    pred_lgbm = scaler_y.inverse_transform(pred_lgbm_s)
    true_aligned = test_target.slice_intersect(pred_lgbm)
    results["Darts LightGBM"] = [
        calc_rmse(true_aligned[c].values().ravel(), pred_lgbm[c].values().ravel())
        for c in target_cols
    ]

    # ==========================================================
    # 3. Darts TCN（多输出因果膨胀卷积，自动调用 GPU）
    # ==========================================================
    print("正在训练 Darts TCN ...")
    tcn = TCNModel(
        input_chunk_length=60,
        output_chunk_length=horizon,
        kernel_size=3,
        num_filters=32,
        dilation_base=2,
        dropout=0.1,
        n_epochs=25,
        batch_size=64,
        random_state=42,
        pl_trainer_kwargs={"accelerator": "auto", "enable_progress_bar": False},
    )
    tcn.fit(series=train_targets_s, past_covariates=train_covs_s)

    pred_tcn_s = tcn.historical_forecasts(
        series=test_target_s,
        past_covariates=test_cov_s,
        start=60,
        forecast_horizon=horizon,
        stride=1,
        retrain=False,
        last_points_only=True,
        verbose=False,
    )
    pred_tcn = scaler_y.inverse_transform(pred_tcn_s)
    true_aligned_tcn = test_target.slice_intersect(pred_tcn)
    results["Darts TCN"] = [
        calc_rmse(true_aligned_tcn[c].values().ravel(), pred_tcn[c].values().ravel())
        for c in target_cols
    ]

    # ==========================================================
    # 输出对比结果
    # ==========================================================
    print("\n" + "=" * 56)
    print(f"{'模型':<20} | {'出口含水率 RMSE (%)':<16} | {'出口温度 RMSE (℃)'}")
    print("-" * 56)
    for name, (m_rmse, t_rmse) in results.items():
        print(f"{name:<20} | {m_rmse:<18.4f} | {t_rmse:.4f}")
    print("=" * 56)


if __name__ == "__main__":
    main()