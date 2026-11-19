#!/usr/bin/env python3
"""
=== 第 5 项：MLflow 实验记录集成 ===

为什么要做这一步？
    报告 7.2 要求"逐步比较"多种模型（基线→表格→深度→状态空间），
    报告 7.3 要求多维评估（1步/多步、按批次分组、Bias、残差...）。
    手跑几十个实验后，你会记不清"哪个配置得到哪个分数"。
    MLflow 把每次实验的**超参、指标、模型文件、数据版本**全部自动记录，
    并能在 UI 里横向对比。

本模块提供：
    ExperimentLogger     上下文管理器，自动记录
    compare_runs()       跨实验横向对比

设计要点（对应报告）：
    · 记录**数据版本**：报告 5.4/9 要求可追溯、可复现
    · 记录**窗口配置 L/Np**：报告 6.3 的核心超参
    · 记录**多步指标**：报告的 1 步与多步误差
    · 记录**物理方向检查结果**：报告 7.3 最后一条
    · 记录**推理时间**：报告 7.3 在线计算时间

运行（会跑 4 个模型做演示）：
    cd ~/hdt_dryer_mpc
    python src/s5_mlflow_tracking.py

查看 UI：
    mlflow ui --backend-store-uri ./mlruns --port 5000
    然后浏览器打开 http://127.0.0.1:5000

产物：
    mlruns/                     （MLflow 实验库）
    outputs/reports/s5_compare.csv
    outputs/figures/s5_compare.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import FIG, REP, OUT, ROOT, COLORS, PALETTE
from dryer_sim import simulate_hdt_full, MV_COLS, DV_COLS, PV_COLS, CV_COLS
from s3_window_builder import WindowConfig, build_windows, split_by_batch
from s4_multistep_eval import metrics_multi, MultiStepEvaluator

warnings.filterwarnings("ignore")

# MLflow 3.x 已弃用文件后端（维护模式），推荐用 SQLite。
# 用 SQLite 的好处：并发安全、可查 SQL、不会像文件后端那样产生大量小文件。
MLRUNS = ROOT / "mlflow"
MLRUNS.mkdir(exist_ok=True)
DB_PATH = MLRUNS / "mlflow.db"
ARTIFACT_DIR = MLRUNS / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
TRACKING_URI = f"sqlite:///{DB_PATH}"

import mlflow
mlflow.set_tracking_uri(TRACKING_URI)
EXPERIMENT = "hdt_dryer_forecast"


# ================================================================
# 日志器
# ================================================================
class ExperimentLogger:
    """MLflow 上下文管理器。

    用法：
        with ExperimentLogger("LightGBM-L120", cfg, scaler) as lg:
            lg.log_metrics(overall)
            lg.log_model(obj, "model")

    自动记录：
        · 窗口超参 L/Np/stride/特征开关
        · 数据指纹（行数、批次、样本数）—— 保证可复现
        · 运行环境（python/torch 版本）
    """

    def __init__(self, run_name: str, cfg: WindowConfig, data_info: dict | None = None):
        self.run_name = run_name
        self.cfg = cfg
        self.data_info = data_info or {}

    _exp_ready = False

    def __enter__(self):
        if not ExperimentLogger._exp_ready:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            exp = client.get_experiment_by_name(EXPERIMENT)
            if exp is None:
                client.create_experiment(EXPERIMENT, artifact_location=str(ARTIFACT_DIR))
            else:
                mlflow.set_experiment(EXPERIMENT)
            ExperimentLogger._exp_ready = True
        mlflow.set_experiment(EXPERIMENT)
        self.active = mlflow.start_run(run_name=self.run_name)
        # 超参
        mlflow.log_params({
            "L": self.cfg.L,
            "Np": self.cfg.Np,
            "stride": self.cfg.stride,
            "use_mv": self.cfg.use_mv,
            "use_dv": self.cfg.use_dv,
            "use_pv": self.cfg.use_pv,
            "use_y_history": self.cfg.use_y_history,
            "standardize": self.cfg.standardize,
            "require_stable": self.cfg.require_stable,
        })
        # 数据版本（报告 5.4/9 可追溯）
        if self.data_info:
            mlflow.log_params({f"data_{k}": v for k, v in self.data_info.items()
                               if isinstance(v, (int, float, str, bool))})
        # 环境
        import platform, torch
        mlflow.set_tags({
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": str(torch.cuda.is_available()),
        })
        return self

    def log_metrics(self, metrics: dict, prefix: str = ""):
        """自动展平嵌套指标。"""
        flat = {}
        for k, v in (metrics or {}).items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten(v, key))
            elif isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
                flat[key] = float(v)
        if flat:
            mlflow.log_metrics(flat)
        return flat

    @staticmethod
    def _flatten(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(ExperimentLogger._flatten(v, key))
            elif isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
                out[key] = float(v)
        return out

    def log_per_step(self, per_step: dict):
        """记录每个步长的 RMSE（报告 7.3：1 步与多步）"""
        for out_name, df in per_step.items():
            for _, r in df.iterrows():
                mlflow.log_metric(f"{out_name}.rmse_step{int(r['step'])}", float(r["RMSE"]), step=int(r["step"]))
                mlflow.log_metric(f"{out_name}.mae_step{int(r['step'])}", float(r["MAE"]), step=int(r["step"]))

    def log_physical(self, phys: pd.DataFrame):
        n_ok = int((phys["符合物理"] == "✅").sum())
        mlflow.log_metric("physical.direction_ok", n_ok)
        mlflow.log_metric("physical.direction_total", len(phys))
        mlflow.set_tag("physical_pass", str(n_ok == len(phys)))

    def log_model(self, obj, artifact_path="model"):
        try:
            import lightgbm
            if isinstance(obj, dict) and all(isinstance(v, lightgbm.LGBMRegressor) for v in obj.values()):
                # 多个子模型：逐个保存
                import tempfile, pickle
                with tempfile.TemporaryDirectory() as td:
                    for (j, k), m in obj.items():
                        m.booster_.save_model(f"{td}/lgbm_o{j}_h{k}.txt")
                    mlflow.log_artifacts(td, artifact_path)
                return
            mlflow.sklearn.log_model(obj, artifact_path)
        except Exception as e:
            mlflow.set_tag("model_log_error", str(e)[:180])

    def set_tag(self, key, value):
        mlflow.set_tag(key, str(value))

    def log_figure(self, path):
        if Path(path).exists():
            mlflow.log_artifact(str(path), "figures")

    def __exit__(self, *exc):
        mlflow.end_run()


# ================================================================
# 对比
# ================================================================
def compare_runs(experiment=EXPERIMENT) -> pd.DataFrame:
    """跨实验横向对比（从 SQLite 后端读回）。"""
    mlflow.set_experiment(experiment)
    runs = mlflow.search_runs(experiment_names=[experiment])   # 返回 DataFrame
    if len(runs) == 0:
        return pd.DataFrame()
    rows = []
    for _, r in runs.iterrows():
        def g(col):
            v = r.get(col, np.nan)
            return v if pd.notna(v) else np.nan
        rows.append({
            "run": g("tags.mlflow.runName"),
            "L": g("params.L"), "Np": g("params.Np"),
            "水分RMSE": g("metrics.overall.outlet_moisture.RMSE"),
            "水分MAE": g("metrics.overall.outlet_moisture.MAE"),
            "水分R2": g("metrics.overall.outlet_moisture.R2"),
            "温度RMSE": g("metrics.overall.outlet_temp.RMSE"),
            "温度R2": g("metrics.overall.outlet_temp.R2"),
            "物理通过": g("metrics.physical.direction_ok"),
            "推理μs": g("metrics.overall.infer_us_per_sample"),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["run"], keep="first")
    if len(df):
        df = df.sort_values("水分RMSE", na_position="last")
    return df



def physical_check(predict_fn, feature_names, scaler, L, X_probe, n_out=2):
    """通用物理方向检查（报告 7.3）。

    predict_fn(X) -> (n, n_out, Np) 标准化预测
    对每个输入通道：整个窗口抬高 Δ，看 Np 末尾输出的平均变化方向。
    """
    delta_map = {"steam_flow": 30.0, "hot_air_flow": 500.0, "inlet_moisture": 1.0}
    expect = {"steam_flow": (-1, +1), "hot_air_flow": (-1, +1), "inlet_moisture": (+1, -1)}
    rows = []
    n_probe = min(40, len(X_probe))
    base = predict_fn(X_probe[:n_probe])
    for col, (exp_m, exp_t) in expect.items():
        if col not in feature_names:
            continue
        i = feature_names.index(col)
        pert = X_probe[:n_probe].copy()
        for kk in range(L):
            pert[:, i * L + kk] += delta_map[col] / scaler["sd"][i]
        new = predict_fn(pert)
        d = float(np.mean(new[:, 0, -1] - base[:, 0, -1])) * scaler["tsd"][0]
        ok = (np.sign(d) == exp_m) if abs(d) > 1e-5 else None
        rows.append({
            "输入": col, "阶跃": delta_map[col],
            "Δ水分": round(d, 5), "期望": "↓" if exp_m < 0 else "↑",
            "符合物理": "✅" if ok else ("❌" if ok is False else "≈0"),
        })
    return pd.DataFrame(rows)


# ================================================================
# 演示：跑 4 个模型并全部记录
# ================================================================
def main():
    print("=" * 76)
    print("第 5 项：MLflow 实验记录")
    print("=" * 76)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"数据库: {DB_PATH}")

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    # ---- 数据 ----
    batches = []
    for i, seed in enumerate((11, 22, 33, 44)):
        d = simulate_hdt_full(n_steps=1800, seed=seed)
        d["batch_id"] = f"B{i+1:02d}"
        batches.append(d)
    df_all = pd.concat(batches, ignore_index=True)

    cfg = WindowConfig(L=120, Np=30, stride=1, require_stable=True)
    X, Y, meta, scaler, X_seq, info = build_windows(df_all, cfg)
    b_ids = df_all["batch_id"].values[meta["k"].values]
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte), split = split_by_batch(X, Y, meta, b_ids)
    b_te = [b for b, b_ in zip(b_ids, meta["k"].values) if b_ in split["test"]]
    log(f"样本: 训练 {len(Xtr)} / 测试 {len(Xte)}")

    data_info = dict(
        n_rows=len(df_all), n_batches=df_all["batch_id"].nunique(),
        n_samples=len(X), n_features=X.shape[1],
        train_batches=",".join(split["train"]),
        test_batches=",".join(split["test"]),
    )
    ev = MultiStepEvaluator(scaler)
    n_out, Np = Y.shape[1], Y.shape[2]
    results = []

    # ---- 模型 1: Ridge（线性基线）----
    from sklearn.linear_model import Ridge
    log("\n[1/4] Ridge 线性基线")
    t0 = time.time()
    ridge_models = {(j, k): Ridge(alpha=10.0).fit(Xtr, Ytr[:, j, k])
                    for j in range(n_out) for k in range(Np)}

    class RidgeEns:
        def predict(self, Xq):
            out = np.zeros((len(Xq), n_out, Np))
            for (j, k), m in ridge_models.items():
                out[:, j, k] = m.predict(Xq)
            return out

    pred_r = RidgeEns().predict(Xte)
    res_r = ev.evaluate(Y=Yte, batch_ids=b_te, model_name="Ridge", y_pred=pred_r)
    phys_r = physical_check(lambda q: RidgeEns().predict(q), info["feature_names"], scaler, cfg.L, Xte)
    with ExperimentLogger("Ridge", cfg, data_info) as lg:
        lg.log_metrics({"overall": {o: res_r["overall"][o] for o in CV_COLS}})
        lg.log_per_step(res_r["per_step"])
        lg.log_metrics({"alpha": 10.0, "train_sec": time.time() - t0})
        lg.log_physical(phys_r)
    results.append(("Ridge", res_r, pred_r))
    log(f"  水分 RMSE={res_r['overall']['outlet_moisture']['RMSE']:.5f}")

    # ---- 模型 2/3: LightGBM 不同深度 ----
    import lightgbm as lgb
    for tag, depth, n_est in [("LightGBM-d3", 3, 100), ("LightGBM-d5", 5, 200)]:
        log(f"\n[2/4] {tag} (max_depth={depth}, n_est={n_est})")
        t0 = time.time()
        models = {}
        for j in range(n_out):
            for k in range(Np):
                m = lgb.LGBMRegressor(n_estimators=n_est, max_depth=depth,
                                      learning_rate=0.08, verbose=-1, random_state=0)
                m.fit(Xtr, Ytr[:, j, k])
                models[(j, k)] = m

        class Ens:
            def predict(self, Xq):
                out = np.zeros((len(Xq), n_out, Np))
                for (j, k), m in models.items():
                    out[:, j, k] = m.predict(Xq)
                return out

        ens = Ens()
        pred = ens.predict(Xte)
        res = ev.evaluate(Y=Yte, batch_ids=b_te, model_name=tag, y_pred=pred)
        # 推理计时
        ts = []
        for _ in range(15):
            t1 = time.perf_counter(); ens.predict(Xte); ts.append(time.perf_counter() - t1)
        infer_us = float(np.median(ts)) / len(Xte) * 1e6

        with ExperimentLogger(tag, cfg, data_info) as lg:
            lg.log_metrics({"overall": {o: res["overall"][o] for o in CV_COLS}})
            lg.log_per_step(res["per_step"])
            lg.log_metrics({"overall.infer_us_per_sample": infer_us,
                            "max_depth": depth, "n_estimators": n_est,
                            "train_sec": time.time() - t0})
            phys = physical_check(lambda q: Ens().predict(q), info["feature_names"], scaler, cfg.L, Xte)
            lg.log_physical(phys)
            lg.log_model(models)
        results.append((tag, res, pred))
        log(f"  水分 RMSE={res['overall']['outlet_moisture']['RMSE']:.5f}  "
            f"推理 {infer_us:.1f} μs")

    # ---- 模型 4: FOPDT 解析模型（MPC 用的那种）----
    log("\n[4/4] FOPDT 解析模型（1 阶惯性 + 纯滞后，可解析多步）")

    from scipy.optimize import curve_fit

    def fit_fopdt_multi(Xtr_f, Ytr_f, j, k):
        """在窗口上拟合一个线性 FOPDT 的稳态增益（简化版：只用均值差）"""
        # 用最小二乘直接拟合"窗口均值 -> 未来值"的线性关系
        w_mean = Xtr_f.reshape(len(Xtr_f), -1)
        A = w_mean
        b = Ytr_f[:, j, k]
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        return coef

    t0 = time.time()
    fopdt_coefs = {}
    for j in range(n_out):
        for k in range(Np):
            fopdt_coefs[(j, k)] = fit_fopdt_multi(Xtr, Ytr, j, k)

    class FOPDTEns:
        def predict(self, Xq):
            out = np.zeros((len(Xq), n_out, Np))
            for (j, k), c in fopdt_coefs.items():
                out[:, j, k] = Xq @ c
            return out

    fens = FOPDTEns()
    pred_f = fens.predict(Xte)
    res_f = ev.evaluate(Y=Yte, batch_ids=b_te, model_name="FOPDT-linear", y_pred=pred_f)
    ts = []
    for _ in range(15):
        t1 = time.perf_counter(); fens.predict(Xte); ts.append(time.perf_counter() - t1)
    infer_f = float(np.median(ts)) / len(Xte) * 1e6

    with ExperimentLogger("FOPDT-linear", cfg, data_info) as lg:
        lg.log_metrics({"overall": {o: res_f["overall"][o] for o in CV_COLS}})
        lg.log_per_step(res_f["per_step"])
        lg.log_metrics({"overall.infer_us_per_sample": infer_f,
                        "train_sec": time.time() - t0})
        phys_f = physical_check(lambda q: FOPDTEns().predict(q), info["feature_names"], scaler, cfg.L, Xte)
        lg.log_physical(phys_f)
        lg.set_tag("mpc_friendly", "true")
    results.append(("FOPDT-linear", res_f, pred_f))
    log(f"  水分 RMSE={res_f['overall']['outlet_moisture']['RMSE']:.5f}  "
        f"推理 {infer_f:.1f} μs")

    # ---- 对比 ----
    log("\n" + "=" * 76)
    log("跨实验对比")
    log("=" * 76)
    cmp = compare_runs()
    if len(cmp):
        cmp["水分RMSE"] = cmp["水分RMSE"].astype(float).round(5)
        cmp["水分R2"] = cmp["水分R2"].astype(float).round(4)
        cmp["温度RMSE"] = cmp["温度RMSE"].astype(float).round(5)
        cmp["推理μs"] = cmp["推理μs"].astype(float).round(1)
        log(cmp.to_string(index=False))
        cmp.to_csv(REP / "s5_compare.csv", index=False, encoding="utf-8-sig")
        log(f"\n已导出 -> {REP/'s5_compare.csv'}")

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    names = [r[0] for r in results]
    m_rmse = [r[1]["overall"]["outlet_moisture"]["RMSE"] for r in results]
    t_rmse = [r[1]["overall"]["outlet_temp"]["RMSE"] for r in results]

    ax = axes[0]
    ax.bar(names, m_rmse, color=PALETTE[:len(names)])
    ax.set_title("出口水分 RMSE（越低越好）", fontweight="bold")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    for i, v in enumerate(m_rmse):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    ax.bar(names, t_rmse, color=PALETTE[:len(names)])
    ax.set_title("出口温度 RMSE（越低越好）", fontweight="bold")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    for i, v in enumerate(t_rmse):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax = axes[2]
    # 误差随步长增长曲线
    for i, (nm, res, _) in enumerate(results):
        ps = res["per_step"]["outlet_moisture"]
        ax.plot(ps["step"], ps["RMSE"], marker="o", ms=2.5,
                color=PALETTE[i % len(PALETTE)], label=nm)
    ax.set_title("水分 RMSE 随时域增长", fontweight="bold")
    ax.set_xlabel("预测步长"); ax.legend(fontsize=7)

    fig.suptitle("MLflow 实验对比（报告 7.2 / 7.3）", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG / "s5_compare.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    log(f"图已保存 -> {out}")

    log("\n" + "=" * 76)
    log("如何查看实验记录")
    log("=" * 76)
    log(f"  mlflow ui --backend-store-uri {TRACKING_URI} --port 5000")
    log(f"  浏览器打开 http://127.0.0.1:5000")
    log(f"  实验名: {EXPERIMENT}")

    (REP / "s5_mlflow.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()