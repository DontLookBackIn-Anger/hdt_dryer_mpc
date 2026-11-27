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
from s3_window_builder import WindowConfig, build_windows, split_by_batch, build_split_windows
from s4_multistep_eval import metrics_multi, MultiStepEvaluator
from s2_delay_identification import simulate_prbs_id_experiment, identify_mimo_fopdt

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
# 真实物理/辨识 FOPDT-MIMO 解析多步预测器 + 灰箱混合模型（Stage 2 & 3）
# ================================================================
class FOPDTMIMOPredictor:
    """基于 s2 辨识出的 MIMO FOPDT 参数的解析多步预测器。

    特点：
      - 真正使用 (K_m, K_t, tau_m, tau_t, theta) 递推预测未来 1..Np 步
      - 采用输出偏置校正（类似 DMC/Offset-free MPC），保证第 1 步平滑衔接当前测量值
      - 天然保证 100% 物理方向正确、可微分、毫秒级推理
    """

    def __init__(self, mimo_params: dict, feature_names: list, scaler: dict, L: int, Np: int, dt: float = 1.0):
        self.p = mimo_params
        self.feature_names = list(feature_names)
        self.scaler = scaler
        self.L = L
        self.Np = Np
        self.dt = dt
        self.u_cols = list(mimo_params["u_cols"])
        self.u_idx = [self.feature_names.index(c) for c in self.u_cols]
        self.y_idx = [self.feature_names.index("outlet_moisture"),
                      self.feature_names.index("outlet_temp")]

    def predict(self, X_flat: np.ndarray) -> np.ndarray:
        """输入展平标准化窗口 (n, n_feat*L)，返回标准化多步预测 (n, 2, Np)。"""
        n = len(X_flat)
        nf = len(self.feature_names)
        X_seq = X_flat.reshape(n, nf, self.L)
        # 反标准化出物理量的历史轨迹
        U_hist = np.zeros((n, 4, self.L), dtype=float)
        for ch, fi in enumerate(self.u_idx):
            U_hist[:, ch, :] = X_seq[:, fi, :] * self.scaler["sd"][fi] + self.scaler["mu"][fi] - self.p["u_base"][ch]
        y_now = np.column_stack([
            X_seq[:, self.y_idx[0], -1] * self.scaler["sd"][self.y_idx[0]] + self.scaler["mu"][self.y_idx[0]],
            X_seq[:, self.y_idx[1], -1] * self.scaler["sd"][self.y_idx[1]] + self.scaler["mu"][self.y_idx[1]],
        ])

        a_m = self.dt / (float(self.p["tau_m"]) + self.dt)
        a_t = self.dt / (float(self.p["tau_t"]) + self.dt)
        Km = np.asarray(self.p["K_m"], float)
        Kt = np.asarray(self.p["K_t"], float)
        thetas = np.asarray(self.p["theta"], int)

        # 用历史窗口后 60 步推演物理内部状态到当前时刻 k
        warm = min(self.L - int(thetas.max()) - 1, 80)
        start_t = self.L - warm
        xm = np.zeros(n)
        xt = np.zeros(n)
        for t in range(start_t, self.L):
            u_del = np.column_stack([U_hist[:, ch, max(0, t - thetas[ch])] for ch in range(4)])
            xm = xm + a_m * (u_del @ Km - xm)
            xt = xt + a_t * (u_del @ Kt - xt)

        # 当前时刻的偏置校正（y_meas(k) - y_phys(k)），用于抵消未测扰动与稳态漂移
        bias_m = y_now[:, 0] - (float(self.p["y_base"][0]) + xm)
        bias_t = y_now[:, 1] - (float(self.p["y_base"][1]) + xt)

        # 未来 Np 步递推（未知未来输入按零阶保持：u(k+h) = u(k)）
        Y_phys = np.zeros((n, 2, self.Np), dtype=float)
        for h in range(self.Np):
            # 未来时刻 k+1+h 对应的延迟输入索引 = (L - 1) + (h + 1) - theta
            u_del = np.column_stack([
                U_hist[:, ch, min(self.L - 1, max(0, self.L + h - thetas[ch]))]
                for ch in range(4)
            ])
            xm = xm + a_m * (u_del @ Km - xm)
            xt = xt + a_t * (u_del @ Kt - xt)
            Y_phys[:, 0, h] = float(self.p["y_base"][0]) + xm + bias_m
            Y_phys[:, 1, h] = float(self.p["y_base"][1]) + xt + bias_t

        # 回到标准化空间与评估器统一接口
        Y_std = np.zeros_like(Y_phys)
        Y_std[:, 0, :] = (Y_phys[:, 0, :] - self.scaler["tmu"][0]) / self.scaler["tsd"][0]
        Y_std[:, 1, :] = (Y_phys[:, 1, :] - self.scaler["tmu"][1]) / self.scaler["tsd"][1]
        return Y_std


# ================================================================
# 演示：跑 5 类代表性模型并全部记录（严格无泄漏 + 物理与灰箱对比）
# ================================================================
def main():
    print("=" * 76)
    print("第 5 项：MLflow 实验记录（多范式 + 物理 FOPDT + 灰箱混合模型）")
    print("=" * 76)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"数据库: {DB_PATH}")

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    # ---- 数据（严格防泄漏：先按批次切分，仅在训练集拟合 scaler）----
    batches = []
    for i, seed in enumerate((11, 22, 33, 44, 55)):
        d = simulate_hdt_full(n_steps=3600, seed=seed)
        d["batch_id"] = f"B{i+1:02d}"
        batches.append(d)
    df_all = pd.concat(batches, ignore_index=True)

    cfg = WindowConfig(L=120, Np=30, stride=2, require_stable=True)
    ds = build_split_windows(df_all, cfg, batch_col="batch_id")
    Xtr, Ytr, _, meta_tr, b_tr = ds["train"]
    Xva, Yva, _, meta_va, b_va = ds["val"]
    Xte, Yte, _, meta_te, b_te = ds["test"]
    scaler, info, split = ds["scaler"], ds["info"], ds["split"]
    log(f"样本（严格零泄漏）: 训练 {len(Xtr)} ({split['train']}) / "
        f"验证 {len(Xva)} ({split['val']}) / 测试 {len(Xte)} ({split['test']})")

    data_info = dict(
        n_rows=len(df_all), n_batches=df_all["batch_id"].nunique(),
        n_samples=len(Xtr) + len(Xva) + len(Xte), n_features=Xtr.shape[1],
        train_batches=",".join(split["train"]),
        val_batches=",".join(split["val"]),
        test_batches=",".join(split["test"]),
    )
    ev = MultiStepEvaluator(scaler)
    n_out, Np = Ytr.shape[1], Ytr.shape[2]
    results = []

    def _bench_us(fn, Xq, repeats=15):
        ts = []
        for _ in range(repeats):
            t1 = time.perf_counter(); fn(Xq); ts.append(time.perf_counter() - t1)
        return float(np.median(ts)) / len(Xq) * 1e6

    # ---- 模型 1: 同口径多步 Persistence（下限）----
    log("\n[1/5] Persistence-MultiStep（同口径多步下限）")
    nf = len(info["feature_names"])
    yi0 = info["feature_names"].index("outlet_moisture")
    yi1 = info["feature_names"].index("outlet_temp")

    def persist_predict(Xq):
        xs = Xq.reshape(len(Xq), nf, cfg.L)
        out = np.zeros((len(Xq), n_out, Np))
        out[:, 0, :] = xs[:, yi0, -1:]
        out[:, 1, :] = xs[:, yi1, -1:]
        return out

    pred_p = persist_predict(Xte)
    res_p = ev.evaluate(Y=Yte, batch_ids=b_te, model_name="Persistence-Multi", y_pred=pred_p)
    infer_p = _bench_us(persist_predict, Xte)
    phys_p = physical_check(persist_predict, info["feature_names"], scaler, cfg.L, Xte)
    with ExperimentLogger("Persistence-Multi", cfg, data_info) as lg:
        lg.log_metrics({"overall": {o: res_p["overall"][o] for o in CV_COLS}})
        lg.log_per_step(res_p["per_step"])
        lg.log_metrics({"overall.infer_us_per_sample": infer_p})
        lg.log_physical(phys_p)
    results.append(("Persistence", res_p, pred_p))
    log(f"  水分 RMSE={res_p['overall']['outlet_moisture']['RMSE']:.5f}  推理 {infer_p:.1f} μs")

    # ---- 模型 2: Ridge（线性 Direct 多步）----
    from sklearn.linear_model import Ridge
    log("\n[2/5] Ridge-Direct 线性基线")
    t0 = time.time()
    ridge_models = {(j, k): Ridge(alpha=50.0).fit(Xtr, Ytr[:, j, k])
                    for j in range(n_out) for k in range(Np)}

    class RidgeEns:
        def predict(self, Xq):
            out = np.zeros((len(Xq), n_out, Np))
            for (j, k), m in ridge_models.items():
                out[:, j, k] = m.predict(Xq)
            return out

    rens = RidgeEns()
    pred_r = rens.predict(Xte)
    res_r = ev.evaluate(Y=Yte, batch_ids=b_te, model_name="Ridge-Direct", y_pred=pred_r)
    infer_r = _bench_us(rens.predict, Xte)
    phys_r = physical_check(rens.predict, info["feature_names"], scaler, cfg.L, Xte)
    with ExperimentLogger("Ridge-Direct", cfg, data_info) as lg:
        lg.log_metrics({"overall": {o: res_r["overall"][o] for o in CV_COLS}})
        lg.log_per_step(res_r["per_step"])
        lg.log_metrics({"overall.infer_us_per_sample": infer_r,
                        "alpha": 50.0, "train_sec": time.time() - t0})
        lg.log_physical(phys_r)
    results.append(("Ridge-Direct", res_r, pred_r))
    log(f"  水分 RMSE={res_r['overall']['outlet_moisture']['RMSE']:.5f}  推理 {infer_r:.1f} μs")

    # ---- 模型 3: LightGBM-Direct（非线性树多步）----
    import lightgbm as lgb
    log("\n[3/5] LightGBM-Direct (max_depth=4, n_est=150)")
    t0 = time.time()
    models = {}
    for j in range(n_out):
        for k in range(Np):
            m = lgb.LGBMRegressor(n_estimators=150, max_depth=4,
                                  learning_rate=0.06, verbose=-1, random_state=0)
            m.fit(Xtr, Ytr[:, j, k])
            models[(j, k)] = m

    class LGBMEns:
        def predict(self, Xq):
            out = np.zeros((len(Xq), n_out, Np))
            for (j, k), m in models.items():
                out[:, j, k] = m.predict(Xq)
            return out

    lens = LGBMEns()
    pred_l = lens.predict(Xte)
    res_l = ev.evaluate(Y=Yte, batch_ids=b_te, model_name="LightGBM-Direct", y_pred=pred_l)
    infer_l = _bench_us(lens.predict, Xte)
    phys_l = physical_check(lens.predict, info["feature_names"], scaler, cfg.L, Xte)
    with ExperimentLogger("LightGBM-Direct", cfg, data_info) as lg:
        lg.log_metrics({"overall": {o: res_l["overall"][o] for o in CV_COLS}})
        lg.log_per_step(res_l["per_step"])
        lg.log_metrics({"overall.infer_us_per_sample": infer_l,
                        "max_depth": 4, "n_estimators": 150,
                        "train_sec": time.time() - t0})
        lg.log_physical(phys_l)
        lg.log_model(models)
    results.append(("LightGBM-Direct", res_l, pred_l))
    log(f"  水分 RMSE={res_l['overall']['outlet_moisture']['RMSE']:.5f}  推理 {infer_l:.1f} μs")

    # ---- 模型 4: 真实辨识 FOPDT-MIMO 解析多步模型 ----
    log("\n[4/5] FOPDT-MIMO 解析模型（加载 s2 辨识参数，物理递推多步）")
    t0 = time.time()
    mimo_path = OUT / "identified_mimo_model.npz"
    if mimo_path.exists():
        z = np.load(mimo_path, allow_pickle=True)
        mimo_params = {k: z[k] for k in z.files}
        mimo_params["u_cols"] = [str(x) for x in mimo_params["u_cols"]]
    else:
        mimo_params = identify_mimo_fopdt(simulate_prbs_id_experiment(2400, seed=123), dt=1.0)

    fopdt_pred = FOPDTMIMOPredictor(mimo_params, info["feature_names"], scaler, cfg.L, Np)
    pred_f = fopdt_pred.predict(Xte)
    res_f = ev.evaluate(Y=Yte, batch_ids=b_te, model_name="FOPDT-MIMO", y_pred=pred_f)
    infer_f = _bench_us(fopdt_pred.predict, Xte)
    phys_f = physical_check(fopdt_pred.predict, info["feature_names"], scaler, cfg.L, Xte)
    with ExperimentLogger("FOPDT-MIMO", cfg, data_info) as lg:
        lg.log_metrics({"overall": {o: res_f["overall"][o] for o in CV_COLS}})
        lg.log_per_step(res_f["per_step"])
        lg.log_metrics({"overall.infer_us_per_sample": infer_f,
                        "train_sec": time.time() - t0})
        lg.log_physical(phys_f)
        lg.set_tag("mpc_friendly", "true")
    results.append(("FOPDT-MIMO", res_f, pred_f))
    log(f"  水分 RMSE={res_f['overall']['outlet_moisture']['RMSE']:.5f}  推理 {infer_f:.1f} μs")

    # ---- 模型 5: 灰箱混合模型（Hybrid Gray-Box: FOPDT-MIMO + Ridge 残差学习）----
    log("\n[5/5] Hybrid-GrayBox 灰箱混合模型（物理 FOPDT + 正则残差补偿）")
    t0 = time.time()
    Ytr_phys = fopdt_pred.predict(Xtr)
    Etr = Ytr - Ytr_phys
    # 强正则（alpha=500）只学残差中的系统性非线性偏差，不破坏物理主通道方向
    res_models = {(j, k): Ridge(alpha=500.0).fit(Xtr, Etr[:, j, k])
                  for j in range(n_out) for k in range(Np)}

    class HybridGrayBox:
        def predict(self, Xq):
            base = fopdt_pred.predict(Xq)
            corr = np.zeros_like(base)
            for (j, k), m in res_models.items():
                corr[:, j, k] = m.predict(Xq)
            return base + corr

    hens = HybridGrayBox()
    pred_h = hens.predict(Xte)
    res_h = ev.evaluate(Y=Yte, batch_ids=b_te, model_name="Hybrid-GrayBox", y_pred=pred_h)
    infer_h = _bench_us(hens.predict, Xte)
    phys_h = physical_check(hens.predict, info["feature_names"], scaler, cfg.L, Xte)
    with ExperimentLogger("Hybrid-GrayBox", cfg, data_info) as lg:
        lg.log_metrics({"overall": {o: res_h["overall"][o] for o in CV_COLS}})
        lg.log_per_step(res_h["per_step"])
        lg.log_metrics({"overall.infer_us_per_sample": infer_h,
                        "resid_alpha": 500.0, "train_sec": time.time() - t0})
        lg.log_physical(phys_h)
        lg.set_tag("hybrid_graybox", "true")
    results.append(("Hybrid-GrayBox", res_h, pred_h))
    log(f"  水分 RMSE={res_h['overall']['outlet_moisture']['RMSE']:.5f}  推理 {infer_h:.1f} μs")

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