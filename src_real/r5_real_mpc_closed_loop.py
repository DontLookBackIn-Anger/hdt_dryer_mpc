#!/usr/bin/env python3
"""
=== 真实数据脚本 5：基于真实加料回潮辨识模型的 MPC 闭环仿真验证 ===

与仿真线 src/s6_mpc_closed_loop.py 严格隔离，针对真实加料回潮工序（1B_steady.csv 工况）：
  - 稳态工作点（取自 1B_steady.csv 真实均值）：
      MV1: 补偿蒸汽流量 steam_flow       ≈ 75.0 kg/h  (约束 [55, 105], |Δu| ≤ 2.5/s)
      MV2: 排潮风量     exhaust_air_flow ≈ 2800 m³/h  (约束 [2550, 3050], |Δu| ≤ 25/s)
      DV1: 入口含水率   inlet_moisture   ≈ 16.68 %    (直接回放 1B 真实实测扰动轨迹！)
      DV2: 回风温度     return_air_temp  ≈ 53.37 ℃    (直接回放 1B 真实实测扰动轨迹！)
      CV1: 出口含水率   outlet_moisture  ≈ 20.83 %
      CV2: 烟叶出口温度 outlet_temp      ≈ 49.00 ℃
  - 动力学参数（结合 r2 真实滞后辨识与物理机理标定）：
      纯滞后 θ_steam ≈ 35s, θ_exhaust ≈ 40s, θ_inlet ≈ 45s；一阶时间常数 τ_m = 60s, τ_t = 50s；
      物理增益方向严格满足加料回潮机理（蒸汽↑ → 水分↑、温度↑；排潮风量↑ → 水分↓、温度↓）。
  - 控制器对比：
      ① 工业双回路 PID（蒸汽控水分、排潮控温度，带抗积分饱和）
      ② 真实加料回潮 MIMO DMC/MPC（预测时域 Hp=90s > 纯滞后 35s，控制时域 Hu=15s，
         带执行器幅值/速率硬约束 + 滚动输出误差校正 d(k)=y_meas-y_mod 实现无静差跟踪 + 可测前馈）

  - 3 大闭环验证场景：
      1) 设定值跟踪（水分 +0.25% / 温度 +1.0℃ 阶跃）
      2) 真实扰动抑制（回放 1B_steady.csv 实测入口含水率与回风温度波动）
      3) 模型失配鲁棒性（真实对象增益 +20% 且纯滞后 +6s 失配）

运行：
    cd ~/hdt_dryer_mpc
    /opt/conda_envs/dryer-ts/bin/python src_real/r5_real_mpc_closed_loop.py

产出（独立留痕于 outputs_real/）：
    outputs_real/reports/r5_real_mpc_metrics.csv
    outputs_real/reports/r5_real_mpc_report.txt
    outputs_real/figures/r5_real_mpc_closed_loop.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from real_common import REAL_PROC, REAL_FIG, REAL_REP, COLORS, save_real_report


# ---------------------------------------------------------------- 1. 加料回潮 MIMO FOPDT 对象模型
class RealCasingPlant:
    """基于 1B 真实工况与 r2 辨识滞后标定的加料回潮 2×2 MIMO + 2 DV 动态对象。"""

    def __init__(self, gain_scale: float = 1.0, delay_offset: int = 0, noise_seed: int = 42):
        # 基准工作点（取自 1B_steady.csv）
        self.u0 = np.array([75.0, 2800.0])       # [steam_flow, exhaust_air_flow]
        self.d0 = np.array([16.68, 53.37])       # [inlet_moisture, return_air_temp]
        self.y0 = np.array([20.83, 49.00])       # [outlet_moisture, outlet_temp]

        # 物理增益矩阵 K_u (2x2): 行=[水分, 温度], 列=[蒸汽, 排潮风量]
        # 加蒸汽 +10 kg/h -> 水分 +0.22%, 温度 +0.75℃；加排潮 +100 -> 水分 -0.06%, 温度 -0.25℃
        self.Ku = np.array([
            [+0.022, -0.0006],
            [+0.075, -0.0025],
        ]) * gain_scale

        # 扰动增益矩阵 K_d (2x2): 行=[水分, 温度], 列=[入口含水率, 回风温度]
        self.Kd = np.array([
            [+0.55, -0.03],
            [-0.40, +0.45],
        ]) * gain_scale

        # 纯滞后（秒，来自 r2 辨识区间）
        self.th_u = [max(1, 35 + delay_offset), max(1, 40 + delay_offset)]
        self.th_d = [max(1, 45 + delay_offset), max(1, 20 + delay_offset)]
        # 一阶惯性时间常数（秒）
        self.tau = np.array([60.0, 50.0])
        self.rng = np.random.default_rng(noise_seed)
        self.reset()

    def reset(self):
        self.max_delay = max(self.th_u + self.th_d) + 5
        self.u_hist = [self.u0.copy() for _ in range(self.max_delay)]
        self.d_hist = [self.d0.copy() for _ in range(self.max_delay)]
        self.y_state = self.y0.copy()

    def step(self, u: np.ndarray, d: np.ndarray, add_noise: bool = True) -> np.ndarray:
        self.u_hist.append(np.asarray(u, dtype=float).copy())
        self.d_hist.append(np.asarray(d, dtype=float).copy())

        du_eff = np.array([
            self.u_hist[-1 - self.th_u[0]][0] - self.u0[0],
            self.u_hist[-1 - self.th_u[1]][1] - self.u0[1],
        ])
        dd_eff = np.array([
            self.d_hist[-1 - self.th_d[0]][0] - self.d0[0],
            self.d_hist[-1 - self.th_d[1]][1] - self.d0[1],
        ])
        y_ss = self.y0 + self.Ku @ du_eff + self.Kd @ dd_eff
        alpha = 1.0 / (self.tau + 1.0)
        self.y_state = self.y_state + alpha * (y_ss - self.y_state)
        if add_noise:
            noise = np.array([self.rng.normal(0, 0.005), self.rng.normal(0, 0.025)])
            return self.y_state + noise
        return self.y_state.copy()


# ---------------------------------------------------------------- 2. 工业双回路 PID 与 MIMO DMC/MPC
class DecentralizedPID:
    """现场常见双回路 PI 控制器（蒸汽控水分、排潮控温度，带速率限幅与抗饱和）。"""

    def __init__(self, u0, u_min, u_max, du_max):
        self.u = u0.copy()
        self.u_min, self.u_max, self.du_max = u_min, u_max, du_max
        self.kp = np.array([8.0, -35.0])
        self.ki = np.array([0.12, -0.55])
        self.e_prev = np.zeros(2)

    def act(self, y: np.ndarray, sp: np.ndarray) -> np.ndarray:
        e = sp - y
        # 增量式 PI（天然抗积分饱和）
        du = self.kp * (e - self.e_prev) + self.ki * e
        du = np.clip(du, -self.du_max, self.du_max)
        self.u = np.clip(self.u + du, self.u_min, self.u_max)
        self.e_prev = e.copy()
        return self.u.copy()


class RealMimoDMC:
    """基于阶跃响应矩阵的约束 MIMO DMC/MPC（带滚动输出误差校正与可测前馈）。"""

    def __init__(self, model: RealCasingPlant, Hp: int = 90, Hu: int = 12,
                 u_min=None, u_max=None, du_max=None, q_w=(120.0, 6.0), r_w=(0.08, 0.0004)):
        self.model = model
        self.Hp, self.Hu = Hp, Hu
        self.u_min, self.u_max, self.du_max = u_min, u_max, du_max
        self.q_w = np.asarray(q_w, dtype=float)
        self.r_w = np.asarray(r_w, dtype=float)

        # 离线构建 2×2 动态阶跃响应矩阵 S_u (Hp, 2, 2) 与扰动阶跃矩阵 S_d (Hp, 2, 2)
        self.Su = np.zeros((Hp, 2, 2))
        for j in range(2):
            m = RealCasingPlant(gain_scale=1.0, delay_offset=0)
            u_step = m.u0.copy(); u_step[j] += 1.0
            for k in range(Hp):
                yk = m.step(u_step, m.d0, add_noise=False)
                self.Su[k, :, j] = yk - m.y0

        # 组装压缩脉冲控制矩阵 A (2*Hp, 2*Hu)
        self.A = np.zeros((2 * Hp, 2 * Hu))
        for p in range(Hp):
            for c in range(min(Hu, p + 1)):
                self.A[2 * p : 2 * p + 2, 2 * c : 2 * c + 2] = self.Su[p - c]

        # 构建二次规划解析矩阵 H = A^T Q A + R（每步仅需 24 维极快投影求解）
        Q_diag = np.tile(self.q_w, Hp)
        R_diag = np.tile(self.r_w, Hu)
        self.Q_diag = Q_diag
        self.H = (self.A.T * Q_diag) @ self.A + np.diag(R_diag)
        # 内部影子模型用于滚动误差校正
        self.shadow = RealCasingPlant(gain_scale=1.0, delay_offset=0)
        self.u_curr = model.u0.copy()
        self.d_curr = model.d0.copy()
        self.y_mod_curr = model.y0.copy()

    def act(self, y_meas: np.ndarray, d_meas: np.ndarray, sp: np.ndarray) -> np.ndarray:
        # 1. 滚动输出误差校正（DMC 核心：消除模型失配和未测扰动静差！）
        dist_bias = y_meas - self.y_mod_curr

        # 2. 用影子模型预测未来 Hp 步自由响应（假设未来 Δu = 0，d 保持当前值）
        free_model = RealCasingPlant(gain_scale=1.0, delay_offset=0)
        free_model.u_hist = [x.copy() for x in self.shadow.u_hist]
        free_model.d_hist = [x.copy() for x in self.shadow.d_hist]
        free_model.y_state = self.shadow.y_state.copy()

        y_free = np.zeros(2 * self.Hp)
        for p in range(self.Hp):
            yp = free_model.step(self.u_curr, d_meas, add_noise=False) + dist_bias
            y_free[2 * p : 2 * p + 2] = yp

        sp_vec = np.tile(sp, self.Hp)
        err_free = sp_vec - y_free  # (2*Hp,)

        # 3. 求解无约束最优 ΔU，再做速率+幅值硬约束裁剪（1ms 级工业 DMC 实时解法）
        g = (self.A.T * self.Q_diag) @ err_free
        du_opt = np.linalg.solve(self.H, g)
        du0 = np.clip(du_opt[:2], -self.du_max, self.du_max)
        u_next = np.clip(self.u_curr + du0, self.u_min, self.u_max)

        # 4. 推进影子模型
        self.u_curr = u_next.copy()
        self.d_curr = d_meas.copy()
        self.y_mod_curr = self.shadow.step(self.u_curr, self.d_curr, add_noise=False)
        return u_next


# ---------------------------------------------------------------- 3. 3 大工业场景闭环仿真
def run_scenario(name: str, n_steps: int, sp_traj: np.ndarray, d_traj: np.ndarray,
                 gain_scale: float = 1.0, delay_offset: int = 0) -> tuple[dict, dict]:
    u_min = np.array([55.0, 2550.0])
    u_max = np.array([105.0, 3050.0])
    du_max = np.array([2.5, 25.0])

    results = {}
    metrics = {}
    for ctrl_name in ["PID", "Real_MPC"]:
        plant = RealCasingPlant(gain_scale=gain_scale, delay_offset=delay_offset, noise_seed=42)
        if ctrl_name == "PID":
            ctrl = DecentralizedPID(plant.u0, u_min, u_max, du_max)
        else:
            ctrl = RealMimoDMC(RealCasingPlant(), Hp=90, Hu=12, u_min=u_min, u_max=u_max, du_max=du_max,
                               q_w=(120.0, 8.0), r_w=(0.25, 0.0012))

        Y = np.zeros((n_steps, 2))
        U = np.zeros((n_steps, 2))
        y_curr = plant.y0.copy()
        for k in range(n_steps):
            if ctrl_name == "PID":
                u_k = ctrl.act(y_curr, sp_traj[k])
            else:
                u_k = ctrl.act(y_curr, d_traj[k], sp_traj[k])
            y_curr = plant.step(u_k, d_traj[k], add_noise=True)
            Y[k], U[k] = y_curr, u_k

        err = Y - sp_traj
        rmse_m = float(np.sqrt(np.mean(err[:, 0] ** 2)))
        rmse_t = float(np.sqrt(np.mean(err[:, 1] ** 2)))
        iae_m = float(np.mean(np.abs(err[:, 0])))
        iae_t = float(np.mean(np.abs(err[:, 1])))
        tv_steam = float(np.mean(np.abs(np.diff(U[:, 0]))))
        results[ctrl_name] = {"Y": Y, "U": U, "SP": sp_traj}
        metrics[ctrl_name] = {
            "scenario": name, "controller": ctrl_name,
            "moist_rmse": round(rmse_m, 4), "moist_mae": round(iae_m, 4),
            "temp_rmse": round(rmse_t, 4), "temp_mae": round(iae_t, 4),
            "steam_tv": round(tv_steam, 4),
        }
    return results, metrics


def main():
    df_1b = pd.read_csv(REAL_PROC / "1B_steady.csv")
    N = 600  # 每场景仿真 600 秒（10 分钟）
    y0 = np.array([20.83, 49.00])
    d0 = np.array([16.68, 53.37])

    # 场景 1：设定值跟踪（水分 +0.25%, 温度 +1.0℃）
    sp1 = np.tile(y0, (N, 1))
    sp1[120:, 0] += 0.25
    sp1[300:, 1] += 1.00
    d1 = np.tile(d0, (N, 1))

    # 场景 2：真实扰动抑制（直接回放 1B_steady.csv 中 600s 实测入口水分与回风温度！）
    sp2 = np.tile(y0, (N, 1))
    d2 = df_1b[["inlet_moisture", "return_air_temp"]].iloc[200 : 200 + N].to_numpy(dtype=float)

    # 场景 3：对象-模型失配鲁棒性（真实增益 +20% 且纯滞后 +6s，叠加设定值阶跃+真实扰动）
    sp3 = sp1.copy()
    d3 = d2.copy()

    scenarios = [
        ("1_Setpoint_Tracking", sp1, d1, 1.0, 0),
        ("2_Real_1B_Disturbance", sp2, d2, 1.0, 0),
        ("3_Plant_Model_Mismatch", sp3, d3, 1.20, 6),
    ]

    all_metrics = []
    fig, axes = plt.subplots(3, 2, figsize=(14, 10.5), sharex=True)

    for row_i, (sname, sp, d, gs, doff) in enumerate(scenarios):
        res, mets = run_scenario(sname, N, sp, d, gain_scale=gs, delay_offset=doff)
        all_metrics.extend([mets["PID"], mets["Real_MPC"]])

        # 左：出口含水率；右：出口温度
        for col_i, (cv_label, unit) in enumerate([("出口含水率", "%"), ("烟叶出口温度", "℃")]):
            ax = axes[row_i, col_i]
            ax.plot(sp[:, col_i], color="black", ls="--", lw=1.2, label="设定值 SP")
            ax.plot(res["PID"]["Y"][:, col_i], color=COLORS["warn"], lw=1.2, alpha=0.9, label="工业双回路 PID")
            ax.plot(res["Real_MPC"]["Y"][:, col_i], color=COLORS["pred"], lw=1.4, label="Real_MPC (DMC)")
            m_pid = mets["PID"]["moist_rmse" if col_i == 0 else "temp_rmse"]
            m_mpc = mets["Real_MPC"]["moist_rmse" if col_i == 0 else "temp_rmse"]
            imp = (1.0 - m_mpc / (m_pid + 1e-12)) * 100.0
            ax.set_title(
                f"[{sname}] {cv_label} | PID={m_pid:.4f}, MPC={m_mpc:.4f} (降低 {imp:+.1f}%)",
                fontsize=9.5,
                fontweight="bold",
            )
            ax.set_ylabel(unit)
            ax.grid(True, alpha=0.3)
            if row_i == 0:
                ax.legend(fontsize=8, loc="best")
            if row_i == 2:
                ax.set_xlabel("仿真时间 (秒)")

    fig.suptitle("真实加料回潮工况闭环验证：工业 PID vs 约束 MIMO MPC（含 1B 真实扰动回放与失配测试）", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REAL_FIG / "r5_real_mpc_closed_loop.png", dpi=120)
    plt.close(fig)

    mdf = pd.DataFrame(all_metrics)
    mdf.to_csv(REAL_REP / "r5_real_mpc_metrics.csv", index=False, encoding="utf-8-sig")

    lines = [
        "=" * 82,
        "真实数据脚本 5：加料回潮 MIMO MPC 闭环验证报告（PID vs Real_MPC）",
        "=" * 82,
        mdf.to_string(index=False),
        "",
        "【核心控制收益对比（MPC 相对 PID 的 RMSE 降幅）】",
    ]
    for sname, *_ in scenarios:
        p = mdf[(mdf["scenario"] == sname) & (mdf["controller"] == "PID")].iloc[0]
        m = mdf[(mdf["scenario"] == sname) & (mdf["controller"] == "Real_MPC")].iloc[0]
        im = (1.0 - m["moist_rmse"] / p["moist_rmse"]) * 100.0
        it = (1.0 - m["temp_rmse"] / p["temp_rmse"]) * 100.0
        lines.append(f"  {sname:24s} | 出口水分 RMSE 降低 {im:+5.1f}% | 出口温度 RMSE 降低 {it:+5.1f}%")

    rep_text = "\n".join(lines)
    print("\n" + rep_text)
    save_real_report("r5_real_mpc_report.txt", rep_text)


if __name__ == "__main__":
    main()
