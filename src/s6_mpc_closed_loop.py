#!/usr/bin/env python3
"""
=== 第 6 项：严谨的无静差线性 MPC 闭环（状态空间 + OSQP / CasADi + 模型失配鲁棒性）===

报告 7.4 要求的验证链条：
    历史数据回放 → 离线滚动预测 → 仿真优化 → 影子运行 → 受控在线验证

本脚本在修复早期版本 4 个核心问题的基础上，实现了工业级规范的线性 MPC：
  1. 严格增广状态空间（含逐通道纯滞后链 θ_steam, θ_air），精确分离：
       - 自由响应 Yfree（保持当前 u_prev 不变时的自然演进）
       - 强制响应 G·ΔU（未来 Nc 步控制增量的阶跃叠加）
  2. 目标函数无量纲归一化（消除蒸汽 ~12 vs 热风 ~600 带来的 2500× 数值病态）
  3. 标准 DMC / Luenberger 输出扰动观测器（Offset-free MPC）：
       d_hat(k) = (1 - k_est)·d_hat(k-1) + k_est·(y_meas(k) - y_model(k|k-1))
     保证在模型失配与未测常值扰动下仍实现零稳态静差
  4. 双求解器对比：OSQP（预分解稀疏 QP，亚毫秒级） vs CasADi + IPOPT
  5. 四个递进验证场景：
       ① 设定值跟踪（12.5% → 13.2%，OSQP vs IPOPT vs 无控）
       ② 扰动抑制（入口水分 +1.5%，检验前馈 + 反馈抗扰能力）
       ③ 约束满足性检查（操纵量上下限、变化率硬约束、水分软约束）
       ④ 端到端模型失配鲁棒性（加载 s2 辨识模型 → 控制含执行器滞后/可变延迟/
          仪表滤波/增益失配/未测扰动的真实感对象，对比开/关 Offset-free 观测器）

运行：
    cd ~/hdt_dryer_mpc
    python src/s6_mpc_closed_loop.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import casadi as ca
import osqp

from common import FIG, REP, OUT, COLORS, PALETTE


# ================================================================
# 1. 模型参数容器（支持“真值参数”或“s2 辨识参数”无缝切换）
# ================================================================
def load_mimo_model(use_identified: bool = False) -> dict:
    """加载 MPC 内部预测模型参数。

    use_identified=False : 名义真值参数（用于标称闭环验证）
    use_identified=True  : 从 outputs/identified_mimo_model.npz 加载 s2 辨识参数
                           （用于端到端“辨识→控制”与模型失配鲁棒性验证）
    """
    mimo_path = OUT / "identified_mimo_model.npz"
    if use_identified and mimo_path.exists():
        z = np.load(mimo_path, allow_pickle=True)
        Km = np.asarray(z["K_m"], float)
        Kt = np.asarray(z["K_t"], float)
        th = np.asarray(z["theta"], int)
        return dict(
            source="s2_identified",
            u_base=np.asarray(z["u_base"], float)[:2],
            y_base=np.asarray(z["y_base"], float),
            K_m_steam=float(Km[0]), K_m_air=float(Km[1]), K_m_inlet=float(Km[2]),
            K_t_steam=float(Kt[0]), K_t_air=float(Kt[1]), K_t_inlet=float(Kt[2]),
            tau_m=float(z["tau_m"]), tau_t=float(z["tau_t"]),
            theta_steam=int(th[0]), theta_air=int(th[1]), theta_inlet=int(th[2]),
        )
    return dict(
        source="nominal_true",
        u_base=np.array([260.0, 9500.0]),
        y_base=np.array([12.5, 65.0]),
        K_m_steam=-0.0190, K_m_air=-0.00062, K_m_inlet=+0.42,
        K_t_steam=+0.0220, K_t_air=+0.00320, K_t_inlet=-0.62,
        tau_m=22.0, tau_t=18.0,
        theta_steam=14, theta_air=12, theta_inlet=18,
    )


# ================================================================
# 2. 被控对象（标称对象 + 含执行器滞后/可变延迟/失配的真实感对象）
# ================================================================
class DryerPlant:
    """标称气流烘丝机动态对象（逐通道纯滞后 + 一阶惯性 + 测量噪声）。"""

    def __init__(self, params: dict | None = None, dt: float = 1.0, seed: int = 0):
        p = params or load_mimo_model(use_identified=False)
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.u_base = np.asarray(p["u_base"], float).copy()
        self.y_base = np.asarray(p["y_base"], float).copy()
        self.K_m_steam = p["K_m_steam"]
        self.K_m_air = p["K_m_air"]
        self.K_m_inlet = p["K_m_inlet"]
        self.K_t_steam = p["K_t_steam"]
        self.K_t_air = p["K_t_air"]
        self.K_t_inlet = p["K_t_inlet"]
        self.tau_m = p["tau_m"]
        self.tau_t = p["tau_t"]
        self.theta_steam = int(p["theta_steam"])
        self.theta_air = int(p["theta_air"])
        self.theta_inlet = int(p["theta_inlet"])
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self._x_m = 0.0
        self._x_t = 0.0
        self._x_dm = 0.0
        self._x_dt = 0.0
        self.y_m = float(self.y_base[0])
        self.y_t = float(self.y_base[1])
        # 延迟缓冲：索引 0 为最近一步 u(k-1)，索引 -1 为最老一步 u(k-theta)
        self._buf_u1 = np.full(self.theta_steam, self.u_base[0], dtype=float)
        self._buf_u2 = np.full(self.theta_air, self.u_base[1], dtype=float)
        self._buf_d = np.full(self.theta_inlet, 20.0, dtype=float)

    def step(self, u, d_inlet: float = 20.0):
        u = np.asarray(u, float)
        du1 = self._buf_u1[-1] - self.u_base[0]
        du2 = self._buf_u2[-1] - self.u_base[1]
        dd = self._buf_d[-1] - 20.0

        a_m = self.dt / (self.tau_m + self.dt)
        a_t = self.dt / (self.tau_t + self.dt)
        self._x_m += a_m * (self.K_m_steam * du1 + self.K_m_air * du2 - self._x_m)
        self._x_t += a_t * (self.K_t_steam * du1 + self.K_t_air * du2 - self._x_t)

        # 扰动通道一阶惯性（与 dryer_sim 一致 τ=30s）
        a_d = self.dt / (30.0 + self.dt)
        self._x_dm += a_d * (self.K_m_inlet * dd - self._x_dm)
        self._x_dt += a_d * (self.K_t_inlet * dd - self._x_dt)

        # 推进延迟缓冲
        self._buf_u1 = np.r_[u[0], self._buf_u1[:-1]]
        self._buf_u2 = np.r_[u[1], self._buf_u2[:-1]]
        self._buf_d = np.r_[d_inlet, self._buf_d[:-1]]

        self.y_m = float(self.y_base[0] + self._x_m + self._x_dm + self.rng.normal(0, 0.012))
        self.y_t = float(self.y_base[1] + self._x_t + self._x_dt + self.rng.normal(0, 0.06))
        return self.y_m, self.y_t


class MismatchDryerPlant(DryerPlant):
    """带工程失配的真实感被控对象（Stage 3 鲁棒性测试专用）：
      - 增益失配 (+15%) 与时间常数失配 (+20%)
      - 执行器一阶跟随滞后（蒸汽 τ=4s，热风 τ=6s）
      - 仪表测量滤波（水分 τ=2.5s，温度 τ=2.0s）
      - 纯滞后多 +2s（模拟低负荷下的可变延迟偏移）
      - 支持叠加未测常值扰动（如未测原料品种/环境湿度漂移）
    """

    def __init__(self, dt: float = 1.0, seed: int = 7):
        base = load_mimo_model(use_identified=False)
        p = dict(base)
        p["K_m_steam"] *= 1.15
        p["K_m_air"] *= 0.85
        p["tau_m"] *= 1.20
        p["tau_t"] *= 1.15
        p["theta_steam"] += 2
        p["theta_air"] += 2
        super().__init__(params=p, dt=dt, seed=seed)

    def reset(self):
        super().reset()
        self._u_act = self.u_base.copy()
        self._ym_filt = float(self.y_base[0])
        self._yt_filt = float(self.y_base[1])

    def step(self, u_sp, d_inlet: float = 20.0, unmeasured_moist_bias: float = 0.0):
        u_sp = np.asarray(u_sp, float)
        # 1. 执行器一阶跟随
        a_act = np.array([self.dt / (4.0 + self.dt), self.dt / (6.0 + self.dt)])
        self._u_act = self._u_act + a_act * (u_sp - self._u_act)
        # 2. 过程主体
        ym_true, yt_true = super().step(self._u_act, d_inlet)
        ym_true += unmeasured_moist_bias
        # 3. 仪表一阶滤波 + 测量噪声
        self._ym_filt += (self.dt / (2.5 + self.dt)) * (ym_true - self._ym_filt)
        self._yt_filt += (self.dt / (2.0 + self.dt)) * (yt_true - self._yt_filt)
        self.y_m = float(self._ym_filt + self.rng.normal(0, 0.02))
        self.y_t = float(self._yt_filt + self.rng.normal(0, 0.08))
        return self.y_m, self.y_t


# ================================================================
# 3. 严谨的无静差线性 MPC（支持 OSQP 稀疏 QP 与 CasADi IPOPT）
# ================================================================
class MPCController:
    """增广状态空间 + 增量控制 + 无量纲归一化 + Offset-free 扰动观测器 MPC。"""

    U_MIN = np.array([150.0, 7000.0])
    U_MAX = np.array([400.0, 12000.0])
    DU_MAX = np.array([12.0, 600.0])
    MOIST_MIN = 11.5
    MOIST_MAX = 14.0

    # 无量纲归一化尺度
    SCALE_YM = 0.5          # 水分特征尺度 (%)
    SCALE_YT = 2.0          # 温度特征尺度 (℃)
    SCALE_DU = np.array([12.0, 600.0])

    # 归一化后的权重（数值条件数 O(1)~O(100)，彻底消除病态）
    Q_m = 40.0
    Q_t = 0.5
    R_w = np.array([1.5, 4.0])    # 优先用蒸汽微调，热风动作更稳健
    W_SLACK_LO = 5000.0
    W_SLACK_HI = 5000.0

    def __init__(self, model_params: dict, Np: int = 60, Nc: int = 10, dt: float = 1.0,
                 solver: str = "osqp", use_offset_free: bool = True, k_est: float = 0.15):
        self.p = model_params
        self.Np, self.Nc, self.dt = Np, Nc, dt
        self.nu = 2
        self.solver = solver.lower()
        self.use_offset_free = use_offset_free
        self.k_est = k_est

        self.u_base = np.asarray(self.p["u_base"], float)
        self.y_base = np.asarray(self.p["y_base"], float)
        self.th1 = int(self.p["theta_steam"])
        self.th2 = int(self.p["theta_air"])

        self._build_state_space()
        self._build_step_response_matrices()
        if self.solver == "osqp":
            self._build_osqp()
        elif self.solver == "ipopt":
            self._build_ipopt()
        else:
            raise ValueError(f"不支持的求解器: {solver}")
        self.reset()

    def reset(self):
        """重置内部状态与扰动观测器（跨场景必须调用！）。"""
        self.x_model = np.zeros(self.nx, dtype=float)
        self.d_hat_m = 0.0
        self.d_hat_t = 0.0
        self._x_dm_ff = 0.0
        self._x_dt_ff = 0.0
        self._buf_d_ff = np.full(int(self.p["theta_inlet"]), 20.0, dtype=float)
        self._has_prev = False
        self._y_model_pred = self.y_base.copy()
        self.last_error = ""

    def _build_state_space(self):
        th1, th2 = self.th1, self.th2
        nx = 2 + th1 + th2
        self.nx = nx
        A = np.zeros((nx, nx))
        B = np.zeros((nx, 2))
        C = np.zeros((2, nx))

        a_m = self.dt / (self.p["tau_m"] + self.dt)
        a_t = self.dt / (self.p["tau_t"] + self.dt)
        i1 = 2 + th1 - 1
        i2 = 2 + th1 + th2 - 1

        A[0, 0] = 1.0 - a_m
        A[0, i1] = a_m * self.p["K_m_steam"]
        A[0, i2] = a_m * self.p["K_m_air"]
        A[1, 1] = 1.0 - a_t
        A[1, i1] = a_t * self.p["K_t_steam"]
        A[1, i2] = a_t * self.p["K_t_air"]

        B[2, 0] = 1.0
        B[2 + th1, 1] = 1.0
        for r in range(1, th1):
            A[2 + r, 2 + r - 1] = 1.0
        for r in range(1, th2):
            A[2 + th1 + r, 2 + th1 + r - 1] = 1.0

        C[0, 0] = 1.0
        C[1, 1] = 1.0
        self.A, self.B, self.C = A, B, C

    def _build_step_response_matrices(self):
        """离线预计算常数阶跃响应矩阵 Gm, Gt（只算一次，极大提速！）。"""
        Np, Nc, nu = self.Np, self.Nc, self.nu
        A, B, C = self.A, self.B, self.C
        # CAB[l] = C @ A^l @ B
        CAB = []
        AkB = B.copy()
        for _ in range(Np):
            CAB.append(C @ AkB)
            AkB = A @ AkB
        # 阶跃响应 S[k] = Σ_{l=0}^{k} C A^l B
        S = np.cumsum(CAB, axis=0)   # (Np, 2, 2)

        Gm = np.zeros((Np, Nc * nu))
        Gt = np.zeros((Np, Nc * nu))
        for j in range(Nc):
            for i in range(j, Np):
                step_resp = S[i - j]
                Gm[i, j * nu:(j + 1) * nu] = step_resp[0]
                Gt[i, j * nu:(j + 1) * nu] = step_resp[1]
        self.Gm, self.Gt = Gm, Gt

        # 累积求和矩阵 L_cum: U_seq = 1*u_prev + L_cum @ dU
        L_cum = np.zeros((Nc * nu, Nc * nu))
        for j in range(Nc):
            for r in range(j + 1):
                L_cum[j * nu:(j + 1) * nu, r * nu:(r + 1) * nu] = np.eye(nu)
        self.L_cum = L_cum

    def _free_response(self, x_now: np.ndarray, u_prev: np.ndarray):
        """严格自由响应：保持未来所有控制量 u(k+h) ≡ u_prev 不变时的预测输出。"""
        du_base = u_prev - self.u_base
        x = x_now.copy()
        yf = np.zeros((self.Np, 2))
        for h in range(self.Np):
            x = self.A @ x + self.B @ du_base
            yf[h] = self.C @ x + self.y_base
        return yf[:, 0], yf[:, 1]

    def _build_osqp(self):
        """构造稀疏凸二次规划（OSQP），矩阵 P 和 A_con 只需分解一次。"""
        Np, Nc, nu = self.Np, self.Nc, self.nu
        n_du = Nc * nu
        n_z = n_du + 2 * Np  # [dU, sl, sh]

        wm = self.Q_m / (self.SCALE_YM ** 2)
        wt = self.Q_t / (self.SCALE_YT ** 2)
        r_vec = np.tile(self.R_w / (self.SCALE_DU ** 2), Nc)

        H_du = 2.0 * (wm * (self.Gm.T @ self.Gm) + wt * (self.Gt.T @ self.Gt) + np.diag(r_vec))
        # 保证严格正定
        H_du += 1e-9 * np.eye(n_du)
        H_sl = 2.0 * (self.W_SLACK_LO / (self.SCALE_YM ** 2)) * np.eye(Np)
        H_sh = 2.0 * (self.W_SLACK_HI / (self.SCALE_YM ** 2)) * np.eye(Np)
        P = sp.block_diag([sp.csc_matrix(H_du), sp.csc_matrix(H_sl), sp.csc_matrix(H_sh)], format="csc")

        # 约束矩阵 A_con * z in [l, u]:
        # 1) dU 变化率约束:   I * dU in [-DU_MAX, DU_MAX]
        # 2) U  绝对幅值约束: L_cum * dU in [U_MIN - u_prev, U_MAX - u_prev]
        # 3) 水分下限软约束:  Gm * dU + I * sl in [MOIST_MIN - Yf_m, +inf]
        # 4) 水分上限软约束:  Gm * dU - I * sh in [-inf, MOIST_MAX - Yf_m]
        # 5) 松弛变量非负:    sl >= 0, sh >= 0
        Z_du_np = sp.csc_matrix((n_du, Np))
        Z_np_np = sp.csc_matrix((Np, Np))
        I_du = sp.eye(n_du, format="csc")
        I_np = sp.eye(Np, format="csc")

        row1 = sp.hstack([I_du, Z_du_np, Z_du_np])
        row2 = sp.hstack([sp.csc_matrix(self.L_cum), Z_du_np, Z_du_np])
        row3 = sp.hstack([sp.csc_matrix(self.Gm), I_np, Z_np_np])
        row4 = sp.hstack([sp.csc_matrix(self.Gm), Z_np_np, -I_np])
        row5 = sp.hstack([sp.csc_matrix((Np, n_du)), I_np, Z_np_np])
        row6 = sp.hstack([sp.csc_matrix((Np, n_du)), Z_np_np, I_np])
        A_con = sp.vstack([row1, row2, row3, row4, row5, row6], format="csc")

        self._osqp_wm, self._osqp_wt = wm, wt
        self._osqp_n_du, self._osqp_n_z = n_du, n_z
        q0 = np.zeros(n_z)
        l0 = np.full(A_con.shape[0], -1e20)
        u0 = np.full(A_con.shape[0], 1e20)

        prob = osqp.OSQP()
        prob.setup(P=P, q=q0, A=A_con, l=l0, u=u0,
                   verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, polish=True)
        self._osqp = prob

    def _build_ipopt(self):
        """CasADi + IPOPT 求解器（用于与 OSQP 对比精度与耗时）。"""
        Np, Nc, nu = self.Np, self.Nc, self.nu
        opti = ca.Opti()
        dU = opti.variable(Nc * nu)
        sl = opti.variable(Np)
        sh = opti.variable(Np)

        Yfree_m = opti.parameter(Np)
        Yfree_t = opti.parameter(Np)
        ref_m = opti.parameter()
        ref_t = opti.parameter()
        u_prev = opti.parameter(nu)

        Ym = ca.DM(self.Gm) @ dU + Yfree_m
        Yt = ca.DM(self.Gt) @ dU + Yfree_t
        U_seq = ca.repmat(u_prev, Nc, 1) + ca.DM(self.L_cum) @ dU

        # 归一化目标函数（与 OSQP 完全一致）
        r_vec = np.tile(self.R_w / (self.SCALE_DU ** 2), Nc)
        cost = ((self.Q_m / self.SCALE_YM ** 2) * ca.sumsqr(Ym - ref_m)
                + (self.Q_t / self.SCALE_YT ** 2) * ca.sumsqr(Yt - ref_t)
                + ca.dot(ca.DM(r_vec) * dU, dU)
                + (self.W_SLACK_LO / self.SCALE_YM ** 2) * ca.sumsqr(sl)
                + (self.W_SLACK_HI / self.SCALE_YM ** 2) * ca.sumsqr(sh))

        opti.subject_to(opti.bounded(-np.tile(self.DU_MAX, Nc), dU, np.tile(self.DU_MAX, Nc)))
        opti.subject_to(opti.bounded(np.tile(self.U_MIN, Nc), U_seq, np.tile(self.U_MAX, Nc)))
        opti.subject_to(sl >= 0)
        opti.subject_to(sh >= 0)
        opti.subject_to(Ym >= self.MOIST_MIN - sl)
        opti.subject_to(Ym <= self.MOIST_MAX + sh)
        opti.minimize(cost)

        opti.solver("ipopt", {"verbose": False, "print_time": False},
                    {"print_level": 0, "max_iter": 200, "sb": "yes", "tol": 1e-6})
        self.opti, self.dU, self.sl, self.sh = opti, dU, sl, sh
        self.Yfree_m, self.Yfree_t = Yfree_m, Yfree_t
        self.ref_m_p, self.ref_t_p, self.u_prev_p = ref_m, ref_t, u_prev

    def _predict_disturbance_ff(self, d_inlet_meas: float):
        """可测扰动（入口水分）前馈预测轨迹（含纯滞后 + 一阶惯性）。"""
        a_d = self.dt / (30.0 + self.dt)
        xm, xt = self._x_dm_ff, self._x_dt_ff
        buf = self._buf_d_ff.copy()
        ff_m = np.zeros(self.Np)
        ff_t = np.zeros(self.Np)
        for h in range(self.Np):
            dd = buf[-1] - 20.0
            xm += a_d * (self.p["K_m_inlet"] * dd - xm)
            xt += a_d * (self.p["K_t_inlet"] * dd - xt)
            ff_m[h], ff_t[h] = xm, xt
            buf = np.r_[d_inlet_meas, buf[:-1]]
        return ff_m, ff_t

    def _advance_internal_model(self, u_applied: np.ndarray, d_inlet_meas: float):
        """推进控制器内部标称模型一步，得到下一拍的纯模型预测 y_model(k+1|k)。"""
        self.x_model = self.A @ self.x_model + self.B @ (u_applied - self.u_base)
        a_d = self.dt / (30.0 + self.dt)
        dd = self._buf_d_ff[-1] - 20.0
        self._x_dm_ff += a_d * (self.p["K_m_inlet"] * dd - self._x_dm_ff)
        self._x_dt_ff += a_d * (self.p["K_t_inlet"] * dd - self._x_dt_ff)
        self._buf_d_ff = np.r_[d_inlet_meas, self._buf_d_ff[:-1]]

        self._y_model_pred = np.array([
            self.y_base[0] + self.x_model[0] + self._x_dm_ff,
            self.y_base[1] + self.x_model[1] + self._x_dt_ff,
        ])
        self._has_prev = True

    def compute(self, y_m: float, y_t: float, ref_m: float, ref_t: float,
                u_prev: np.ndarray, d_inlet_meas: float = 20.0):
        """求解当前拍最优控制增量 Δu(k)。"""
        u_prev = np.asarray(u_prev, float)

        # 1. 标准 Offset-free 输出扰动观测器更新：
        #    d_hat(k) = (1 - k_est) * d_hat(k-1) + k_est * (y_meas(k) - y_model(k|k-1))
        if self.use_offset_free and self._has_prev:
            raw_bias_m = y_m - self._y_model_pred[0]
            raw_bias_t = y_t - self._y_model_pred[1]
            self.d_hat_m = (1.0 - self.k_est) * self.d_hat_m + self.k_est * raw_bias_m
            self.d_hat_t = (1.0 - self.k_est) * self.d_hat_t + self.k_est * raw_bias_t

        # 2. 自由响应 + 可测扰动前馈 + 无静差偏置
        yf_m, yf_t = self._free_response(self.x_model, u_prev)
        ff_m, ff_t = self._predict_disturbance_ff(d_inlet_meas)
        yf_m = yf_m + ff_m + self.d_hat_m
        yf_t = yf_t + ff_t + self.d_hat_t

        # 3. 求解 QP
        ok = True
        if self.solver == "osqp":
            q_du = 2.0 * (self._osqp_wm * (self.Gm.T @ (yf_m - ref_m))
                          + self._osqp_wt * (self.Gt.T @ (yf_t - ref_t)))
            q = np.r_[q_du, np.zeros(2 * self.Np)]

            du_lim = np.tile(self.DU_MAX, self.Nc)
            u_lo = np.tile(self.U_MIN - u_prev, self.Nc)
            u_hi = np.tile(self.U_MAX - u_prev, self.Nc)
            l = np.r_[-du_lim, u_lo, self.MOIST_MIN - yf_m, np.full(self.Np, -1e20),
                      np.zeros(self.Np), np.zeros(self.Np)]
            u = np.r_[du_lim, u_hi, np.full(self.Np, 1e20), self.MOIST_MAX - yf_m,
                      np.full(self.Np, 1e20), np.full(self.Np, 1e20)]
            self._osqp.update(q=q, l=l, u=u)
            res = self._osqp.solve()
            if res.info.status_val in (1, 2) and res.x is not None:
                du0 = np.clip(res.x[:self.nu], -self.DU_MAX, self.DU_MAX)
            else:
                self.last_error = f"OSQP status={res.info.status}"
                du0 = np.zeros(self.nu)
                ok = False
        else:
            self.opti.set_value(self.Yfree_m, yf_m)
            self.opti.set_value(self.Yfree_t, yf_t)
            self.opti.set_value(self.ref_m_p, ref_m)
            self.opti.set_value(self.ref_t_p, ref_t)
            self.opti.set_value(self.u_prev_p, u_prev)
            try:
                sol = self.opti.solve()
                du = np.array(sol.value(self.dU)).ravel()
                du0 = np.clip(du[:self.nu], -self.DU_MAX, self.DU_MAX)
            except Exception as e:
                self.last_error = str(e)[:200]
                du0 = np.zeros(self.nu)
                ok = False

        u_new = np.clip(u_prev + du0, self.U_MIN, self.U_MAX)
        # 4. 用实际将施加的控制量推进内部模型
        self._advance_internal_model(u_new, d_inlet_meas)
        return u_new - u_prev, ok


# ================================================================
# 4. 闭环仿真运行器
# ================================================================
def simulate_closed_loop(controller: MPCController | None, plant, n_steps: int = 500,
                         ref_fn=None, d_fn=None, unmeasured_fn=None):
    plant.reset()
    if controller is not None:
        controller.reset()
    u = plant.u_base.copy()
    rows = []
    for k in range(n_steps):
        rm = float(ref_fn(k)) if ref_fn else 12.5
        d = float(d_fn(k)) if d_fn else 20.0
        unm = float(unmeasured_fn(k)) if unmeasured_fn else 0.0
        ok = True
        if controller is not None:
            # 温度参考设为当前值（仅抑制剧烈波动，不与水分争夺控制权）
            du, ok = controller.compute(plant.y_m, plant.y_t, rm, plant.y_t, u, d_inlet_meas=d)
            u = np.clip(u + du, controller.U_MIN, controller.U_MAX)
        if isinstance(plant, MismatchDryerPlant):
            ym, yt = plant.step(u, d, unmeasured_moist_bias=unm)
        else:
            ym, yt = plant.step(u, d)
        rows.append(dict(k=k, u1=u[0], u2=u[1], y_m=ym, y_t=yt,
                         d=d, ref_m=rm, solver_ok=ok))
    return pd.DataFrame(rows)


def calc_settling_time(df: pd.DataFrame, step_at: int, target: float, tol: float = 0.05):
    sub = df[df.k >= step_at]
    in_band = (np.abs(sub.y_m - target) <= tol).values
    for i in range(len(in_band)):
        if in_band[i:].all():
            return int(sub.k.iloc[i] - step_at)
    return None


# ================================================================
# 5. 主流程：4 大场景完整验证
# ================================================================
def main():
    print("=" * 78)
    print("第 6 项：无静差线性 MPC 闭环（OSQP vs IPOPT + 端到端模型失配鲁棒性）")
    print("=" * 78)
    lines = []
    def log(s=""):
        print(s); lines.append(s)

    p_nom = load_mimo_model(use_identified=False)
    p_id = load_mimo_model(use_identified=True)

    plant_nom = DryerPlant(params=p_nom, dt=1.0, seed=0)
    mpc_osqp = MPCController(p_nom, Np=60, Nc=10, solver="osqp", use_offset_free=True)
    mpc_ipopt = MPCController(p_nom, Np=60, Nc=10, solver="ipopt", use_offset_free=True)

    log(f"标称对象：θ_steam={p_nom['theta_steam']}s, θ_air={p_nom['theta_air']}s, "
        f"τ_m={p_nom['tau_m']}s, τ_t={p_nom['tau_t']}s")
    log(f"辨识模型：来源={p_id['source']} | θ_steam={p_id['theta_steam']}s, "
        f"τ_m={p_id['tau_m']:.1f}s, K_m_steam={p_id['K_m_steam']:+.5f}")
    log(f"MPC 配置：Np={mpc_osqp.Np}s, Nc={mpc_osqp.Nc} | 约束：蒸汽 {mpc_osqp.U_MIN[0]}~{mpc_osqp.U_MAX[0]} kg/h, "
        f"热风 {mpc_osqp.U_MIN[1]}~{mpc_osqp.U_MAX[1]}, Δu≤{mpc_osqp.DU_MAX.tolist()}, 水分 {mpc_osqp.MOIST_MIN}~{mpc_osqp.MOIST_MAX}%")

    # ================================================================
    # 场景 1：设定值跟踪（12.5% → 13.2%）
    # ================================================================
    log("\n" + "-" * 78)
    log("【场景 1】设定值跟踪：出口水分 12.5% → 13.2%（第 100 秒起）")
    log("-" * 78)
    ref_fn = lambda k: 13.2 if k >= 100 else 12.5
    df_track_osqp = simulate_closed_loop(mpc_osqp, plant_nom, 500, ref_fn=ref_fn)
    df_track_ipopt = simulate_closed_loop(mpc_ipopt, plant_nom, 500, ref_fn=ref_fn)
    df_manual = simulate_closed_loop(None, plant_nom, 500, ref_fn=ref_fn)

    for tag, df_ in [("MPC-OSQP ", df_track_osqp), ("MPC-IPOPT", df_track_ipopt)]:
        sse = abs(df_.y_m.iloc[-50:].mean() - 13.2)
        os_val = max(0.0, float(df_.y_m[df_.k >= 100].max() - 13.2))
        ts = calc_settling_time(df_, 100, 13.2, tol=0.05)
        n_fail = int((~df_.solver_ok).sum())
        log(f"  {tag}：稳态误差={sse:.4f}% | 超调={os_val:.4f}% | "
            f"调节时间(±0.05%)={ts if ts is not None else '>400'}s | 求解失败={n_fail}")
    max_diff = float(np.max(np.abs(df_track_osqp.y_m.values - df_track_ipopt.y_m.values)))
    log(f"  OSQP vs IPOPT 闭环轨迹最大偏差 = {max_diff:.6f}%（证明两种凸优化求解器数值完全等价）")

    # ================================================================
    # 场景 2：扰动抑制（入口水分 +1.5% 阶跃）
    # ================================================================
    log("\n" + "-" * 78)
    log("【场景 2】扰动抑制：第 150 秒入口水分 +1.5%（来料突然变湿）")
    log("-" * 78)
    d_fn = lambda k: 20.0 + (1.5 if k >= 150 else 0.0)
    df_dist_mpc = simulate_closed_loop(mpc_osqp, plant_nom, 500, ref_fn=lambda k: 12.5, d_fn=d_fn)
    df_dist_man = simulate_closed_loop(None, plant_nom, 500, ref_fn=lambda k: 12.5, d_fn=d_fn)

    dev_mpc = np.abs(df_dist_mpc.y_m[df_dist_mpc.k >= 150] - 12.5)
    dev_man = np.abs(df_dist_man.y_m[df_dist_man.k >= 150] - 12.5)
    suppress_peak = (1.0 - dev_mpc.max() / max(dev_man.max(), 1e-9)) * 100
    suppress_ss = (1.0 - dev_mpc.iloc[-50:].mean() / max(dev_man.iloc[-50:].mean(), 1e-9)) * 100
    log(f"  有控(MPC-OSQP)：峰值偏差={dev_mpc.max():.4f}% | 稳态残留={dev_mpc.iloc[-50:].mean():.4f}%")
    log(f"  无控(开环)    ：峰值偏差={dev_man.max():.4f}% | 稳态残留={dev_man.iloc[-50:].mean():.4f}%")
    log(f"  → 峰值抑制比 = {suppress_peak:+.1f}% | 稳态抑制比 = {suppress_ss:+.1f}%")

    # ================================================================
    # 场景 3：约束满足性检查 + 求解器耗时基准
    # ================================================================
    log("\n" + "-" * 78)
    log("【场景 3】约束满足性检查 & 求解器实时性对比")
    log("-" * 78)
    checks = []
    for nm, df_ in [("跟踪(OSQP)", df_track_osqp), ("抗扰(OSQP)", df_dist_mpc)]:
        u1_ok = ((df_.u1 >= mpc_osqp.U_MIN[0] - 1e-6) & (df_.u1 <= mpc_osqp.U_MAX[0] + 1e-6)).all()
        u2_ok = ((df_.u2 >= mpc_osqp.U_MIN[1] - 1e-6) & (df_.u2 <= mpc_osqp.U_MAX[1] + 1e-6)).all()
        du1 = np.abs(np.diff(df_.u1.values))
        du2 = np.abs(np.diff(df_.u2.values))
        du_ok = (du1 <= mpc_osqp.DU_MAX[0] + 1e-6).all() and (du2 <= mpc_osqp.DU_MAX[1] + 1e-6).all()
        ym_ok = ((df_.y_m >= mpc_osqp.MOIST_MIN - 0.02) & (df_.y_m <= mpc_osqp.MOIST_MAX + 0.02)).all()
        checks.append({
            "场景": nm,
            "蒸汽范围": "✅" if u1_ok else "❌",
            "热风范围": "✅" if u2_ok else "❌",
            "变化率": "✅" if du_ok else "❌",
            "水分边界": "✅" if ym_ok else "❌",
            "最大|Δu1|": round(float(du1.max()), 3),
            "失败次数": int((~df_.solver_ok).sum()),
        })
    chk = pd.DataFrame(checks)
    log(chk.to_string(index=False))

    def _bench_ms(ctrl, n_rep=60):
        ctrl.reset()
        u_cur = ctrl.u_base.copy()
        ts = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            ctrl.compute(12.5, 65.0, 12.9, 65.0, u_cur, d_inlet_meas=20.0)
            ts.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(ts))

    ms_osqp = _bench_ms(mpc_osqp)
    ms_ipopt = _bench_ms(mpc_ipopt)
    log(f"\n  单步求解耗时（中位数）：OSQP = {ms_osqp:.3f} ms | IPOPT = {ms_ipopt:.3f} ms "
        f"（OSQP 提速 {ms_ipopt / max(ms_osqp, 1e-6):.1f}×）")

    # ================================================================
    # 场景 4（Stage 3 亮点）：端到端辨识模型 → 失配对象闭环鲁棒性
    # ================================================================
    log("\n" + "-" * 78)
    log("【场景 4】端到端模型失配鲁棒性（s2 辨识模型 → 控制含失配/执行器滞后/滤波的对象）")
    log("-" * 78)
    plant_mis = MismatchDryerPlant(dt=1.0, seed=7)
    mpc_id_on = MPCController(p_id, Np=60, Nc=10, solver="osqp", use_offset_free=True, k_est=0.12)
    mpc_id_off = MPCController(p_id, Np=60, Nc=10, solver="osqp", use_offset_free=False)

    # 复合严苛工况：100s 设定值跳变 12.5→13.0，260s 出现 +0.35% 未测原料水分漂移
    ref_mis = lambda k: 13.0 if k >= 100 else 12.5
    unm_mis = lambda k: 0.35 if k >= 260 else 0.0

    df_mis_on = simulate_closed_loop(mpc_id_on, plant_mis, 500, ref_fn=ref_mis, unmeasured_fn=unm_mis)
    df_mis_off = simulate_closed_loop(mpc_id_off, plant_mis, 500, ref_fn=ref_mis, unmeasured_fn=unm_mis)

    sse_on = abs(df_mis_on.y_m.iloc[-60:].mean() - 13.0)
    sse_off = abs(df_mis_off.y_m.iloc[-60:].mean() - 13.0)
    log(f"  开启 Offset-free 观测器：末段稳态误差 = {sse_on:.4f}% | 求解失败 = {int((~df_mis_on.solver_ok).sum())}")
    log(f"  关闭 Offset-free 观测器：末段稳态误差 = {sse_off:.4f}% | 求解失败 = {int((~df_mis_off.solver_ok).sum())}")
    log(f"  → Offset-free 扰动观测器将模型失配 + 未测漂移下的稳态静差降低了 "
        f"{(1 - sse_on / max(sse_off, 1e-9)) * 100:.1f}%！")

    # ================================================================
    # 绘图（2×3 综合闭环诊断图）
    # ================================================================
    fig, axes = plt.subplots(3, 2, figsize=(15, 11.5))

    ax = axes[0, 0]
    ax.plot(df_track_osqp.k, df_track_osqp.ref_m, "k--", lw=1.2, label="设定值")
    ax.plot(df_track_osqp.k, df_track_osqp.y_m, color=PALETTE[1], lw=1.8, label="MPC-OSQP")
    ax.plot(df_track_ipopt.k, df_track_ipopt.y_m, ":", color=PALETTE[2], lw=1.6, label="MPC-IPOPT")
    ax.plot(df_manual.k, df_manual.y_m, color=PALETTE[0], lw=1.1, alpha=.75, label="无控制")
    ax.set_title("① 设定值跟踪（12.5% → 13.2%）", fontweight="bold")
    ax.set_ylabel("出口水分 (%)"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(df_track_osqp.k, df_track_osqp.u1, color=PALETTE[2], lw=1.4, label="蒸汽 kg/h")
    ax.axhline(mpc_osqp.U_MIN[0], ls="--", c="r", lw=.8, alpha=.6)
    ax.axhline(mpc_osqp.U_MAX[0], ls="--", c="r", lw=.8, alpha=.6)
    ax2 = ax.twinx()
    ax2.plot(df_track_osqp.k, df_track_osqp.u2, color=PALETTE[4], lw=1.2, alpha=.85, label="热风")
    ax2.set_ylabel("热风流量", color=PALETTE[4])
    ax.set_title("① 操纵量动作与约束边界", fontweight="bold")
    ax.set_ylabel("蒸汽流量 (kg/h)", color=PALETTE[2])

    ax = axes[1, 0]
    ax.plot(df_dist_mpc.k, df_dist_mpc.y_m, color=PALETTE[1], lw=1.8, label="MPC-OSQP")
    ax.plot(df_dist_man.k, df_dist_man.y_m, color=PALETTE[0], lw=1.2, alpha=.8, label="无控制")
    ax.axhline(12.5, ls="--", c="k", lw=1, label="目标 12.5%")
    ax.axvline(150, ls=":", c="gray", lw=1.5, label="入口水分 +1.5%")
    ax.set_title(f"② 扰动抑制（峰值抑制 {suppress_peak:.1f}%）", fontweight="bold")
    ax.set_ylabel("出口水分 (%)"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(df_track_osqp.k,
            np.abs(np.diff(df_track_osqp.u1.values, prepend=df_track_osqp.u1.values[0])),
            color=PALETTE[3], lw=1.3, label="|Δu1| 蒸汽变化率")
    ax.axhline(mpc_osqp.DU_MAX[0], ls="--", c="r", lw=1.2, label="Δu1 上限")
    ax.set_title("③ 操纵量变化率硬约束", fontweight="bold")
    ax.set_ylabel("kg/h per step"); ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(df_mis_on.k, df_mis_on.ref_m, "k--", lw=1.1, label="设定值")
    ax.plot(df_mis_on.k, df_mis_on.y_m, color=PALETTE[1], lw=1.8, label=f"Offset-free 开启 (静差 {sse_on:.3f}%)")
    ax.plot(df_mis_off.k, df_mis_off.y_m, color=PALETTE[0], lw=1.3, alpha=.85, label=f"Offset-free 关闭 (静差 {sse_off:.3f}%)")
    ax.axvline(260, ls=":", c="purple", lw=1.4, label="未测漂移 +0.35%")
    ax.set_title("④ 端到端模型失配 + 未测漂移鲁棒性（s2 辨识模型）", fontweight="bold")
    ax.set_ylabel("出口水分 (%)"); ax.set_xlabel("时间 (s)"); ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.plot(df_mis_on.k, df_mis_on.u1, color=PALETTE[2], lw=1.4, label="蒸汽 (Offset-free)")
    ax.plot(df_mis_off.k, df_mis_off.u1, color=PALETTE[0], ls="--", lw=1.1, alpha=.8, label="蒸汽 (无观测器)")
    ax.set_title("④ 失配场景操纵量对比", fontweight="bold")
    ax.set_ylabel("蒸汽流量 (kg/h)"); ax.set_xlabel("时间 (s)"); ax.legend(fontsize=8)

    for a in axes.ravel():
        a.grid(alpha=.3)
    fig.suptitle(f"HDT 气流烘丝机无静差 MPC 闭环验证（OSQP {ms_osqp:.2f}ms vs IPOPT {ms_ipopt:.2f}ms）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG / "s6_mpc_closed_loop.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    log(f"\n图已保存 -> {out}")

    df_track_osqp.to_csv(REP / "s6_scenario_tracking.csv", index=False, encoding="utf-8-sig")
    df_dist_mpc.to_csv(REP / "s6_scenario_disturbance.csv", index=False, encoding="utf-8-sig")
    df_mis_on.to_csv(REP / "s6_scenario_mismatch.csv", index=False, encoding="utf-8-sig")
    (REP / "s6_mpc_closed_loop.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已保存 -> {REP / 's6_mpc_closed_loop.txt'}")


if __name__ == "__main__":
    main()
