# HDT 气流烘丝机：核心理论知识与科研基本功手册

> 本手册把工程代码（`src/s1` ~ `src/s6`）背后的**控制理论、系统辨识、时序预测与科研方法论**串成一条完整脉络。阅读时建议对照对应脚本的实跑结果一起理解。

---

## 一、系统辨识（System Identification）

对应脚本：`src/s2_delay_identification.py`

### 1.1 一阶加纯滞后模型（FOPDT）

工业热工与干燥过程最经典的可解释动态模型是 FOPDT（First-Order Plus Dead Time）：

$$G(s) = \frac{Y(s)}{U(s)} = \frac{K e^{-\theta s}}{\tau s + 1}$$

- **静态增益 $K$**：操纵量变化 1 个单位，稳态时输出变化多少（例如蒸汽 $+100\text{ kg/h} \to$ 水分 $-1.9\%$，故 $K = -0.019$）。**符号决定控制方向**，符号错则闭环必然发散。
- **纯滞后 $\theta$**：输入改变后，输出**完全没有反应**的死区时间（物料在管道/干燥筒内的输送时间 + 仪表延迟）。
- **时间常数 $\tau$**：越过死区后，输出上升到最终变化量 $63.2\%$ 所需的时间，刻画过程惯性大小。

离散化（采样周期 $\Delta t$，记 $\alpha = \frac{\Delta t}{\tau + \Delta t}$）：

$$x(k) = (1 - \alpha) x(k-1) + \alpha K \tilde{u}(k - \theta)$$

### 1.2 为什么单输入（SISO）阶跃拟合会出现 4 倍误差？

在早期 `s2_delay_identification.py` 中，虽然仿真器真实参数是 $\theta=14\text{s}, \tau=22\text{s}, K=-0.019$，但单通道阶跃拟合却得到 $\tau=90.4\text{s}, K=-0.052$。核心原因有两个：
1. **多变量混叠（Confounding / Collinearity）**：蒸汽阶跃的同时，热风、入口水分、进料量都在做正弦波动。单输入模型只能把其他 3 个通道的能量强行“吸收”进蒸汽通道的 $K$ 和 $\tau$。
2. **激励不足（Poor Excitation）**：阶跃窗口长度不足 $3\sim 5\tau$，无法区分“增益大+惯性大”与“增益小+惯性小”。

### 1.3 工业标准解法：PRBS 正交激励 + MIMO 联合辨识

- **PRBS（伪随机二值/多级序列）**：各输入通道独立随机切换，通道间互相关近似为 0，频域覆盖宽，满足**持续激励条件（Persistent Excitation）**。
- **变量投影法（Variable Projection）**：对非线性参数 $(\theta_1,\dots,\theta_m, \tau)$ 做搜索，对线性参数 $K$ 用最小二乘解析求解，彻底避开梯度下降的局部极小值。在 `s2_delay_identification.py` 的第 ⑤ 步中，误差从 $>150\%$ 压降到 $1\sim 5\%$。

---

## 二、多步时序预测与“面向控制的建模”

对应脚本：`src/s2_baseline.py`, `src/s3_window_builder.py`, `src/s4_multistep_eval.py`, `src/s5_mlflow_tracking.py`

### 2.1 三种多步预测策略对比

MPC 一次需要未来 $N_p$ 步的预测轨迹 $\hat{y}(k+1 \dots k+N_p)$，常见有三种策略：

| 策略 | 做法 | 优点 | 缺点 | 本项目代表 |
| :--- | :--- | :--- | :--- | :--- |
| **Recursive（递归单步）** | 训练 1 个单步模型，把预测值喂回输入滚动 $N_p$ 次 | 参数少、训练快 | 误差逐步累积，长时域易发散 | ARX / 状态空间递推 |
| **Direct（直接多步）** | 对每个步长 $h=1\dots N_p$ 单独训一个模型 | 无误差累积 | 需训 $n_{\text{out}} \times N_p$ 个模型，步间缺乏平滑一致性 | `s4`/`s5` 的 Ridge-Direct、LightGBM-Direct |
| **MIMO Seq2Seq（联合多步）** | 单一网络/解析模型一次输出整条 $N_p$ 步轨迹 | 步间连贯、共享表征 | 深度模型需要较大数据量 | `FOPDT-MIMO`、Darts `TCN` |

### 2.2 为什么预测 RMSE 最低的模型未必适合 MPC？

科研与工程中一个极重要的认知：**面向预测的模型 $\neq$ 面向控制的模型（Control-Oriented Modeling）**。
- **物理增益符号（Monotonicity / Sign Consistency）**：树模型（LightGBM）是分段常数函数，没有解析梯度，且在训练数据存在共线性时可能学出反方向响应（如蒸汽升高反而预测水分升高）。在闭环里，这会导致控制器越调越偏。
- **外推能力（Extrapolation）**：树模型无法外推，一旦遇到训练集边界外的工况就会输出常数；物理/状态空间模型天然支持合理外推。
- **灰箱混合建模（Hybrid Gray-Box）**：`s5_mlflow_tracking.py` 中的 `Hybrid-GrayBox` 展示了工业与论文中极具价值的折中思路：

$$\hat{y}_{\text{hybrid}}(k+h) = \underbrace{\hat{y}_{\text{FOPDT}}(k+h)}_{\text{保物理方向、保外推}} + \underbrace{f_{\text{resid}}(X_k)}_{\text{强正则补偿非线性残差}}$$

---

## 三、增广状态空间与无静差线性 MPC（Offset-free MPC）

对应脚本：`src/s6_mpc_closed_loop.py`

### 3.1 含纯滞后的增广状态空间

设偏差控制量 $\tilde{u}_k = u_k - u_{\text{base}}$，把过去 $\theta_1$ 拍蒸汽和 $\theta_2$ 拍热风放进状态向量 $x_k \in \mathbb{R}^{2+\theta_1+\theta_2}$：

$$x_{k+1} = A x_k + B \tilde{u}_k, \qquad y_k = C x_k + y_{\text{base}}$$

### 3.2 增量控制（Velocity Form）的精确自由/强制分解

令优化变量为未来 $N_c$ 步的**控制增量** $\Delta U = [\Delta u_0, \dots, \Delta u_{N_c-1}]^T$，其中 $\tilde{u}_{k+j} = \tilde{u}_{k-1} + \sum_{r=0}^{\min(j, N_c-1)} \Delta u_r$：

$$Y = Y_{\text{free}} + G \Delta U$$

- **自由响应 $Y_{\text{free}}$**：必须按“未来所有控制量保持 $\tilde{u}_{k-1}$ 不变”递推 $x_{h+1} = Ax_h + B\tilde{u}_{k-1}$ 计算（⚠️ 早期代码直接算 $A^h x_k$，等价于假设未来输入瞬间归零到 $u_{\text{base}}$，造成严重基准错配）。
- **阶跃响应矩阵 $G$**：第 $j$ 列对第 $i$ 步的贡献为 $G_{i,j} = \sum_{l=0}^{i-j} C A^l B$。因为 $A, B, C$ 时不变，$G$ 只需在初始化时计算一次。

### 3.3 无量纲归一化与凸二次规划（QP）

当两个操纵量量纲悬殊（蒸汽 $\Delta u_1 \le 12$，热风 $\Delta u_2 \le 600$），不归一化直接写 $\|\Delta U\|_R^2$ 会导致 Hessian 条件数恶化数千倍。归一化目标函数为：

$$\min_{\Delta U,\, s_l \ge 0,\, s_h \ge 0} \sum_{i=1}^{N_p} \left[ Q_m \left(\frac{y_{m,i}-r_m}{s_{ym}}\right)^2 + Q_t \left(\frac{y_{t,i}-r_t}{s_{yt}}\right)^2 \right] + \sum_{j=0}^{N_c-1} \sum_{c=1}^{2} R_c \left(\frac{\Delta u_{c,j}}{s_{uc}}\right)^2 + \rho_s (\|s_l\|^2 + \|s_h\|^2)$$

这是一个**严格凸二次规划（Convex QP）**：
- **硬约束**：执行器物理上下限 $u_{\min} \le u \le u_{\max}$、变化率 $|\Delta u| \le \Delta u_{\max}$。
- **软约束（Slack）**：出口水分边界 $y_{m,\min} - s_l \le y_m \le y_{m,\max} + s_h$，确保即使遭遇极端扰动，QP 也**永远可行（Always Feasible）**。
- **求解器选择**：因为 Hessian $P$ 和约束矩阵 $A_{\text{con}}$ 恒定，**OSQP** 只需在初始化时做一次矩阵分解，在线每步仅更新线性项 $(q, l, u)$，单步耗时可压至 **< 1 ms**，比通用非线性求解器 IPOPT 快一个数量级以上。

### 3.4 为什么必须有 Offset-free 扰动观测器？

真实工厂里**永远存在模型失配**（增益偏 15%、延迟漂移、执行器滞后、未测来料变化）。若只用标称模型预测，闭环必然留下稳态静差。引入 DMC / Luenberger 输出扰动观测器：

$$\hat{d}_k = (1 - k_{\text{est}})\hat{d}_{k-1} + k_{\text{est}} \big(y_{\text{meas}}(k) - y_{\text{model}}(k \mid k-1)\big)$$

并将 $\hat{d}_k$ 叠加到整个未来 $N_p$ 步的自由响应中。稳态时 $\hat{d}_\infty = y_{\text{meas}} - y_{\text{model}}$，自动把所有常值模型误差和未测扰动完全抵消，实现**零稳态静差（Zero Steady-State Offset）**。

---

## 四、科研基本功自查清单（每次做实验前核对）

1. **杜绝任何形式的数据泄漏**：
   - 时间序列严禁随机打乱，必须按时间顺序或独立批次切分；
   - **先切分数据集，再仅在训练集上 `fit` 标准化器（Scaler）**，验证/测试集只做 `transform`；
   - 严禁跨批次拼接滑动窗口。
2. **同口径基线对比（Apples-to-Apples）**：
   - 预测 $H$ 步的模型，必须与 **$H$ 步 Persistence**（$y(t-H)$）对比，不能拿 1 步 Persistence 混为一谈；
   - 平滑信号上 $R^2$ 极易虚高，必须同时看 **MAE / RMSE / Bias / 相对基线提升率**。
3. **警惕维度与样本量失衡（$p \gg n$）**：
   - 展平窗口特征数为 $n_{\text{feat}} \times L$（如 $10 \times 120 = 1200$），训练样本数至少应显著大于特征维数，否则必须降采样、做特征提取或加大正则化。
4. **看到离谱数字先查代码，不盲信模型**：
   - 求解失败次数、残差自相关、物理方向检查（增益符号）、跨场景状态重置，缺一不可。
5. **用“模型失配”检验控制算法真本事**：
   - 控制器用“上帝视角真值参数”跑通只是第一步；必须用**从数据辨识出的模型**去控**带失配、滞后与未测扰动的对象**，才能证明算法具备工程与学术价值。
