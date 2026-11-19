
# HDT 气流烘丝机预测模型 —— 软件包与训练模型详解

> 依据：《HDT气流烘丝机预测模型数据集构建研究报告_修订版》
> 目标：在**还没有真实数据**的阶段，把工具链和模型全部准备好、验证通。
> 生成时间：2026-11-19

---

## 零、先看清楚报告说了什么

报告是一份**数据需求与工艺机理分析**文件，它的核心主张可以压缩成 5 条：

1. **动态，不是静态拟合。** 同一时刻的蒸汽流量和出口水分**不对应同一批物料**。必须用滑动窗口 `X(k)=[u(k−L+1:k), d(…), y(…)]` → `Y(k)=[Wout(k+1:k+Np), Tout(…)]`。
2. **双输出 MIMO。** 出口含水率 + 出口温度，两个都是目标。
3. **两个操纵量（u）非常有限。** 只有蒸汽流量、热风流量。入口水分、入口温度、进料量是**可测扰动 d**，不是操纵量。
4. **延迟分三层。** 物料输送时间 + 工艺动态时间 + 测量链路时间，不能混为一谈。
5. **模型路线是递进的**（报告 7.2）：可解释基线 → 非线性表格 → 时序深度 → MPC 友好模型。

**注意：报告第 7.2 节的模型路线是主线索。** 你现在的全部前期工作，都应该围绕这条路线展开。

---

## 一、环境里已装好的软件包全景

全部 25+ 包已验证可用（`dryer-ts` 环境，Python 3.11，GPU: RTX 4060 8GB）。

### 1.1 按报告环节分组的包清单

| 报告环节                | 软件包             | 版本         | 作用                                          |
| ----------------------- | ------------------ | ------------ | --------------------------------------------- |
| **数据读取/质量** | `pandas`         | 3.1.1        | 时间戳对齐、重采样、批次切分                  |
|                         | `pyarrow`        | 26.0.1       | 高效读写 parquet（比 CSV 快 10 倍+）          |
|                         | `scipy`          | 1.17.1       | 信号滤波、插值、互相关                        |
| **延迟辨识**      | `scipy.signal`   | 内置         | `correlate` 做互相关找滞后                  |
|                         | `sysidentpy`     | 0.9.0        | **ARX/NARX 系统辨识**（报告明确推荐）   |
|                         | `statsmodels`    | 0.15.0       | ARIMA、ACF/PACF、Granger 因果                 |
| **可解释基线**    | `sklearn`        | 1.9.1        | 线性回归、PLS、SVR、随机森林                  |
| **非线性表格**    | `lightgbm`       | 4.7.0        | 报告推荐，速度最快                            |
|                         | `xgboost`        | 3.2.0        | 报告推荐                                      |
| **时序深度**      | `darts`          | 0.48.1       | TCN、RNN、LightGBM 封装                       |
|                         | `neuralforecast` | 3.3.1        | TimeXer、iTransformer、NHITS、PatchTST        |
|                         | `pytorch`        | 2.15.0+cu132 | 底层，CUDA 可用                               |
|                         | `pytorch_tcn`    | 1.2.3        | **纯 TCN 实现**，比 Darts 轻            |
| **MPC 友好模型**  | `do-mpc`         | 5.1.2        | **★ 刚装** 完整 MPC 框架               |
|                         | `control`        | 0.10.2       | **★ 刚装** 状态空间、FOPDT、子空间辨识 |
|                         | `casadi`         | 3.8.2        | 符号计算与非线性优化                          |
| **MPC 求解**      | `cvxpy`          | 1.9.3        | 凸优化建模                                    |
|                         | `osqp`           | 1.1.3        | QP 求解器（MPC 常用）                         |
|                         | `scipy.optimize` | 内置         | `minimize` 做非线性 MPC                     |
| **物理约束**      | `neuromancer`    | 1.5.6        | 物理约束神经网络                              |
| **实验管理**      | `mlflow`         | 3.18.0       | 记录每次训练的超参和指标                      |
|                         | `optuna`         | 5.1.0        | 超参搜索                                      |
| **可视化**        | `matplotlib`     | 3.x          | 已解决中文字体问题                            |

### 1.2 关键包的定位（别用错）

```
                  可解释性 ↑
    FOPDT/ARX ─────────────────┐
    (sysidentpy)                │  MPC 最爱的形式
                                │
    状态空间 ───────────────────┤
    (control / do-mpc)          │
                                │
    树模型 ─────────────────────┤
    (lightgbm/xgboost)          │
                                │
    深度网络 ───────────────────┘
    (darts/neuralforecast)        精度 ↑（数据足够时）
```

**报告 7.2 节的原话**："建议先训练可解释的动态基线模型，再逐步比较非线性模型。用于MPC时，不能只比较单步RMSE，还要检查多步预测误差、输入变化后的响应合理性、预测稳定性和在线计算时间。"

这句话是**整个项目的技术纲领**，下面所有内容都围绕它展开。

---

## 二、训练模型详解（按报告 7.2 的四阶段）

### 阶段 1：可解释基线 —— FOPDT / ARX / NARX

**为什么先做这个？**

- 报告要求"检验主要通道、变量作用和滞后结构"
- 参数少（FOPDT 只有 K、τ、θ 三个参数），物理意义明确
- **MPC 可以几乎直接使用** —— 这是最大的价值
- 能快速确认：蒸汽↑ → 水分↓？滞后多少？时间常数多少？

#### FOPDT（一阶加纯滞后）

```
        K·e^(-θs)
G(s) = ───────────
        τs + 1
```

- **K**：增益（蒸汽 +100kg/h → 水分 -?%）
- **τ**：时间常数（响应 63% 所需时间）
- **θ**：纯滞后（比 τ 更容易被数据决定）

**怎么用（scipy curve_fit 即可，无需装包）：**

```python
import numpy as np
from scipy.optimize import curve_fit

def fopdt(t, K, tau, theta, u0, y0):
    """阶跃响应解析解。t 从阶跃时刻算起。"""
    y = np.full_like(t, y0, dtype=float)
    m = t > theta
    y[m] = y0 + K * u0 * (1 - np.exp(-(t[m] - theta) / tau))
    return y

# 从某次蒸汽阶跃实验的数据拟合
popt, _ = curve_fit(fopdt, t, y, p0=[-0.02, 20, 15, 30, 12.5],
                    bounds=([-1, 1, 0, 1, 0], [1, 500, 120, 500, 30]))
K, tau, theta = popt[:3]
```

**这是验证报告 6.2 节"延迟辨识"最直接的工具。**

#### ARX / NARX（sysidentpy 0.9.0）

ARX 是离散线性差分方程：

```
A(q)y(t) = B(q)u(t) + e(t)
```

即 `y(t) = a₁y(t−1) + ... + b₁u(t−1) + ... + e(t)`

**为什么推荐 sysidentpy？** 它能**自动做结构选择**（FROLS 算法），从一大堆候选滞后项里挑出真正重要的，输出的是显式方程 —— 可解释性极强。

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

model = FROLS(
    ylag=5,          # 自回归阶数
    xlag=[[1,2,3,10,20,30]],   # 输入的候选滞后（关键！）
    n_terms=8,       # 保留多少项
    basis_function=Polynomial(degree=2),   # 2 阶可捕捉非线性
    estimator=LeastSquares(),
)
model.fit(X=u_train, y=y_train)
model.final_model   # 看它选出了哪些项 ← 这就是物理洞察
```

> 你的 `src/benchmark_dryer.py` 里已经用了 FROLS，那条路是对的。

#### 状态空间 / 子空间辨识（control 0.10.2，刚装）

```python
import control as ct
# 子空间辨识：直接从数据得到状态空间模型
sys = ct.ssest  # 或 n4sid 风格
# FOPDT 也有现成实现
from control import tf
```

---

### 阶段 2：非线性表格模型 —— LightGBM / XGBoost

**适用前提**：滞后特征已经人为构造好。

```python
import lightgbm as lgb
import numpy as np

def make_lag_features(df, cols, lags):
    """构造滞后特征 —— 这是表格模型的核心工作量"""
    out = {}
    for c in cols:
        for l in lags:
            out[f"{c}_lag{l}"] = df[c].shift(l)
    return pd.DataFrame(out)

lags = [1, 2, 3, 5, 8, 12, 18, 24, 30, 45, 60]
X = make_lag_features(df, ["steam_flow", "hot_air_flow", "inlet_moisture"], lags)
```

**优点**：快、稳、不需要大量数据。
**缺点**：**无法外推**。树模型只能插值，遇到训练集没见过的工况会输出常数。这点在 MPC 里是致命问题——控制器会倾向于把操纵量推到训练边界。

> ⚠️ **重要提醒**：报告 6.2 强调延迟要辨识。树模型不会自己学延迟，**你必须把正确的滞后作为特征喂进去**。这就是为什么阶段 1 必须先做。

---

### 阶段 3：时序深度模型 —— TCN / RNN / Transformer

按你的场景（双输出、两个操纵量、强滞后），推荐顺序是：

#### ① TCN（temporal convolutional network）—— **首选**

**原理**：因果膨胀卷积

```
                    膨胀因子 1,2,4,8
    x(t-8)  x(t-4)  x(t-2)  x(t-1)  x(t)
       │       │       │       │      │
       └───────┴───────┴───────┴──────┴──→ 感受野 = 2^L 步
```

- **因果**：只看过去，不偷看未来（MPC 必须）
- **膨胀**：堆 L 层就能覆盖 2^L 步历史，参数却很少
- **快**：可并行，比 LSTM 快 5~10 倍

你的实测结果（`s3_dl_forecast.py`）：

```
TCN  RMSE=0.01636   R²=0.99923   ← 只比 Persistence 差 10.5%
```

**TCN 是四个模型里唯一接近理论下限的**，这也印证了它是 MPC 内部模型的合理选择。

用法（Darts）：

```python
from darts.models import TCNModel
tcn = TCNModel(
    input_chunk_length=120,      # 回看窗口 L
    output_chunk_length=15,       # 预测步长 Np
    kernel_size=3, num_filters=32, dilation_base=2, dropout=0.1,
    n_epochs=30, batch_size=64,
)
```

#### ② LSTM / GRU（Darts RNNModel）

- 长记忆好，但**串行计算慢**，训练不容易收敛
- 报告里列为可选，我建议作为对照，不作为主力

#### ③ TimeXer（neuralforecast）

**原理**：外生变量走"变量 token"，内因走"patch token"：

```
内因 y 的历史 [被切成 patch] ──→ patch token ─┐
                                              ├→ Transformer → 预测
外生变量 u,d [每个变量一个 token] ──→ var token ┘
```

**这个设计理论上最符合你的问题**：用蒸汽/热风/入口水分来解释出口水分。

但你的实测结果是 **RMSE=0.052**，比 TCN 差 3.5 倍。这说明：

- 要么训练不够（2000 步对 Transformer 偏少）
- 要么 `input_size=120` 的 patch 划分不合适
- 需要调参，**不是模型本身不行**

> 这正好是一个很好的学习点：**架构再合适，也要调好才有用。**

#### ④ iTransformer

每个变量整体作为一个 token。擅长学**变量之间**的关系。

⚠️ 你实测 R²=0.299 是异常的（输出未反归一化），**暂时不要采信这个结果**。

---

### 阶段 4：MPC 友好模型 —— 状态空间 / 子空间辨识

这是**整个项目的终点**，也是报告 7.2 明确指出的方向。

**为什么 MPC 偏爱状态空间？**

```
x(k+1) = A·x(k) + B·u(k) + E·d(k)
y(k)   = C·x(k) + D·u(k)
```

- 可以**多步预测**（直接迭代 Np 次）
- 可以**约束优化**（u 的上下限、变化率）
- 计算快（矩阵乘法）
- 理论有保证（可控性、可观性、稳定性）

#### 子空间辨识（N4SID）

直接用输入输出数据估出 A/B/C/D，**不需要先猜模型结构**。这是工业 MPC 的标准做法。

```python
import control as ct
# control 包提供 stdio / n4sid 风格的子空间方法
```

#### do-mpc 5.1.2（刚装）

一个完整的 MPC 框架，包含：

- 模型定义（符号）
- 控制器配置（时域、约束、目标函数）
- 估计器（扩展卡尔曼）
- 仿真与闭环测试

```python
import do_mpc
model = do_mpc.model.Model('discrete')
```

#### 线性 MPC 的快速实现（casadi，无需 do-mpc）

```python
import casadi as ca
opti = ca.Opti()
U = opti.variable(2, Np)        # 两个操纵量的未来 Np 步
# 目标：出口水分跟踪设定值
# 约束：u 上下限、变化率、温度/压力安全限
opti.minimize(  # ... )
opti.subject_to( opti.bounded(u_min, U, u_max) )
sol = opti.solve()
```

---

## 三、报告里没写但你必须准备的东西

### 3.1 三层延迟辨识工具（报告 6.2 / 11.2）

报告把延迟分成三层，**代码上要分开处理**：

| 层           | 含义                   | 工具                  |
| ------------ | ---------------------- | --------------------- |
| 物料输送时间 | 从进料到出口检测点     | 追踪数据 / 累计投料量 |
| 工艺动态时间 | 热质传递的滞后+惯性    | 互相关 + FOPDT 拟合   |
| 测量链路时间 | 水分仪滤波、通讯、归档 | 查仪表手册            |

```python
import numpy as np
def find_lag(x, y, max_lag=120):
    """互相关找滞后。返回峰值位置。"""
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    cc = [np.corrcoef(x[:len(x)-k], y[k:])[0,1] for k in range(max_lag)]
    return int(np.argmax(np.abs(cc)))
```

⚠️ 报告 6.2 明确警告：**闭环控制、共同扰动、趋势会干扰互相关的解释**。所以互相关只当作初筛，还要用 FOPDT 拟合和阶跃事件验证。

### 3.2 多步预测评估（报告 7.3）

报告明确要求："**分别报告 1 步和多个预测步长的误差**"。

```python
for nstep in [1, 5, 10, 15, 30, 60]:
    # 评估误差随预测时域的增长
    # 如果误差增长过快 → 模型不适合长时域 MPC
```

**这一条非常关键**：单步 RMSE 好看不代表能用于 MPC，必须看多步。

### 3.3 状态筛选

报告 2.3 节要求标记：停机、预热、待料、启动、稳定、断料、收尾、清洗、故障。

第一版模型**只用稳定生产段**。所以你需要一个状态筛选器：

```python
df_stable = df[df["machine_state"] == "stable"]
```

---

## 四、你现在（没有数据）能做的 6 件事

### ✅ 1. 把仿真器升级成"符合报告"的版本

你现在的 `simulate_hdt_batch` 有几个不符合报告的地方：

| 项目      | 现状              | 报告要求                       | 该改什么               |
| --------- | ----------------- | ------------------------------ | ---------------------- |
| 输出      | 双输出 ✅         | 双输出 ✅                      | 保持                   |
| 操纵量    | 蒸汽、热风 ✅     | 蒸汽、热风 ✅                  | 保持                   |
| 扰动      | 入口水分、进料 ✅ | 同 ✅                          | 保持                   |
| 缺失变量  | 无                | P1: 入口温度、工艺气体温度     | **加上**         |
| 运行阶段  | 无（全程稳定）    | 需标记停机/启动/断料           | **加上**         |
| 噪声/尖峰 | 只有白噪声        | 需模拟测点冻结、尖峰、通讯中断 | **加上**         |
| 延迟      | 固定 12~18s       | 三层延迟，随负荷变化           | **改成可变延迟** |
| 采样      | 1 Hz              | 应能降采样实验                 | **加降采样开关** |

**升级后的仿真器可以让你在没有真实数据时，就验证你的清洗和辨识流程是否正确**——这是最有价值的前期工作。

### ✅ 2. 实现完整的延迟辨识流水线（三层）

在仿真数据上跑一遍：

- 互相关找初筛滞后
- FOPDT 拟合得到 K、τ、θ
- 从阶跃事件验证
- 与真实滞后对比（仿真里你**知道**答案，可以验证方法对不对）

**这是"已知答案的考试"**，价值极高。

### ✅ 3. 实现滑动窗口样本构造器

完全按报告 6.3 的公式：

```python
X(k) = [u(k−L+1:k), d(k−L+1:k), y(k−L+1:k)]
Y(k) = [Wout(k+1:k+Np), Tout(k+1:k+Np)]
```

写成可复用的函数，等真实数据来了直接用。

### ✅ 4. 实现多步预测评估器

按报告 7.3：1 步、5 步、15 步、60 步分别报 MAE/RMSE/R²/Bias，并画**误差随时域增长曲线**。

### ✅ 5. 搭好 MLflow 实验记录

每次训练自动记录：数据版本、超参、指标、模型文件。这样等你跑了几十个实验后不会乱。

### ✅ 6. 实现一个最小 MPC 闭环（在仿真器上）

用 FOPDT/状态空间 + casadi 做 MPC，在仿真器上闭环测试：

- 设定值跟踪
- 扰动抑制
- 约束满足

**这一步做完，你就走完了报告的完整路线**，等真实数据只需替换数据源。

---

## 五、推荐的技术选型（直接照做）

考虑到你的时间有限，我给一个**最小可行路线**：

```
第 1 周：仿真器升级 + 延迟辨识流水线
         ↓（验证方法正确性）
第 2 周：滑动窗口 + 基线模型（FOPDT/ARX/LightGBM）
         ↓
第 3 周：多步预测评估 + MLflow 记录
         ↓
第 4 周：TCN 深度模型（已知它表现最好）
         ↓
第 5 周：状态空间辨识 + 最小 MPC 闭环
         ↓
拿到真实数据后：替换数据源，重跑全流程
```

**为什么不先做 TimeXer？**
因为数据量大才需要 Transformer。你现在没有数据，它的优势发挥不出来。而且 TCN 在你实测中最好，MPC 也更友好。

---

## 六、快速命令

```bash
# 环境体检
/opt/conda_envs/dryer-ts/bin/python ~/文档/deepseek-harness/default-workspace/check_env.py

# 装包（阿里镜像，快 10 倍）
/opt/conda_envs/dryer-ts/bin/python -m pip install -i https://mirrors.aliyun.com/pypi/simple 包名

# 当前已跑的脚本
cd ~/hdt_dryer_mpc
python src/s1_explore.py
python src/s2_baseline.py
python src/s3_dl_forecast.py
```

---

## 七、一句话总结

**报告给的是"要什么数据、怎么对齐、怎么验证"，不是"用什么包"。**
你现在的软件包已经**完全覆盖**报告 7.2 提出的全部四阶段模型路线。
在没有数据的阶段，最有价值的不是调模型，而是**把数据处理和辨识流程写对、在已知答案的仿真数据上验证**。
因为等真实数据来了，你会发现 90% 的时间花在数据质量和延迟辨识上，而不是模型结构。
