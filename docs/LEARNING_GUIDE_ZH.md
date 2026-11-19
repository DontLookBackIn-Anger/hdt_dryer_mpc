# HDT 气流烘丝机 —— 预测模型学习路线

这份文档带你从零熟悉预测模型。**每个脚本你都可以自己跑，跑完对照这里的讲解理解结果。**

---

## 一、先跑起来

```bash
# 1. 激活环境
conda activate dryer-ts
#    如果 conda 激活有问题，直接用绝对路径的 python：
#    /opt/conda_envs/dryer-ts/bin/python

# 2. 进入项目
cd ~/hdt_dryer_mpc

# 3. 按顺序跑（每个脚本独立可跑）
python src/s1_explore.py       # 看数据（~5 秒）
python src/s2_baseline.py      # 基线模型（~60 秒）
python src/s3_dl_forecast.py   # 深度模型（~5 分钟，用 GPU）
```

所有产出都在 `outputs/`：

- `outputs/figures/` —— 图
- `outputs/reports/` —— CSV 指标 + txt 结论

> 图里的中文如果显示成方框，说明字体缓存问题。`src/common.py` 已自动注册 Noto CJK 字体，正常不会出现。

---

## 二、⚠️ 最重要的一个发现：你的基准环境有个陷阱

我跑完 `s2_baseline.py` 的结果是这样的：

| 模型                         | RMSE             | MAE    | R²    |
| ---------------------------- | ---------------- | ------ | ------ |
| ① Persistence（复制上一步） | **0.0145** | 0.0118 | 0.9994 |
| ③ ExponentialSmoothing      | 0.0231           | 0.0171 | 0.9985 |
| ⑤ LightGBM + 外生变量       | 0.0406           | 0.0324 | 0.9953 |
| ④ LightGBM（仅自回归）      | 0.0709           | 0.0513 | 0.9857 |
| ② 24 步滑动均值             | 0.1322           | 0.1160 | 0.9504 |

**请仔细看第一行。** 那个「什么都不学，直接把上一秒的值当作下一秒的预测」的 Persistence，RMSE 只有 0.0145，**比所有正经模型都准**。

这不是 Bug，是 `simulate_hdt_batch` 这个仿真的一个特性：

```python
w_out[k] = w_out[k-1] + 0.05 * (w_target - w_out[k-1]) + rng.normal(0, 0.01)
```

每步只朝目标移动 **5%**，再加上 N(0, 0.01) 的小噪声。所以相邻两秒的值几乎一样——**信号本身变化太慢，可预测的上限被"惯性"锁死了**。

这带来两个必须记住的教训：

### 教训 1：R² 高不代表模型好

大家习惯看 R²，但 R² 对**平滑信号**极不敏感。上面 Persistence 的 R² = 0.9994，看起来"完美"，实际它什么都没学。判断模型好坏要看 **RMSE / MAE**，而且必须**和 Persistence 对比**。

### 教训 2：深度学习很可能打不过 Persistence

在 15 秒预测步长下，预测难度非常低。你后面跑 TCN / TimeXer 时，**大概率会出现「深度模型 RMSE 反而不如 Persistence」的局面**。这不是环境坏了，是这个基准数据太"平滑"。

真实 HDT 数据不会这样：真实工况有来料波动、蒸汽压力抖动、清洗换牌、断料，出口水分真正在动。所以——

> **换到真实数据之前，任何关于"哪个模型更好"的结论都是不可靠的。**

---

## 二·五、S3 实测结果（2026-11-18 实跑）

| 模型                         | RMSE              | MAE     | R²     | 相对 Persistence             |
| ---------------------------- | ----------------- | ------- | ------- | ---------------------------- |
| ① Persistence（复制上一步） | **0.01480** | 0.01218 | 0.99937 | — 下限                      |
| ④ **TCN (Darts)**     | **0.01636** | 0.01333 | 0.99923 | **-10.5%** ← 几乎追平 |
| ② TimeXer                   | 0.05204           | 0.04148 | 0.99225 | -251.5%                      |
| ③ iTransformer              | 0.49482           | 0.40525 | 0.29910 | -3242.5%                     |

**怎么读这张表：**

**TCN 值得表扬。** RMSE 只比 Persistence 差 10.5%，说明它已经学到了接近仿真噪声水平的下限。对于"预测下一秒"这种任务，这基本就是天花板了。TCN 也确实是 MPC 内部模型的合理选择：因果卷积、不偷看未来、推理快。

**TimeXer 差 3.5 倍，需要排查。** 它理论上是本场景最合适的模型（外生变量 token + 内因 patch token）。差距这么大，可能原因：

- 训练步数不够（2000 步对 Transformer 偏少）
- `input_size=120` 对它的 patch 划分不合适
- 学习率需要调

**iTransformer 结果异常，别信。** R²=0.299 不是"模型弱"，而是出了问题：**它输出的是未反归一化的值**。实测预测值 12.75、实际值 12.5，系统性偏高。这是 `iTransformer` + `use_norm=True` 在 NeuralForecast 3.3.1 下的行为问题。

> 教训：**看到离谱的数字先怀疑代码，不要怀疑模型。** 我第一版把 `cross_validation` 用错，导致 R²=-2.29，排查后才发现是只预测了 15 个点却拿去和 1200 点比较。

---

## 二·六、NeuralForecast 踩坑记录（省你自己踩）

**坑 1：`nf.predict(df=...)` 只返回 h 个点**

```python
fc = nf.predict(df=test_nf)   # 只有 15 行！
# 拿它和 len=1200 的测试集比较 → 错位 → R² 变成负数
```

**坑 2：`cross_validation(refit=False)` 与 `input_size` 冲突**

```
Exception: Time series is too short for training,
           consider setting a smaller input size or set start_padding_enabled=True
```

原因：CV 每个窗口只有 h 个点，短于 `input_size=120`。加 `start_padding_enabled=True` 也不解决。

**坑 3：可行方案是手动滚动**

```python
for t in range(INPUT_SIZE, len(y_test), STRIDE):
    ctx = pd.concat([train_nf, test_nf.iloc[:t]])   # 截到当前时刻
    fc = nf.predict(df=ctx)
    sub = fc[fc["unique_id"] == "test"]             # 只取测试序列
    pred = sub[col].iloc[0]
```

慢（~100s/模型），但结果可靠。

**坑 4：多条序列必须错开时间轴**
3 个训练批次时间戳起点相同，直接 concat 会重叠，CV 返回 4 倍重复样本（n=4800 而非 1200）。解决：

```python
o["ds"] = o["ds"] + pd.Timedelta(days=offset)   # 每条错开
```

**坑 5：`n_series` 必须等于实际序列数**，否则形状报错：

```
RuntimeError: Sizes of tensors must match except in dimension 2
```

**坑 6：训练步数要够**
60 步 → RMSE 0.38（没学会）；2000 步 → 0.052（收敛）。差一个数量级。

---

## 三、每个脚本在讲什么

### `s1_explore.py` —— 看懂数据

三个必看的东西：

**① ACF（自相关）**：回答"目标自身的记忆有多长"。
实测结果：ACF 从 1.0 衰减到 0.5 需要 **58 秒**。这意味着目标极其平滑，是上面"精度虚高"的根源。

**② 互相关**：回答"输入比输出提前多少步"。
实测峰值滞后：蒸汽流量 29 秒、热风流量 47 秒、入口水分/物料流量 ≥60 秒。
**这个数直接决定模型的回看窗口 `input_size` 该怎么设**——至少要比最大滞后长，否则模型"看不到"激励。

**③ 分布与统计**：出料水分均值约 12.5%，标准差很小。

> 真实数据上，这一步会看到更多东西：停机段、正弦之外的漂移、缺失值、传感器跳变。**建模前先看数据**是铁律。

---

### `s2_baseline.py` —— 先建立下限

核心思想：**在造复杂模型之前，先知道"最笨的方法"能做到多好。**

模型原理速览：

| 模型                 | 原理                           | 一句话理解                 |
| -------------------- | ------------------------------ | -------------------------- |
| Persistence          | ŷ(t) = y(t-1)                 | 复制上一步，绝对下限       |
| 滑动均值             | ŷ(t) = mean(y[t-24:t])        | 用最近平均，检验噪声       |
| ExponentialSmoothing | 水平+趋势+季节分解，近期加权高 | 经典统计方法，几十年的老牌 |
| LightGBM             | 把过去 60 步当 60 个特征做回归 | 梯度提升树，特征工程派     |
| LightGBM+外生        | 同上 + 蒸汽/热风/入口水分      | ★ 检验物理量是否真的有用  |

**这里唯一有实际意义的结论**：加入外生变量后 RMSE 从 0.0709 → 0.0406，**提升 42.7%**。

这说明：**只靠目标自身历史是不够的，工艺操作量和扰动确实驱动出口水分。** 这对 MPC 至关重要——控制必须有"抓手"，而抓手就是这些操作量。

> 技巧：这里的对比是公平的（同一训练集、同一测试批次、同一滞后长度）。做实验对比时，**只改一个变量**，否则你永远不知道提升来自哪里。

---

### `s3_dl_forecast.py` —— 深度模型

去跑它，感受时间维度。讲解见脚本内注释的"原理"部分。

---

## 四、关于真实数据的建议（拿到数据后照做）

你现在的仿真数据只适合**熟悉工具链**。拿到真实数据后：

1. **先替换数据，再调模型。** 把 `data/raw/` 里的真实数据接入 `common.py`，别急着改模型。
2. **重新做 EDA。** 真实数据的滞后可能完全不同，`input_size` 要重新定。
3. **重新算 Persistence 基线。** 真实数据的这个数会明显变差，那时模型才有发挥空间。
4. **注意采样频率。** 1 Hz 对机器学习偏快（信息冗余大）。真实数据可能是 1 min 或更粗，届时 `H`、`input_size` 全部要重设。
5. **警惕数据泄漏。** 时间序列**绝不能随机打乱**切分。`s2` 里特意用"3 批次训练 + 1 独立批次测试"就是为了避免同批次内偷看未来。

---

## 五、文件说明

| 文件                       | 作用                                              |
| -------------------------- | ------------------------------------------------- |
| `src/common.py`          | 公共模块：数据生成、指标计算、中文绘图、路径      |
| `src/s1_explore.py`      | 学习脚本 1：数据探索                              |
| `src/s2_baseline.py`     | 学习脚本 2：基线模型                              |
| `src/s3_dl_forecast.py`  | 学习脚本 3：深度模型                              |
| `src/benchmark_dryer.py` | 你原有的 benchmark（SysIdentPy / LightGBM / TCN） |
| `outputs/figures/`       | 图片                                              |
| `outputs/reports/`       | 指标 CSV 与文字报告                               |

---

## 六、环境速查

```bash
# 环境体检（25 个包是否都正常）
/opt/conda_envs/dryer-ts/bin/python ~/文档/deepseek-harness/default-workspace/check_env.py

# 装包用阿里镜像（比官方源快约 10 倍）
/opt/conda_envs/dryer-ts/bin/python -m pip install -i https://mirrors.aliyun.com/pypi/simple 包名
```

GPU：RTX 4060 Laptop 8GB，CUDA 可用，深度模型会自动用它。
