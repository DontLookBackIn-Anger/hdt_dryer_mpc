#!/usr/bin/env python3
"""
=== 第 1 项：升级仿真器（对齐研究报告要求）===

原 simulate_hdt_batch 缺失的 7 项，本模块全部补齐：

  ┌───┬────────────────────┬──────────────────────────────────────┐
  │ # │ 报告要求            │ 本模块实现                            │
  ├───┼────────────────────┼──────────────────────────────────────┤
  │ 1 │ 运行阶段标记(2.3)   │ machine_state: 停机/预热/待料/启动/   │
  │   │                    │ 稳定/断料/收尾/清洗/故障              │
  │ 2 │ P1 变量(4.1)        │ inlet_temp, process_gas_temp,         │
  │   │                    │ return_gas_temp, gas_pressure         │
  │ 3 │ 测点冻结(5.4)       │ frozen 标记 + 真实冻结段              │
  │ 4 │ 可变延迟(11.2)      │ 延迟随进料负荷线性变化                │
  │ 5 │ 降采样开关(5.1)     │ target_freq 参数                      │
  │ 6 │ 异常尖峰(5.4)       │ spike 标记 + 真实尖峰                 │
  │ 7 │ 通讯中断(5.4)       │ comm_loss 标记 + 真实断点             │
  └───┴────────────────────┴──────────────────────────────────────┘

另外按报告 5.3 补齐了**标签质量码**（quality_flag），对应出口水分仪的
质量码，用于训练时的样本筛选。

核心设计原则（很重要）：
    仿真器把"真实值"和"观测值"分开保存。
      *_true  : 物理真值，无噪声无异常 —— 用于验证辨识方法是否找对了
      （无后缀）: 传感器读数，含噪声/尖峰/冻结/中断 —— 用于训练
    这样你可以在"已知答案"的情况下检验延迟辨识、清洗算法是否正确。
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 常量
MV_COLS = ["steam_flow", "hot_air_flow"]
DV_COLS = ["inlet_moisture", "inlet_temp", "feed_rate"]
PV_COLS = ["process_gas_temp", "return_gas_temp", "gas_pressure"]
CV_COLS = ["outlet_moisture", "outlet_temp"]
EXOG_COLS = MV_COLS + DV_COLS + PV_COLS

# 传感器读数（含异常）与真值列
SENSOR_TRUE_COLS = {
    "steam_flow": "steam_flow_true",
    "hot_air_flow": "hot_air_flow_true",
    "inlet_moisture": "inlet_moisture_true",
    "feed_rate": "feed_rate_true",
    "outlet_moisture": "outlet_moisture_true",
    "outlet_temp": "outlet_temp_true",
}

# 运行状态编码（对应报告 2.3）
STATE_CODES = {
    "offline": 0, "preheat": 1, "idle": 2, "startup": 3, "stable": 4,
    "feed_stop": 5, "shutdown": 6, "cleaning": 7, "fault": 8,
}
STATE_CN = {
    "offline": "停机", "preheat": "预热", "idle": "待料", "startup": "进料启动",
    "stable": "稳定生产", "feed_stop": "短时断料", "shutdown": "生产收尾",
    "cleaning": "清洗维护", "fault": "故障联锁",
}


def _fopdt_step(u: np.ndarray, K: float, tau: float, theta: float, dt: float) -> np.ndarray:
    """一阶惯性 + 纯滞后的离散仿真（theta 单位：秒，可随负荷变化）。

    这是整个仿真器的核心：所有动态响应都由它产生。
    """
    n = len(u)
    y = np.zeros(n)
    alpha = dt / (tau + dt)
    theta_i = np.atleast_1d(theta)
    for k in range(n):
        th = theta_i[k] if len(theta_i) > 1 else theta_i[0]
        lag = int(round(th / dt))
        src = u[k - lag] if k >= lag else u[0]
        y[k] = y[k - 1] + alpha * (src - y[k - 1]) if k else src
    return y


def simulate_hdt_full(
    n_steps: int = 3600,
    dt: float = 1.0,
    seed: int = 42,
    *,
    include_states: bool = True,
    include_faults: bool = True,
    variable_delay: bool = True,
    target_freq: float | None = None,
    base_moisture: float = 12.5,
) -> pd.DataFrame:
    """生成一段完整的 HDT 生产数据（默认 1 小时，1 Hz）。

    Parameters
    ----------
    n_steps : 总步数
    dt      : 采样周期（秒）
    seed    : 随机种子
    include_states : 是否注入运行阶段切换（停机/启动/断料等）
    include_faults : 是否注入尖峰/冻结/通讯中断
    variable_delay : 延迟是否随进料负荷变化
    target_freq    : 若给出，降采样到该频率（秒），如 5.0
    base_moisture  : 基准出口水分

    返回列：
        时间戳 timestamp, 采样序号 step, 运行状态 machine_state(编码)/machine_state_cn
        MV: steam_flow, hot_air_flow
        DV: inlet_moisture, inlet_temp, feed_rate
        PV: process_gas_temp, return_gas_temp, gas_pressure
        CV: outlet_moisture, outlet_temp
        真值: *_true
        质量: quality_flag（0=正常, 1=冻结, 2=尖峰, 3=中断, 4=降级）
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * dt

    # ================================================================
    # 1. 运行阶段时间线（报告 2.3）
    # ================================================================
    state = np.full(n_steps, "stable", dtype=object)
    # 事件位置默认值（无状态注入时也要有，供阶跃事件使用）
    s1 = int(n_steps * 0.35)
    s2 = int(n_steps * 0.85)
    if include_states and n_steps >= 1200:
        # 前 180 秒预热+待料，中间一次断料，末尾收尾
        state[:180] = "preheat"
        s1 = int(n_steps * 0.35)
        state[s1:s1 + 60] = "feed_stop"          # 60 秒短时断料
        s2 = int(n_steps * 0.85)
        state[s2:] = "shutdown"
        state[:60] = "idle"

    is_run = np.isin(state, ["stable", "startup", "feed_stop", "shutdown"])

    # ================================================================
    # 2. 物料负荷（进料量）—— 延迟随它变化
    # ================================================================
    feed_base = 3800.0
    feed_rate_true = feed_base + 180.0 * np.sin(2 * np.pi * t / 600.0) + rng.normal(0, 15, n_steps)
    # 断料段进料量归零（真实事件）
    feed_rate_true = np.where(state == "feed_stop", 0.0, feed_rate_true)
    feed_rate_true = np.where(is_run, feed_rate_true, feed_rate_true * 0.05)
    feed_rate_true = np.maximum(feed_rate_true, 0.0)

    # ================================================================
    # 3. 操纵量 MV（报告 4.1：实际值，另存设定值）
    # ================================================================
    steam_sp = 260.0 + 30.0 * np.sin(2 * np.pi * t / 300.0)
    steam_sp[s1:s1 + 260] += 28.0          # 一次明显的阶跃（用于延迟辨识）
    steam_sp[s2 - 300:s2 - 150] -= 20.0    # 反向阶跃

    hot_air_sp = 9500.0 + 450.0 * np.cos(2 * np.pi * t / 420.0)
    hot_air_sp[s2 - 500:s2 - 320] -= 400.0  # 热风独立阶跃

    # 执行器一阶跟随 + 饱和（设定值 ≠ 实际值，报告 4.1 强调）
    def actuator_follow(sp, tau_s, jitter, lo, hi):
        a = dt / (tau_s + dt)
        out = np.zeros(n_steps)
        out[0] = sp[0]
        for k in range(1, n_steps):
            out[k] = out[k - 1] + a * (sp[k] - out[k - 1])
        return np.clip(out + rng.normal(0, jitter, n_steps), lo, hi)

    steam_flow_true = actuator_follow(steam_sp, 4.0, 1.8, 0.0, 400.0)
    hot_air_flow_true = actuator_follow(hot_air_sp, 6.0, 35.0, 0.0, 12000.0)
    # 停机段操纵量降为 0
    steam_flow_true = np.where(is_run, steam_flow_true, 0.0)
    hot_air_flow_true = np.where(is_run, hot_air_flow_true, 0.0)

    # ================================================================
    # 4. 扰动量 DV 与工艺状态 PV（报告 4.1 的 P0/P1）
    # ================================================================
    inlet_moisture_true = 20.0 + 0.7 * np.sin(2 * np.pi * t / 500.0) + 0.3 * np.sin(2 * np.pi * t / 1100.0) \
                          + rng.normal(0, 0.08, n_steps)
    # 来料水分尖峰（真实扰动）
    inlet_moisture_true[s2 - 800:s2 - 780] += 1.6

    inlet_temp_true = 25.0 + 2.0 * np.sin(2 * np.pi * t / 900.0) + rng.normal(0, 0.25, n_steps)

    # 工艺气体温度：由热风+蒸汽决定，含自身动态
    gas_drive = 0.010 * hot_air_flow_true + 0.020 * steam_flow_true
    process_gas_temp_true = 120.0 + _fopdt_step(gas_drive, 1.0, 25.0, 8.0, dt) * 0.9
    process_gas_temp_true += rng.normal(0, 0.6, n_steps)

    return_gas_temp_true = 0.72 * process_gas_temp_true + 25.0 + rng.normal(0, 0.5, n_steps)
    gas_pressure_true = 120.0 + 0.004 * hot_air_flow_true + rng.normal(0, 0.4, n_steps)

    # ================================================================
    # 5. 可变延迟（报告 11.2：延迟随负荷/输送速度变化）
    # ================================================================
    base_theta_steam, base_theta_air = 14.0, 12.0
    if variable_delay:
        # 进料量 3800→3000 时延迟增加约 3 秒（负荷降低，物料走得更慢）
        load_factor = np.clip((feed_base - feed_rate_true) / 800.0, -0.6, 1.2)
        theta_steam = base_theta_steam + 3.0 * load_factor
        theta_air = base_theta_air + 3.0 * load_factor
    else:
        theta_steam = np.full(n_steps, base_theta_steam)
        theta_air = np.full(n_steps, base_theta_air)

    # ================================================================
    # 6. 出口水分 / 温度（报告 3.1 / 3.2）
    # ================================================================
    # 蒸汽通道（增益为负：蒸汽↑ → 水分↓）
    d_steam = _fopdt_step(steam_flow_true - 260.0, 1.0, 22.0, theta_steam, dt)
    d_air = _fopdt_step(hot_air_flow_true - 9500.0, 1.0, 18.0, theta_air, dt)
    d_inlet = _fopdt_step(inlet_moisture_true - 20.0, 1.0, 30.0, np.full(n_steps, 18.0), dt)
    d_feed = _fopdt_step(feed_rate_true - feed_base, 1.0, 25.0, np.full(n_steps, 16.0), dt)

    outlet_moisture_true = (base_moisture
                            + 0.42 * d_inlet          # 入口水分 +1% → 出口 +0.42%
                            - 0.0190 * d_steam        # 蒸汽 +100kg/h → 出口 -1.9%
                            - 0.00062 * d_air         # 热风 +1000 → 出口 -0.62%
                            + 0.00035 * d_feed)       # 进料 +1000kg/h → 出口 +0.35%
    outlet_moisture_true += rng.normal(0, 0.012, n_steps)

    # 出口温度（报告 3.1 热量传递）：
    # 热风流量是热量的主要载体，强化其对温度的影响，
    # 否则稳定段温度信噪比过低（模型无法学到有效关系）。
    outlet_temp_true = (65.0
                        + 0.00320 * d_air         # 热风 +1000 → +3.2℃
                        + 0.0220 * d_steam        # 蒸汽 +100kg/h → +2.2℃
                        - 0.62 * d_inlet          # 入口水分 +1% → -0.62℃（蒸发吸热）
                        + 0.00075 * d_feed)       # 进料 +1000 → +0.75℃
    outlet_temp_true += rng.normal(0, 0.06, n_steps)

    # 非生产段（预热/待料/清洗）：没有干燥作用，物料不流动，
    # 出口测点读的是静态物料或空管，物理上应截然不同。
    # 若不区分，这些点会以 22% 的"湿料"污染统计和训练集。
    non_prod = ~is_run
    outlet_moisture_true = np.where(non_prod, np.nan, outlet_moisture_true)
    outlet_temp_true = np.where(non_prod, np.nan, outlet_temp_true)

    # ================================================================
    # 7. 传感器层：噪声 / 尖峰 / 冻结 / 通讯中断（报告 5.4）
    # ================================================================
    quality_flag = np.zeros(n_steps, dtype=int)     # 0 正常

    # 状态切换后的过渡期：物理上从"无加热"恢复到"稳定生产"需要时间
    # (报告 2.3：第一版稳态模型优先使用稳定生产数据)。
    # 这里只标记【非生产→生产】切换后的过渡，且只在稳定段内标记。
    warmup_s = 120      # 约 1.3 个时间常数，足够让惯性衰减
    for s_ in ("preheat", "idle", "feed_stop"):
        idx = np.where(state == s_)[0]
        if len(idx):
            end = idx[-1] + 1
            seg = slice(end, min(end + warmup_s, n_steps))
            # 仅当切换后确实是稳定生产时才标降级
            mask_stable = state[seg] == "stable"
            q_seg = quality_flag[seg]
            q_seg[mask_stable] = 4
            quality_flag[seg] = q_seg
    # 生产收尾（料尾）：出口仍有物料但已非稳态。
    # 报告 2.3 将其列为需要单独标记的工况，但数据本身是有效测量，
    # 因此不做质量码降级，交给 require_stable 过滤即可。


    def add_sensor_noise(x, noise, name, spike_prob=0.0006, freeze_prob=0.004):
        y = x + rng.normal(0, noise, n_steps)
        if not include_faults:
            return y
        # 尖峰
        idx = rng.random(n_steps) < spike_prob
        y[idx] += rng.choice([-1, 1], idx.sum()) * rng.uniform(3, 8, idx.sum()) * noise
        # 测点冻结：随机起点，持续 8~25 步
        n_freeze = rng.poisson(freeze_prob * n_steps / 10)
        for _ in range(n_freeze):
            st = rng.integers(0, n_steps - 30)
            ln = rng.integers(8, 25)
            y[st:st + ln] = y[st]
        return y

    steam_flow = add_sensor_noise(steam_flow_true, 1.6, "steam_flow")
    hot_air_flow = add_sensor_noise(hot_air_flow_true, 30.0, "hot_air_flow")
    inlet_moisture = add_sensor_noise(inlet_moisture_true, 0.06, "inlet_moisture")
    inlet_temp = inlet_temp_true + rng.normal(0, 0.2, n_steps)
    feed_rate = add_sensor_noise(feed_rate_true, 12.0, "feed_rate")
    process_gas_temp = process_gas_temp_true + rng.normal(0, 0.5, n_steps)
    return_gas_temp = return_gas_temp_true + rng.normal(0, 0.4, n_steps)
    gas_pressure = gas_pressure_true + rng.normal(0, 0.3, n_steps)

    # 出口仪表：滤波 + 噪声（报告 5.1 的测量链路延迟）
    def instrument_filter(x, tau_s=3.0):
        """一阶滤波。注意 NaN 传播：
        非生产段 x 为 NaN，若直接 y[k-1] + a*(nan-y) 会把整条序列污染成 NaN，
        所以这里跳过无效点、保持上一有效输出，并在首个有效值处初始化。
        """
        a = dt / (tau_s + dt)
        y = np.full(n_steps, np.nan)
        first = True
        for k in range(n_steps):
            if not np.isfinite(x[k]):
                # 保持上一有效值；若尚未初始化则保持 NaN
                if k > 0:
                    y[k] = y[k - 1]
                continue
            if first:
                y[k] = x[k]
                first = False
            else:
                prev = y[k - 1]
                if not np.isfinite(prev):
                    y[k] = x[k]
                else:
                    y[k] = prev + a * (x[k] - prev)
        return y

    outlet_moisture = instrument_filter(outlet_moisture_true, 2.5) + rng.normal(0, 0.035, n_steps)
    outlet_temp = instrument_filter(outlet_temp_true, 2.0) + rng.normal(0, 0.15, n_steps)
    # 非生产段（真值为 NaN）仪表读到的是静态物料，物理上不可作为标签：
    # 保持 NaN，不凭空补值
    outlet_moisture = np.where(non_prod, np.nan, outlet_moisture)
    outlet_temp = np.where(non_prod, np.nan, outlet_temp)

    if include_faults:
        # 通讯中断：两段，标记质量码为 3
        for frac, ln in ((0.22, 25), (0.63, 18)):
            st = int(n_steps * frac)
            outlet_moisture[st:st + ln] = np.nan
            outlet_temp[st:st + ln] = np.nan
            quality_flag[st:st + ln] = 3
        # 出口水分仪冻结（质量码 1）
        st = int(n_steps * 0.45)
        outlet_moisture[st:st + 12] = outlet_moisture[st]
        quality_flag[st:st + 12] = 1
        # 尖峰（质量码 2）
        for frac in (0.31, 0.78):
            st = int(n_steps * frac)
            outlet_moisture[st] += 0.6
            quality_flag[st] = 2
        # 降级：仪器标定漂移段（质量码 4）
        st = int(n_steps * 0.55)
        outlet_moisture[st:st + 40] += 0.08
        quality_flag[st:st + 40] = 4

    # ================================================================
    # 8. 组装
    # ================================================================
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-11-01", periods=n_steps, freq=f"{int(dt)}s" if dt >= 1 else f"{int(dt*1000)}ms"),
        "step": np.arange(n_steps),
        "machine_state": [STATE_CODES[s] for s in state],
        "machine_state_cn": [STATE_CN[s] for s in state],
        "steam_flow": steam_flow,
        "steam_flow_sp": steam_sp,
        "steam_flow_true": steam_flow_true,
        "hot_air_flow": hot_air_flow,
        "hot_air_flow_sp": hot_air_sp,
        "hot_air_flow_true": hot_air_flow_true,
        "inlet_moisture": inlet_moisture,
        "inlet_moisture_true": inlet_moisture_true,
        "inlet_temp": inlet_temp,
        "feed_rate": feed_rate,
        "feed_rate_true": feed_rate_true,
        "process_gas_temp": process_gas_temp,
        "return_gas_temp": return_gas_temp,
        "gas_pressure": gas_pressure,
        "outlet_moisture": outlet_moisture,
        "outlet_moisture_true": outlet_moisture_true,
        "outlet_temp": outlet_temp,
        "outlet_temp_true": outlet_temp_true,
        "quality_flag": quality_flag,
        "delay_steam_true": theta_steam if variable_delay else np.full(n_steps, base_theta_steam),
    })

    # 真实"信息"列（非观测）
    df.attrs["true_delay_steam"] = (base_theta_steam, float(np.mean(theta_steam)))
    df.attrs["true_delay_air"] = (base_theta_air, float(np.mean(theta_air)))

    # ================================================================
    # 9. 降采样（报告 5.1）
    # ================================================================
    if target_freq and target_freq > dt:
        df = downsample(df, target_freq)
    return df


def downsample(df: pd.DataFrame, freq_s: float) -> pd.DataFrame:
    """按时间降采样。数值列取均值，状态列取众数，质量码取最差。"""
    f = f"{int(freq_s)}s"
    d = df.set_index("timestamp")
    num_cols = [c for c in d.columns if d[c].dtype.kind in "fiu" and c not in ("machine_state", "quality_flag", "step")]
    agg = {c: "mean" for c in num_cols}
    agg["machine_state"] = lambda s: int(pd.Series(s).mode().iloc[0]) if len(s) else 0
    agg["quality_flag"] = "max"
    out = d.resample(f).agg(agg).dropna(how="all").reset_index()
    out.attrs = dict(df.attrs)
    out["step"] = np.arange(len(out))
    out["machine_state_cn"] = [STATE_CN.get(_code_to_name(v), "") for v in out["machine_state"]]
    return out


def _code_to_name(code: int) -> str:
    for k, v in STATE_CODES.items():
        if v == code:
            return k
    return "stable"


def stable_mask(df: pd.DataFrame) -> pd.Series:
    """报告 2.3：第一版稳态模型只使用稳定生产段。"""
    return df["machine_state"] == STATE_CODES["stable"]


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """报告 5.4：数据质量审查统计。"""
    n = len(df)
    rows = []
    rows.append({"项目": "总样本数", "数值": n, "占比": "100%"})
    qt = df["quality_flag"]
    for code, name in [(0, "正常"), (1, "测点冻结"), (2, "尖峰"), (3, "通讯中断"), (4, "降级/漂移")]:
        c = int((qt == code).sum())
        rows.append({"项目": f"质量码 {code} ({name})", "数值": c, "占比": f"{c/n*100:.2f}%"})
    # 缺失率
    for c in ["outlet_moisture", "outlet_temp", "steam_flow", "inlet_moisture"]:
        miss = int(df[c].isna().sum())
        rows.append({"项目": f"{c} 缺失", "数值": miss, "占比": f"{miss/n*100:.2f}%"})
    # 冻结段检测（连续相同值）
    frozen = 0
    for c in ["outlet_moisture", "steam_flow"]:
        v = df[c].values
        same = np.r_[False, v[1:] == v[:-1]]
        frozen += int(same.sum())
    rows.append({"项目": "疑似冻结点数(出口水分+蒸汽)", "数值": frozen, "占比": f"{frozen/n*100:.2f}%"})
    # 状态分布
    for s in ["stable", "preheat", "idle", "feed_stop", "shutdown"]:
        c = int((df["machine_state"] == STATE_CODES[s]).sum())
        if c:
            rows.append({"项目": f"工况：{STATE_CN[s]}", "数值": c, "占比": f"{c/n*100:.1f}%"})
    return pd.DataFrame(rows)


if __name__ == "__main__":

    print("=" * 74)
    print("仿真器演示")
    print("=" * 74)
    for kw in [dict(seed=42), dict(seed=42, variable_delay=False),
               dict(seed=42, include_faults=False)]:
        df = simulate_hdt_full(n_steps=1800, **kw)
        tag = " | ".join(f"{k}={v}" for k, v in kw.items())
        print(f"\n[{tag}]  形状={df.shape}")
        print(f"  出口水分 均值={df['outlet_moisture'].mean():.3f} "
              f"std={df['outlet_moisture'].std():.3f} "
              f"缺失={df['outlet_moisture'].isna().sum()}")
        print(f"  真实蒸汽延迟 均值={df.attrs['true_delay_steam'][1]:.2f}s")
    ds = simulate_hdt_full(n_steps=3600, target_freq=5.0)
    print(f"\n降采样到 5s -> {ds.shape}  (原 3600 行 -> {len(ds)} 行)")
    print("\n数据质量审查：")
    print(data_quality_summary(simulate_hdt_full(n_steps=3600)).to_string(index=False))