#!/usr/bin/env python3
"""
=== 真实数据脚本 1B：18 个无表头批次 TXT 的表头核对与规范化输出 ===

本脚本不改动 data/raw/real/batches/*.txt 原始文件，而是基于与 1A.csv / 1B.csv 的
数值分布、料头爬升动态、执行器滞后关系与 30s 控制周期核对结果，为全部 18 个批次
补齐规范英文表头与阶段标注（startup/steady/tail），输出到：
    data/processed/real/batches_named/<batch_id>.csv
并生成字段核对字典：
    outputs_real/reports/r1b_batch_header_mapping.csv

运行：
    cd ~/hdt_dryer_mpc
    /opt/conda_envs/dryer-ts/bin/python src_real/r1b_add_batch_headers.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from real_common import REAL_RAW, REAL_PROC, REAL_REP, segment_batch_phases

# 16 列核对结果（严格区分：高置信度物理列 / 待点表确认辅助列 / 30s控制衍生列[建模禁用]）
BATCH_16_COLUMNS = [
    "feed_rate",               # col 0 : 称瞬时流量 (kg/h) [已确认]
    "inlet_moisture",          # col 1 : 入口含水率 (%) [已确认]
    "casing_flow",             # col 2 : 加料瞬时流量 (kg/h) [已确认: ~201.6/230.2, 对应4.5%~5.1%加料比例]
    "steam_flow",              # col 3 : 补偿蒸汽流量 (kg/h) [已确认]
    "steam_valve_fb",          # col 4 : 补偿蒸汽阀门开度反馈 (%) [已确认: 精确滞后 col11 2秒]
    "outlet_temp",             # col 5 : 烟叶出口温度 (℃) [已确认]
    "water_flow",              # col 6 : 加水流量 (kg/h) [已确认: 全18批次均为0]
    "aux_col7_const79",        # col 7 : 疑似热风/筒体恒定工艺量 (~79.18) [待点表确认]
    "outlet_moisture",         # col 8 : 出口含水率 (%) [已确认]
    "inlet_or_env_temp_est",   # col 9 : 疑似入口烟叶/环境温度 (~31.4℃, 0.1℃分辨率) [待点表确认]
    "return_air_temp",         # col 10: 回风温度PV (℃) [已确认]
    "steam_valve_sp",          # col 11: 补偿蒸汽阀门开度给定/指令 (%) [已确认: 超前 col4 精确2秒, r=0.99915]
    "ctrl_stat_col12_30s",     # col 12: 30s周期更新的控制器统计量1 [衍生量, 建模禁用]
    "ctrl_stat_col13_30s",     # col 13: 30s周期更新的控制器统计量2 (含9966毛刺) [衍生量, 建模禁用]
    "moist_err_30s",           # col 14: 30s周期更新的出口水分偏差指标 [含目标泄露, 建模禁用]
    "temp_err_30s",            # col 15: 30s周期更新的出口温度偏差 (col5+col15≈50.00℃) [含目标泄露, 建模禁用]
]

HEADER_DICT_ROWS = [
    (0,  "feed_rate",             "称瞬时流量",             "kg/h", "DV (扰动量)",   "高 (已核对)",   "稳态~4500 kg/h，与1A/1B一致"),
    (1,  "inlet_moisture",        "入口含水率",             "%",    "DV (扰动量)",   "高 (已核对)",   "稳态~16.34%，料头爬升曲线与1A/1B完全一致"),
    (2,  "casing_flow",           "加料瞬时流量",           "kg/h", "MV/配比量",     "高 (已核对)",   "稳态~201.6/230.2 kg/h (占4500的4.5%~5.1%)，与1A加料累积量积分吻合"),
    (3,  "steam_flow",            "补偿蒸汽流量",           "kg/h", "MV (操纵量)",   "高 (已核对)",   "稳态~70.2 kg/h，料头峰值与1A/1B一致"),
    (4,  "steam_valve_fb",        "补偿蒸汽阀门开度反馈",   "%",    "MV 反馈",       "高 (已核对)",   "稳态~27.6%，启动段精确等于 col11 延迟2秒"),
    (5,  "outlet_temp",           "烟叶出口温度",           "℃",    "CV (被控目标)", "高 (已核对)",   "稳态~49.98℃，且 col5+col15 均值精确等于 50.00℃"),
    (6,  "water_flow",            "加水流量",               "kg/h", "MV (未启用)",   "高 (已核对)",   "全部18批次恒为0，与1A/1B一致"),
    (7,  "aux_col7_const79",      "待确认工艺量(约79.2)",   "未知", "辅助状态",      "中 (待确认)",   "全过程窄幅波动78.9~79.3，几乎恒定，建议向提供者确认点表"),
    (8,  "outlet_moisture",       "出口含水率",             "%",    "CV (被控目标)", "高 (已核对)",   "稳态~20.74%，与1A/1B一致"),
    (9,  "inlet_or_env_temp_est", "疑似入口/环境温度",      "℃",    "DV/环境量",     "中 (待确认)",   "30.1~32.3℃，0.1℃分辨率，与col10回风温度相关系数-0.94"),
    (10, "return_air_temp",       "回风温度PV",             "℃",    "过程状态",      "高 (已核对)",   "稳态~53.4℃，与1A/1B一致"),
    (11, "steam_valve_sp",        "补偿蒸汽阀门开度给定",   "%",    "MV 控制指令",   "高 (已核对)",   "稳态与col4相关系数0.99915，启动段精确超前col4两秒"),
    (12, "ctrl_stat_col12_30s",   "30s控制器统计量1",       "衍生", "禁用(防泄露)",  "高 (衍生确认)", "t=268起每30s阶梯更新一次，含724尖峰，不可作预测输入"),
    (13, "ctrl_stat_col13_30s",   "30s控制器统计量2",       "衍生", "禁用(防泄露)",  "高 (衍生确认)", "t=268起每30s阶梯更新一次，含9966极端毛刺，不可作预测输入"),
    (14, "moist_err_30s",         "30s出口水分偏差指标",    "%",    "禁用(目标泄露)","高 (衍生确认)", "t=268起每30s阶梯更新一次，与出口水分直接相关，严禁入模"),
    (15, "temp_err_30s",          "30s出口温度偏差指标",    "℃",    "禁用(目标泄露)","高 (衍生确认)", "t=268起每30s阶梯更新一次，col5+col15≈50.00℃，严禁入模"),
]


def main():
    raw_dir = REAL_RAW / "batches"
    out_dir = REAL_PROC / "batches_named"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"未找到原始批次文件: {raw_dir}")

    summary = []
    for p in txt_files:
        batch_id = p.stem
        df = pd.read_csv(p, header=None, names=BATCH_16_COLUMNS)
        df.insert(0, "sample_idx", np.arange(len(df), dtype=int))
        df.insert(0, "batch_id", batch_id)
        df = segment_batch_phases(df)

        out_path = out_dir / f"{batch_id}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        ph = df["phase"].value_counts().to_dict()
        summary.append({
            "batch_id": batch_id,
            "rows": len(df),
            "startup": ph.get("startup", 0),
            "steady": ph.get("steady", 0),
            "tail": ph.get("tail", 0),
            "out_file": str(out_path.relative_to(REAL_PROC.parent.parent)),
        })

    # 保存字段核对字典
    dict_df = pd.DataFrame(
        HEADER_DICT_ROWS,
        columns=["col_idx", "en_name", "cn_name", "unit", "control_role", "confidence", "evidence"],
    )
    dict_path = REAL_REP / "r1b_batch_header_mapping.csv"
    dict_df.to_csv(dict_path, index=False, encoding="utf-8-sig")

    sum_df = pd.DataFrame(summary)
    sum_path = REAL_REP / "r1b_batches_named_summary.csv"
    sum_df.to_csv(sum_path, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print(f"已完成全部 {len(txt_files)} 个批次的表头补齐与阶段标注！")
    print(f"  输出目录: {out_dir}")
    print(f"  字段字典: {dict_path}")
    print(f"  批次汇总: {sum_path}")
    print(f"  总样本数: {sum_df['rows'].sum()} | 稳态总样本: {sum_df['steady'].sum()} ({sum_df['steady'].sum()/sum_df['rows'].sum()*100:.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
