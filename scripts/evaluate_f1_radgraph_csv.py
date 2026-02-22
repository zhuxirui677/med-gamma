#!/usr/bin/env python3
"""
RadGraph F1 分数评估 - CSV 版本

逻辑与方法参考：https://github.com/sx2660-png/Redgraph-F1score-calculator

支持 CSV 格式，要求包含 Ground_Truth 和 Generated_Report 列。
适用于 mimic_eval_single_image_final_233.csv 等数据。
"""

import argparse
import csv
import json
import os
import numpy as np

try:
    from radgraph import F1RadGraph
except ImportError:
    print("请安装 radgraph: pip install radgraph")
    raise


def evaluate_csv(
    input_path: str,
    output_path: str = None,
    ref_column: str = "Ground_Truth",
    hyp_column: str = "Generated_Report",
    batch_size: int = 20,
    model_type: str = "modern-radgraph-xl",
    reward_level: str = "all",
):
    """
    对 CSV 中的报告计算 RadGraph F1 分数。

    参考 Redgraph-F1score-calculator 的 evaluate_csv.py 逻辑。
    """
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return

    print(f"读取 CSV: {input_path}")
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    if ref_column not in data[0] or hyp_column not in data[0]:
        print(f"错误: CSV 需包含 {ref_column} 和 {hyp_column} 列")
        print(f"当前列: {list(data[0].keys())}")
        return

    refs = [row[ref_column] if row.get(ref_column) else "" for row in data]
    hyps = [row[hyp_column] if row.get(hyp_column) else "" for row in data]

    print(f"共 {len(data)} 条数据")
    print(f"模型: {model_type}, 评估级别: {reward_level}")

    f1radgraph = F1RadGraph(
        reward_level=reward_level,
        model_type=model_type,
    )

    all_scores = []
    print("\n开始评估...")
    for i in range(0, len(data), batch_size):
        batch_refs = refs[i : i + batch_size]
        batch_hyps = hyps[i : i + batch_size]
        end_idx = min(i + batch_size, len(data))
        print(f"  处理 {i + 1}-{end_idx} / {len(data)}", end="\r")

        try:
            mean_reward, reward_list, _, _ = f1radgraph(
                hyps=batch_hyps, refs=batch_refs
            )
            for j in range(len(batch_refs)):
                all_scores.append({
                    "rg_e": float(reward_list[0][j]),
                    "rg_er": float(reward_list[1][j]),
                    "rg_er_bar": float(reward_list[2][j]),
                })
        except Exception as e:
            print(f"\n批次错误: {e}")
            for _ in range(len(batch_refs)):
                all_scores.append({"rg_e": 0.0, "rg_er": 0.0, "rg_er_bar": 0.0})

    print("\n评估完成")

    avg_e = np.mean([s["rg_e"] for s in all_scores]) * 100
    avg_er = np.mean([s["rg_er"] for s in all_scores]) * 100
    avg_er_bar = np.mean([s["rg_er_bar"] for s in all_scores]) * 100

    print("\n" + "=" * 50)
    print("RadGraph F1 平均分数 (百分制)")
    print("-" * 50)
    print(f"RG_E (仅实体):           {avg_e:.2f}")
    print(f"RG_ER (实体+关系):       {avg_er:.2f}  <- 论文常用")
    print(f"RG_ER_bar (完整匹配):    {avg_er_bar:.2f}")
    print("=" * 50)

    if output_path:
        fieldnames = list(data[0].keys()) + ["RG_E", "RG_ER", "RG_ER_bar"]
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row, score in zip(data, all_scores):
                row["RG_E"] = f"{score['rg_e']:.4f}"
                row["RG_ER"] = f"{score['rg_er']:.4f}"
                row["RG_ER_bar"] = f"{score['rg_er_bar']:.4f}"
                writer.writerow(row)
        print(f"\n结果已保存: {output_path}")

    summary = {
        "rg_e": avg_e / 100,
        "rg_er": avg_er / 100,
        "rg_er_bar": avg_er_bar / 100,
        "num_samples": len(data),
        "model_type": model_type,
    }
    if output_path:
        summary_path = os.path.splitext(output_path)[0] + "_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"汇总已保存: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="RadGraph F1 评估 (参考 Redgraph-F1score-calculator)"
    )
    parser.add_argument("--input", "-i", required=True, help="输入 CSV 路径")
    parser.add_argument("--output", "-o", help="输出 CSV 路径（含分数）")
    parser.add_argument("--ref-column", default="Ground_Truth", help="参考报告列名")
    parser.add_argument("--hyp-column", default="Generated_Report", help="生成报告列名")
    parser.add_argument("--batch-size", type=int, default=20, help="批大小")
    parser.add_argument("--model", default="modern-radgraph-xl", help="RadGraph 模型类型")

    args = parser.parse_args()
    output = args.output or (os.path.splitext(args.input)[0] + "_with_scores.csv")
    evaluate_csv(
        input_path=args.input,
        output_path=output,
        ref_column=args.ref_column,
        hyp_column=args.hyp_column,
        batch_size=args.batch_size,
        model_type=args.model,
    )


if __name__ == "__main__":
    main()
