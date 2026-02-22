#!/usr/bin/env python3
"""
从 mimic_eval_ready_step1.csv 准备评估用 CSV
- Ground_Truth 必须用人类报告 (text 列)，绝不能把 prompt 当 Ground_Truth
- 路径提取逻辑与 Colab 可跑通版本一致
"""

import os
import ast
import pandas as pd

DATASET_ROOT = os.environ.get("DATASET_ROOT", "/kaggle/input/mimic-cxr-dataset/official_data_iccv_final")


def get_single_image_path(cell_value, dataset_root: str) -> str | None:
    """从 PA/AP/Lateral 列提取单张有效图片路径"""
    if pd.isna(cell_value):
        return None
    val_str = str(cell_value).strip()
    target_path = ""

    if val_str.startswith('[') and val_str.endswith(']'):
        try:
            path_list = ast.literal_eval(val_str)
            if path_list:
                target_path = path_list[0] if isinstance(path_list[0], str) else str(path_list[0])
            else:
                return None
        except Exception:
            target_path = val_str.replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",")[0]
    else:
        target_path = val_str

    clean = str(target_path).strip().strip("'").strip('"')
    if "files" in clean:
        clean = "files" + clean.split("files", 1)[1]
    else:
        clean = clean.strip("/")

    full = os.path.join(dataset_root, clean)
    return full if os.path.exists(full) else None


def run(
    csv_path: str,
    output_path: str = "mimic_eval_single_image_for_eval.csv",
    dataset_root: str = None,
):
    dataset_root = dataset_root or DATASET_ROOT
    df = pd.read_csv(csv_path)

    # Ground_Truth 必须来自 text 列（人类报告），不是 prompt
    if "text" not in df.columns:
        raise ValueError("CSV 需包含 text 列作为 Ground_Truth（人类报告）")

    results = []
    for idx, row in df.iterrows():
        final_path = None
        used_view = None

        for col, view in [("PA", "PA"), ("AP", "AP"), ("Lateral", "Lateral")]:
            if col not in df.columns:
                continue
            p = get_single_image_path(row[col], dataset_root)
            if p:
                final_path, used_view = p, view
                break

        if not final_path:
            continue

        # 关键：Ground_Truth = 人类报告 text，绝不是 prompt
        gt = str(row["text"] or "").strip()
        if not gt or gt.startswith("You are an expert"):
            continue  # 防止误把 prompt 当 Ground_Truth

        results.append({
            "subject_id": row["subject_id"],
            "View": used_view,
            "Image_Path": final_path,
            "Ground_Truth": gt,
            "Generated_Report": "",  # 待模型生成后填充
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"已保存 {len(out_df)} 条至 {output_path}")
    return out_df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="mimic_eval_ready_step1.csv")
    p.add_argument("--output", default="mimic_eval_single_image_for_eval.csv")
    p.add_argument("--dataset_root", default=None)
    args = p.parse_args()
    run(args.csv, args.output, args.dataset_root)
