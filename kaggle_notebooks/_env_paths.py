# 环境与路径配置 - Kaggle Notebook 专用

import os
import sys

def get_dataset_root():
    """Kaggle 数据集根目录"""
    return "/kaggle/input/mimic-cxr-dataset/official_data_iccv_final"

def get_csv_paths():
    """Kaggle CSV 路径"""
    root = get_dataset_root()
    candidates = [
        "/kaggle/input/mimic-cxr-dataset/mimic_eval_single_image_final_233.csv",
        "/kaggle/input/mimic-cxr-dataset/official_data_iccv_final/mimic_eval_single_image_final_233.csv",
        "/kaggle/input/mimic-eval-233/mimic_eval_single_image_final_233.csv",
        "/kaggle/working/mimic_eval_single_image_final_233.csv",
        "./mimic_eval_single_image_final_233.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p, root
    return candidates[0], root

# Ground Truth 列名：必须是人类报告，不能是 prompt
GROUND_TRUTH_COL = "Ground_Truth"  # mimic_eval_single_image_final 格式
TEXT_COL = "text"                  # mimic_eval_ready_step1 / mimic_eval_cleaned 格式

# 标准 prompt（与 Colab 可跑通版本一致）
PROMPT_TEMPLATE = (
    "You are an expert radiologist. Describe this {view} view chest X-ray. "
    "Provide a concise report consisting of Findings and Impression. "
    "Focus on the heart, lungs, mediastinum, pleural space, and bones. "
    "Do NOT use bullet points, asterisks, or section headers. "
    "Do NOT include disclaimers or 'AI' warnings. "
    "Output pure medical text only."
)
