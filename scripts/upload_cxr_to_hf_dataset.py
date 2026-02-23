#!/usr/bin/env python3
"""
在 Kaggle/Colab 中运行，将 233 张 CXR 图片上传到 Hugging Face Dataset。
运行前：pip install huggingface_hub pandas
环境变量：HF_TOKEN（你的 HF Write token）

用法：
  export HF_TOKEN='hf_xxx'
  python upload_cxr_to_hf_dataset.py
"""

import os
import csv
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("请设置 HF_TOKEN 环境变量")

# 配置
CSV_PATH = "mimic_eval_single_image_final_233.csv"
DATASET_REPO = "cxr-233-images"  # 或 "你的用户名/cxr-233-images"
DATASET_ROOT = "/kaggle/input/mimic-cxr-dataset"  # Kaggle 路径
# Colab 若用 kagglehub: dataset_path = kagglehub.dataset_download("simhadrisadaram/mimic-cxr-dataset")

# 备选路径（kagglehub 缓存）
ALT_ROOTS = [
    "/kaggle/input/mimic-cxr-dataset",
    "/root/.cache/kagglehub/datasets/simhadrisadaram/mimic-cxr-dataset/versions/2/official_data_iccv_final",
]

def find_csv():
    candidates = [
        CSV_PATH,
        "mimic_eval_single_image_final_233.csv",
        "/kaggle/working/mimic_eval_single_image_final_233.csv",
        "/kaggle/input/clean-data/mimic_eval_single_image_final_233.csv",
        "../mimic_eval_single_image_final_233.csv",
    ]
    try:
        candidates.append(str(Path(__file__).resolve().parent.parent / "mimic_eval_single_image_final_233.csv"))
    except Exception:
        pass
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("未找到 mimic_eval_single_image_final_233.csv，请放在当前目录或 /kaggle/working/")

def find_image_path(raw_path: str) -> str | None:
    """从 CSV 的 raw_path 解析出实际可访问的本地路径"""
    raw = raw_path.strip()
    if os.path.exists(raw):
        return raw
    # 尝试不同根路径
    candidates = [
        raw,
        raw.replace("/root/.cache/kagglehub/datasets/simhadrisadaram/mimic-cxr-dataset/versions/2/official_data_iccv_final", "/kaggle/input/mimic-cxr-dataset/official_data_iccv_final"),
        os.path.join(DATASET_ROOT, raw.split("mimic-cxr-dataset/")[-1]) if "mimic-cxr-dataset" in raw else raw,
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def main():
    csv_path = find_csv()
    api = HfApi(token=HF_TOKEN)
    info = api.whoami()
    username = info["name"]
    repo_id = f"{username}/{DATASET_REPO}"
    create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True, token=HF_TOKEN)
    print(f"✅ Dataset: https://huggingface.co/datasets/{repo_id}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for i, row in enumerate(reader):
            raw_path = row.get("Image_Path", row.get("image_path", ""))
            local_path = find_image_path(raw_path)
            if local_path and os.path.exists(local_path):
                fname = f"{row['subject_id']}_{i}.jpg"
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=fname,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=HF_TOKEN,
                )
                url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{fname}"
                row["Image_Path"] = url  # 用 URL 替换原路径，前端可直接用
                print(f"  [{i+1}/233] {fname}")
            else:
                row["Image_Path"] = ""  # 找不到则留空
                print(f"  [{i+1}/233] ⚠️ 未找到: {raw_path[:60]}...")
            rows.append(row)

    out_csv = "reports-data-with-urls.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n✅ 已生成: {out_csv}")
    print(f"👉 复制到 medgamma-frontend/lib/reports-data.csv 并更新前端")

if __name__ == "__main__":
    main()
