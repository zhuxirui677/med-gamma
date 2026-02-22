# Kaggle 运行说明

## Add Input

1. 添加 `mimic-cxr-dataset`（含 official_data_iccv_final/files/ 图片）
2. `mimic_eval_single_image_final_233.csv` 可：
   - 放入 mimic-cxr-dataset 根目录
   - 或单独建数据集上传，Add Input 添加

## 路径（Kaggle 自动）

- 数据集: `/kaggle/input/mimic-cxr-dataset/official_data_iccv_final`
- CSV: `/kaggle/input/mimic-cxr-dataset/mimic_eval_single_image_final_233.csv` 等

## Ground_Truth

- 必须用人类报告（`text` 或 `Ground_Truth` 列），绝不能把 prompt 当 Ground_Truth
