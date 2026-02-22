# Kaggle Notebooks：MedGemma 图像到报告 + F1 RadGraph

三个 Notebook 用于在 Kaggle 上跑 MedGemma 图像到报告生成与 RadGraph F1 评估，并对比原始、W4A4、W4A8 的分数与 GPU 占用。

- **Kaggle**：Add Input 添加 `mimic-cxr-dataset`，CSV 可放入该数据集或单独建数据集上传
- **环境**：torch 2.6, transformers 4.51, accelerate, bitsandbytes
- **P100 GPU**：支持，16GB 显存，自动用 FP16

## 输入要求

1. **mimic-cxr-dataset**：Kaggle 数据集，包含 `official_data_iccv_final/files/` 下胸片图片
2. **mimic_eval_single_image_final_233.csv**：233 条评估样本，列包括 `Image_Path`、`Ground_Truth`、`Generated_Report`
   - 若 CSV 不在 mimic-cxr-dataset 中，可单独创建数据集上传，并修改各 Notebook 中的 `CSV_PATH`

## 运行顺序

| 顺序 | Notebook | 说明 |
|------|----------|------|
| 1 | 01_medgemma_original_w4a16_f1_radgraph.ipynb | 原始 MedGemma (FP16)，基线 |
| 2 | 02_medgemma_w4a4_f1_radgraph.ipynb | W4A4：4-bit 权重 + 4-bit 激活 |
| 3 | 03_medgemma_w4a8_f1_radgraph.ipynb | W4A8：4-bit 权重 + 8-bit 激活 |
| 4 | 04_medgemma_distillation_233.ipynb | 蒸馏：Teacher=原模型，Student=QLoRA，233 清理数据 |

**重要**：每次跑完一个 Notebook 后，需 `del model` + `gc.collect()` + `torch.cuda.empty_cache()` 再跑下一个，才能正确对比 GPU 占用。

## 输出

- `original_scores.json`：原始模型 RG_E、RG_ER、RG_ER_bar 与 GPU 占用
- `w4a4_results.csv` / `w4a8_results.csv`：带生成报告的 CSV
- 各 Notebook 末尾会打印与原始模型的对比表

## 蒸馏脚本（本地/Kaggle 通用）

`scripts/distill_medgemma_233.py`：在 233 清理数据上做 Teacher-Student 蒸馏。Teacher 目标默认用 CSV 中 `Generated_Report`（因 233 由原模型筛选）；可选 `--no_csv_teacher` 用原模型重新生成。

```bash
python scripts/distill_medgemma_233.py --csv mimic_eval_single_image_final_233.csv --output ./medgemma_distill_lora
```

## 演算逻辑

详见 `W4A8_W4A4_LOGIC.md`。
