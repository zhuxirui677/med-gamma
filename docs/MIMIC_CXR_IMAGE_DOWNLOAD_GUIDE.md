# MIMIC-CXR 图片下载与 233 样本 F1 评估指南

mimic_eval_single_image_final_233.csv 中每行对应一张胸片，需要图片才能让 MedGemma 等视觉模型生成报告，再与 Ground_Truth 做 RadGraph F1 评估。

---

## 一、是否需要下载图片

需要。要做「图像到报告」的生成和 F1 评估，必须：

1. 用模型根据图片生成报告
2. 用 RadGraph 比较生成报告与 Ground_Truth 的 F1

没有图片时，无法做真正的图像到报告评估。若只用文本模型（如 Mistral AWQ），只能做「文本提示到报告」，不能反映图像理解能力。

---

## 二、CSV 中的路径含义

mimic_eval_single_image_final_233.csv 的 Image_Path 示例：

```
/kaggle/input/mimic-cxr-dataset/official_data_iccv_final/files/p10/p10075925/s51010496/2d783c8a-492984b7-28aaf571-bfc30156-61ab26f6.jpg
```

有效部分为：`files/p10/p10075925/s51010496/2d783c8a-492984b7-28aaf571-bfc30156-61ab26f6.jpg`，对应 MIMIC-CXR-JPG 的目录结构。

---

## 三、下载方式

### 方式一：PhysioNet（需申请权限）

MIMIC-CXR 在 PhysioNet 上，需完成认证才能下载。

1. 注册 PhysioNet：https://physionet.org/register/
2. 完成 CITI 培训（Human Research、HIPAA 等）
3. 申请 MIMIC-CXR-JPG：https://physionet.org/content/mimic-cxr-jpg/2.0.0/
4. 审批通过后，在项目页获取下载命令，例如：

```bash
# 安装 wget 或使用 rsync
# 下载整个 MIMIC-CXR-JPG（约 50GB+）
wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/
```

或使用 rsync（通常更快）：

```bash
rsync -avz --progress YOUR_USERNAME@physionet.org:/files/mimic-cxr-jpg/2.0.0/ ./mimic-cxr-jpg/
```

下载后目录结构类似：

```
mimic-cxr-jpg/
  files/
    p10/
      p10075925/
        s51010496/
          2d783c8a-492984b7-28aaf571-bfc30156-61ab26f6.jpg
          ...
```

### 方式二：Kaggle（若已有 Kaggle 账号）

若数据来自 Kaggle 的 MIMIC-CXR 相关数据集：

1. 登录 Kaggle：https://www.kaggle.com/
2. 搜索 "mimic-cxr" 或 "mimic-cxr-jpg"
3. 下载对应数据集
4. 解压到本地，例如 `./mimic-cxr-dataset/`
5. 确认存在 `official_data_iccv_final/files/` 或等效的 `files/` 结构

### 方式三：仅下载 233 张图（需脚本）

若已能访问完整 MIMIC-CXR-JPG，可只复制 233 张图到本地，减少占用。示例脚本：

```python
import pandas as pd
import os
import shutil

df = pd.read_csv("mimic_eval_single_image_final_233.csv")
mimic_base = "/path/to/mimic-cxr-jpg"  # 你下载的 MIMIC-CXR-JPG 根目录
output_dir = "./mimic_eval_233_images"
os.makedirs(output_dir, exist_ok=True)

for _, row in df.iterrows():
    # 从 /kaggle/.../files/p10/... 提取 files 之后的相对路径
    path = row["Image_Path"]
    if "files/" in path:
        rel_path = path.split("files/")[-1]
    else:
        rel_path = path
    src = os.path.join(mimic_base, "files", rel_path)
    dst = os.path.join(output_dir, os.path.basename(rel_path))
    if os.path.exists(src):
        shutil.copy2(src, dst)
```

---

## 四、更新 CSV 中的图片路径

下载完成后，需要把 CSV 里的 Image_Path 改成你本地的路径。

假设图片在 `./mimic_eval_233_images/` 下，且文件名与原来一致，可这样更新：

```python
import pandas as pd

df = pd.read_csv("mimic_eval_single_image_final_233.csv")
base = "./mimic_eval_233_images"

def new_path(old_path):
    fname = os.path.basename(old_path)
    return os.path.join(base, fname)

df["Image_Path"] = df["Image_Path"].apply(new_path)
df.to_csv("mimic_eval_single_image_final_233_local.csv", index=False)
```

若你按 MIMIC-CXR-JPG 原始结构下载，可保留 `files/p10/...` 结构，只改前缀：

```python
df["Image_Path"] = df["Image_Path"].str.replace(
    "/kaggle/input/mimic-cxr-dataset/official_data_iccv_final",
    "/path/to/your/mimic-cxr-jpg"
)
```

---

## 五、跑完 233 样本并做 F1 评估

当前 `evaluate_awq_model.py` 和 `test_medgamma_clean.py` 主要面向「文本列」评估。要做「图像到报告」的 233 样本 F1 评估，需要：

1. 支持按 Image_Path 加载图片
2. 将图片送入 MedGemma 等视觉模型生成报告
3. 用 RadGraph 比较生成报告与 Ground_Truth

可新增脚本 `scripts/evaluate_medgemma_233.py`，逻辑大致为：

- 读取 `mimic_eval_single_image_final_233.csv`
- 对每行：加载 `Image_Path` 对应图片，调用 MedGemma 生成报告
- 收集所有生成报告，与 `Ground_Truth` 一起送入 RadGraph 计算 F1

脚本示例见项目中的 scripts/evaluate_medgemma_233_images.py（若已创建）。

---

## 六、仅做 F1 评估（已有生成报告时）

若 CSV 中已有 Generated_Report 列，且你只想算 F1、不重新生成，可使用项目内置脚本（逻辑参考 [Redgraph-F1score-calculator](https://github.com/sx2660-png/Redgraph-F1score-calculator)）：

```bash
python scripts/evaluate_f1_radgraph_csv.py -i mimic_eval_single_image_final_233.csv -o results_with_scores.csv
```

输出：带 RG_E、RG_ER、RG_ER_bar 的 CSV，以及 `*_summary.json` 汇总。此时不需要图片。

若需手动调用 RadGraph：

```python
import pandas as pd
from radgraph import F1RadGraph

df = pd.read_csv("mimic_eval_single_image_final_233.csv")
hyps = df["Generated_Report"].fillna("").tolist()
refs = df["Ground_Truth"].fillna("").tolist()

f1radgraph = F1RadGraph(reward_level="all", model_type="modern-radgraph-xl")
mean_reward, reward_list, _, _ = f1radgraph(hyps=hyps, refs=refs)
rg_e, rg_er, rg_bar_er = mean_reward
print(f"RG_E: {rg_e:.4f}, RG_ER: {rg_er:.4f}, RG_ER_bar: {rg_bar_er:.4f}")
```

若要用自己的模型重新生成报告，则必须下载图片。

---

## 七、总结

| 步骤 | 说明 |
|------|------|
| 1. 获取 MIMIC-CXR 权限 | PhysioNet 注册、培训、申请 MIMIC-CXR-JPG |
| 2. 下载图片 | 用 wget/rsync 或 Kaggle 下载 |
| 3. 更新 CSV 路径 | 将 Image_Path 改为本地路径 |
| 4. 运行图像到报告 | 用 MedGemma 等模型对 233 张图生成报告 |
| 5. F1 评估 | 用 RadGraph 比较生成报告与 Ground_Truth |

若暂时无法下载 MIMIC-CXR，可考虑使用 CheXpert、OpenI 等公开胸片数据集做替代评估，但需重新整理 CSV 和路径。
