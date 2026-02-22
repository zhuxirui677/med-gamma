# MedGemma 1.5 胸部 X 光报告生成与 RadGraph F1 评估

基于 **Google MedGemma 1.5 (4B)** 的胸部 X 光放射学报告生成项目，支持多种量化与蒸馏方法，并在 MIMIC-CXR 233 样本上评估 RadGraph F1 分数。

---

## 📋 目录

- [项目概述](#项目概述)
- [模型方法总览](#模型方法总览)
- [技术原理详解](#技术原理详解)
- [QLoRA 深度解析](#qlora-深度解析)
- [快速开始](#快速开始)
- [Colab Notebooks 使用指南](#colab-notebooks-使用指南)
- [输出文件说明](#输出文件说明)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [参考文献](#参考文献)

---

## 项目概述

本项目实现 **Google MedGemma 1.5 (4B)** 的胸部 X 光图像到放射学报告生成，支持：

- **原始模型**：FP16 全精度推理
- **W4A4 量化**：4-bit 权重 + 4-bit 激活（bitsandbytes）
- **W4A8 量化**：4-bit 权重 + 8-bit 激活（bitsandbytes）
- **知识蒸馏 + QLoRA**：Teacher-Student 蒸馏，Student 使用 4-bit 量化 + LoRA 微调

**评估指标**：RadGraph F1（RG_E、RG_ER、RG_ER_bar）  
**数据集**：MIMIC-CXR 233 samples  
**评估框架**：RadGraph-XL

---

## 模型方法总览

| 方法 | 说明 | 显存 | 精度 | 推理速度 | Notebook |
|------|------|------|------|----------|----------|
| **原始 (FP16)** | 全精度 MedGemma 1.5 | ~8 GB | 最高 | 基准 | `MedGemma_1_5_Clean.ipynb` |
| **W4A4** | 4-bit 权重 + 4-bit 激活 | ~3-4 GB | 略降 | 最快 | `MedGemma_W4A4_Colab.ipynb` |
| **W4A8** | 4-bit 权重 + 8-bit 激活 | ~4-5 GB | 高 | 较快 | `MedGemma_W4A8_Colab.ipynb` |
| **蒸馏 + QLoRA** | Teacher→Student，Student 用 QLoRA | ~5-7 GB（训练） | 接近原始 | 快 | `MedGemma_Distillation_Colab.ipynb` |

---

## 技术原理详解

### 1. W4A4（4-bit 权重 + 4-bit 激活）

**原理**：
- **权重量化**：使用 bitsandbytes NF4（Normalized Float 4-bit），针对权重分布优化
- **激活量化**：4-bit 对称量化，范围 [-8, 7]，per-tensor scale
- **compute_dtype**：`torch.bfloat16`

**配置**：
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

**优势**：显存最低，推理最快  
**劣势**：精度略低于 W4A8

---

### 2. W4A8（4-bit 权重 + 8-bit 激活）

**原理**：
- **权重量化**：同 W4A4，bitsandbytes NF4
- **激活量化**：8-bit 对称量化，范围 [-128, 127]（有符号 8-bit）
- **compute_dtype**：`torch.float16`

**配置**：
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # 8-bit 激活
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=False
)
```

**优势**：精度与显存平衡较好  
**劣势**：比 W4A4 稍慢、显存稍高

---

### 3. 知识蒸馏（Knowledge Distillation） + QLoRA

**原理**：
- **Teacher**：原始 MedGemma 1.5，生成高质量报告
- **Student**：4-bit 量化 + LoRA 微调的 MedGemma（QLoRA）
- **蒸馏目标**：Student 逐 token 拟合 Teacher 的输出序列（使用 Cross-Entropy 损失）
- **训练框架**：peft + trl

**流程**：
1. Teacher 生成 233 条报告（或使用 CSV 中已有的）
2. 初始化 Student（4-bit + LoRA）
3. 蒸馏训练：Student 学习 Teacher 的输出
4. 用训练后的 Student 生成报告并评估

**优势**：Student 模型更小、更快，同时保持较高生成质量  
**劣势**：需要 2-4 小时训练时间

---

## QLoRA 深度解析

### 什么是 QLoRA？

**QLoRA**（Quantized Low-Rank Adaptation）是一种将**量化**与**低秩适配**结合的微调方法，由 Dettmers 等人于 2023 年提出。它允许在**消费级 GPU**上微调大语言模型，只需约 4-bit 显存即可完成训练。

### 核心思想

| 组件 | 说明 |
|------|------|
| **Q**（Quantized） | 将预训练权重冻结并量化为 4-bit（NF4），大幅降低显存 |
| **LoRA**（Low-Rank Adaptation） | 只训练少量低秩矩阵（Adapter），不更新原始权重 |
| **组合** | 推理时：4-bit 权重 + LoRA 增量 = 等效全精度输出 |

### 数学形式

```
原始前向：y = W·x
QLoRA：   y = (Q(W) + ΔW)·x = Q(W)·x + ΔW·x

其中 ΔW = B·A（低秩分解，A∈R^(r×d), B∈R^(d×r)，r<<d）
```

- **Q(W)**：4-bit 量化后的冻结权重
- **ΔW = B·A**：LoRA 可训练参数，秩 r 通常为 8、16、32

### 为什么 QLoRA 重要？

1. **显存友好**：4-bit 量化使 7B 模型仅需 ~4GB 显存，4B 模型约 ~2GB
2. **训练高效**：只训练 0.1–1% 的参数，收敛快、过拟合风险低
3. **精度保持**：通过 Double Quantization 和 NF4 量化，精度损失可控制在 1% 以内
4. **即插即用**：训练后的 LoRA 权重可单独保存（~几十 MB），可随时加载/卸载

### 本项目中 QLoRA 的应用

在蒸馏流程中：

1. **Student 模型**：MedGemma 1.5 以 4-bit 加载（bitsandbytes NF4）
2. **LoRA 配置**：`LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"])`
3. **训练目标**：Student 的 logits 与 Teacher 的 one-hot 标签做 Cross-Entropy 损失
4. **输出**：训练后的 LoRA adapter + 4-bit 基座，推理时合并

### 关键依赖

```bash
pip install bitsandbytes peft trl
```

- **bitsandbytes**：4-bit 量化
- **peft**：LoRA 实现
- **trl**：SFTTrainer 等训练工具

---

## 快速开始

### 环境要求

- **Python**：3.10-3.12（Colab 默认 3.12 可用）
- **GPU**：A100 或 H100（推荐 40GB+）
- **HuggingFace**：需申请 [MedGemma 访问权限](https://huggingface.co/google/medgemma-1.5-4b-it) 并获取 token

### 前置准备

1. **申请 MedGemma 访问**：https://huggingface.co/google/medgemma-1.5-4b-it  
2. **获取 HF Token**：https://huggingface.co/settings/tokens  
3. **准备 CSV**：`mimic_eval_single_image_final_233.csv`（含 `Image_Path`、`Ground_Truth` 列）  
4. **上传到 Google Drive**：将 CSV 放入 `My Drive/medgamma/` 目录

### 一键运行（Colab）

1. 打开对应 Colab Notebook
2. 选择 **Runtime → Change runtime type → A100 GPU**
3. 左侧 **🔑 Secrets** 添加 token（名称：`zhuxirui11` 或 `HF_TOKEN`）
4. **Run All**

---

## Colab Notebooks 使用指南

### 文件位置与说明

| Notebook | 路径 | 用途 |
|----------|------|------|
| **原始版本** | `MedGemma_1_5_Clean.ipynb` | 基线模型，FP16 全精度 |
| **W4A4** | `MedGemma_W4A4_Colab.ipynb` | 4-bit 权重 + 4-bit 激活 |
| **W4A8** | `MedGemma_W4A8_Colab.ipynb` | 4-bit 权重 + 8-bit 激活 |
| **蒸馏 + QLoRA** | `MedGemma_Distillation_Colab.ipynb` | Teacher-Student 蒸馏，Student 用 QLoRA |

### 运行流程（通用）

```
Step 0: 检查 Python 版本
Step 1: 安装依赖
Step 2: 登录 HuggingFace ⚠️ 必需！
Step 3: 挂载 Google Drive
Step 4: 下载 MIMIC-CXR 数据集（kagglehub）
Step 5: 对齐 233 CSV 的图片路径
Step 6: 加载模型
Step 7: 批量生成报告（233 samples）
Step 7.5: 清理模型，释放显存（W4A4/W4A8）
Step 8: RadGraph F1 评估
```

### 蒸馏 + QLoRA Notebook 流程

```
Step 6: Teacher 生成目标报告
Step 7: 初始化 Student（4-bit + LoRA）
Step 8: 蒸馏训练（2-4 小时）← QLoRA 微调
Step 9: Student 生成报告
Step 10: RadGraph F1 评估
```

### 挂载 Google Drive 代码

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 输出文件说明

### 报告 CSV 保存路径

| 方法 | 路径 |
|------|------|
| 原始 | `/content/drive/MyDrive/medgamma/medgemma_reports_233.csv` |
| W4A4 | `/content/drive/MyDrive/medgamma/medgemma_w4a4_reports_233.csv` |
| W4A8 | `/content/drive/MyDrive/medgamma/medgemma_w4a8_reports_233.csv` |
| 蒸馏 | `/content/drive/MyDrive/medgamma/medgemma_distilled_reports_233.csv` |

### RadGraph F1 指标说明

| 指标 | 含义 |
|------|------|
| **RG_E** | Entity F1（实体匹配） |
| **RG_ER** | Entity + Relation F1（实体+关系，论文常用） |
| **RG_ER_bar** | Complete Match F1（完全匹配） |

所有分数以**百分制**显示（如 33.39 表示 33.39%）。

---

## 目录结构

```
medgamma/
├── README.md                    # 本文件
├── requirements.txt             # 依赖
├── .gitignore
├── mimic_eval_single_image_final_233.csv   # 233 评估样本
│
├── MedGemma_1_5_Clean.ipynb     # 原始模型
├── MedGemma_W4A4_Colab.ipynb    # W4A4 量化
├── MedGemma_W4A8_Colab.ipynb    # W4A8 量化
├── MedGemma_Distillation_Colab.ipynb    # 知识蒸馏 + QLoRA
│
├── kaggle_notebooks/            # Kaggle 版本
│   ├── README.md
│   ├── 01_medgemma_original_w4a16_f1_radgraph_v2.ipynb
│   ├── 02_medgemma_w4a4_f1_radgraph_v2.ipynb
│   ├── 03_medgemma_w4a8_f1_radgraph_v2.ipynb
│   ├── 04_medgemma_distillation_233.ipynb
│   └── 04_compare_results_v2.ipynb
│
├── scripts/                     # 脚本
│   ├── distill_medgemma_233.py  # 蒸馏脚本
│   ├── evaluate_f1_radgraph_csv.py
│   └── prepare_eval_from_ready.py
│
└── docs/                        # 文档
    ├── MIMIC_CXR_IMAGE_DOWNLOAD_GUIDE.md
    └── W4A8_W4A4_LOGIC.md（在 kaggle_notebooks/）
```

---

## 环境要求

### 依赖安装

```bash
pip install torch torchvision transformers accelerate bitsandbytes radgraph pillow pandas
```

### 量化方法额外依赖

| 方法 | 额外依赖 |
|------|----------|
| W4A4 / W4A8 | `bitsandbytes` |
| 蒸馏 + QLoRA | `bitsandbytes peft trl` |

---

## 参考文献

- **MedGemma**: [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
- **QLoRA**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **RadGraph**: [RadGraph-XL (ACL 2024)](https://aclanthology.org/2024.findings-acl.765)
- **F1-RadGraph**: [EMNLP 2022](https://aclanthology.org/2022.findings-emnlp.319)
- **RadGraph F1 Calculator**: [sx2660-png/Redgraph-F1score-calculator](https://github.com/sx2660-png/Redgraph-F1score-calculator)

---

## License

本项目遵循 MedGemma 模型许可协议。详见 [Hugging Face](https://huggingface.co/google/medgemma-1.5-4b-it)。
