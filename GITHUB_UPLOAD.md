# GitHub 上传指南

## 一、初始化仓库（如未初始化）

```bash
cd /Users/zhuxirui/Desktop/medgamma
git init
```

## 二、添加远程仓库

```bash
git remote add origin https://github.com/你的用户名/medgamma.git
```

## 三、添加文件并提交

```bash
git add .
git status   # 检查要提交的文件
git commit -m "Initial commit: MedGemma 1.5 胸部 X 光报告生成与 RadGraph F1 评估"
```

## 四、推送到 GitHub

```bash
git branch -M main
git push -u origin main
```

## 五、建议保留的文件

- `README.md` - 项目说明
- `requirements.txt` - 依赖
- `.gitignore` - 忽略规则
- `MedGemma_*.ipynb` - Colab notebooks
- `kaggle_notebooks/` - Kaggle notebooks
- `scripts/` - 脚本
- `docs/` - 文档
- `mimic_eval_single_image_final_233.csv` - 233 评估样本（可选，若文件较大可放 .gitignore）

## 六、建议忽略的内容

- `__pycache__/`、`.ipynb_checkpoints/`
- 生成的报告 CSV、checkpoint、模型权重
- `medgamma网页/`、`aww/` 等子项目（若不需要）
