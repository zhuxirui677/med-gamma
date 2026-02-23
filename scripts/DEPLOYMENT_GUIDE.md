# MedGemma 部署操作指南

## 推荐入口

**优先使用**：https://maxsine2025-medical-image-analysis.hf.space/

（huggingface.co/spaces/... 页面有时加载慢，此直连地址更稳定）

---

## 一、推送 app.py 到 HF Space

```bash
cd /Users/zhuxirui/Desktop/medgamma

# 若未登录，先执行：huggingface-cli login
python3 scripts/update_space.py
```

或手动设置 token：
```bash
export HF_TOKEN='hf_你的token'
python3 scripts/update_space.py
```

---

## 二、推送前端到 Vercel

```bash
cd /Users/zhuxirui/Desktop/medgamma

git add medgamma-frontend/ medgamma网页/
git commit -m "你的提交信息"
git push origin main
```

若 Vercel 已关联 GitHub，推送后会自动部署。否则在 Vercel 控制台手动 Redeploy。

---

## 三、Vercel 环境变量与套餐要求

| 变量 | 值 | 说明 |
|------|-----|------|
| HF_GRADIO_SPACE | Maxsine2025/medical-image-analysis | 必需 |
| HF_TOKEN | hf_xxx | 可选，Space 需认证时 |

**⚠️ AI 分析超时**：MedGemma 推理约需 2–3 分钟。Vercel Hobby 套餐函数仅 10 秒，会导致超时返回 Demo。需 **Pro 或 Team** 套餐（60s–300s）才能获得真实 AI 分析。

---

## 四、HF Space 环境变量（Settings → Variables and secrets）

| 变量 | 值 |
|------|-----|
| HF_TOKEN | 你的 HF token |
| ADAPTER_REPO | 你的 LoRA 适配器仓库（如 Maxsine2025/xxx） |

---

## 五、修改文件位置

| 文件 | 用途 |
|------|------|
| medgamma网页/files/app.py | HF Space 主程序 |
| medgamma网页/files/README.md | HF Space 说明 |
| medgamma-frontend/ | 前端（Vercel 部署） |
