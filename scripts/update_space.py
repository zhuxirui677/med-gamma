#!/usr/bin/env python3
"""
将 medgamma 网页的 app.py 推送到 Hugging Face Space，触发重新部署。

运行前：pip install huggingface_hub
环境变量：HF_TOKEN（你的 HF Write token，需有 Space 写入权限）

用法：
  export HF_TOKEN='hf_xxx'
  python scripts/update_space.py
"""

import os
from pathlib import Path

from huggingface_hub import HfApi

# 优先用 HF_TOKEN 环境变量，否则使用 huggingface-cli login 保存的 token
HF_TOKEN = os.environ.get("HF_TOKEN") or None

SPACE_REPO = "Maxsine2025/medical-image-analysis"
FILES_DIR = Path(__file__).resolve().parent.parent / "medgamma网页" / "files"
APP_PATH = FILES_DIR / "app.py"
README_PATH = FILES_DIR / "README.md"

if not APP_PATH.exists():
    raise SystemExit(f"未找到 app.py: {APP_PATH}")

def main():
    api = HfApi(token=HF_TOKEN)  # None 时使用 huggingface-cli login 的 token
    print(f"📤 上传文件到 Space: {SPACE_REPO}")

    api.upload_file(
        path_or_fileobj=str(APP_PATH),
        path_in_repo="app.py",
        repo_id=SPACE_REPO,
        repo_type="space",
        token=HF_TOKEN,
        commit_message="fix: 添加推荐入口链接，优先使用 .hf.space 直连",
    )

    if README_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(README_PATH),
            path_in_repo="README.md",
            repo_id=SPACE_REPO,
            repo_type="space",
            token=HF_TOKEN,
            commit_message="docs: README 添加推荐入口",
        )

    print("✅ 上传完成！Space 将自动重新部署。")
    print(f"👉 推荐入口: https://maxsine2025-medical-image-analysis.hf.space/")

if __name__ == "__main__":
    main()
