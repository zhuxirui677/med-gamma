#!/usr/bin/env python3
"""
MedGemma 蒸馏：233 清理数据上 Teacher-Student

逻辑：
- Teacher：原始 MedGemma，233 条由原模型筛选，CSV 中 Generated_Report 即 teacher 输出
- Student：QLoRA（4-bit + LoRA），学习模仿 teacher 的生成序列
- 蒸馏目标：teacher 序列作为 target，CE 损失逐 token 拟合
"""

import os
import sys
import gc
import argparse
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# 环境检查（不强制 Python 版本，以加速兼容为准）
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16


def get_teacher_targets(df: pd.DataFrame, model_id: str, use_csv: bool = True) -> list:
    """获取 teacher 目标：use_csv=True 直接用 CSV 中 Generated_Report，否则用原模型生成"""
    if use_csv and df["Generated_Report"].notna().all() and (df["Generated_Report"].str.len() > 10).all():
        return df["Generated_Report"].fillna("").tolist()

    from transformers import AutoProcessor, AutoModelForImageTextToText

    print("加载 Teacher 生成目标...")
    teacher = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=DTYPE, device_map="auto")
    proc = AutoProcessor.from_pretrained(model_id)

    def gen(img_path: str) -> str:
        if not os.path.exists(img_path):
            return ""
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return ""
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe this chest X-ray in a radiology report format."}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(teacher.device, dtype=DTYPE)
        L = inp["input_ids"].shape[-1]
        with torch.inference_mode():
            out = teacher.generate(**inp, max_new_tokens=512, do_sample=False)
        return proc.decode(out[0][L:], skip_special_tokens=True).strip()

    targets = [gen(row["Image_Path"]) for _, row in tqdm(df.iterrows(), total=len(df))]
    del teacher, proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return targets


def run_distill(
    csv_path: str,
    output_dir: str = "./medgemma_distill_lora",
    use_csv_teacher: bool = True,
    num_samples: int = 233,
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 2e-5,
):
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    df = pd.read_csv(csv_path)
    df = df.head(num_samples)

    # Teacher 目标
    teacher_targets = get_teacher_targets(df, "google/medgemma-1.5-4b-it", use_csv=use_csv_teacher)
    df["teacher_target"] = teacher_targets

    # 过滤有效样本
    train_df = df[df["teacher_target"].str.len() > 20].reset_index(drop=True)
    print(f"有效蒸馏样本: {len(train_df)}")

    proc = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Student: QLoRA
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE, bnb_4bit_quant_type="nf4")
    model = AutoModelForImageTextToText.from_pretrained("google/medgemma-1.5-4b-it", quantization_config=bnb, device_map="auto")
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for ep in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for i in tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {ep+1}"):
            batch_df = train_df.iloc[i : i + batch_size]
            opt.zero_grad()
            for _, row in batch_df.iterrows():
                path = row["Image_Path"]
                target = str(row["teacher_target"] or "")
                if not os.path.exists(path) or len(target) < 10:
                    continue
                img = Image.open(path).convert("RGB")
                msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe this chest X-ray in a radiology report format."}]}]
                inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
                target_ids = proc(text=target, return_tensors="pt", truncation=True, max_length=400)["input_ids"]
                prompt_len = inp["input_ids"].shape[-1]
                full_ids = torch.cat([inp["input_ids"], target_ids], dim=1)
                labels = full_ids.clone()
                labels[0, :prompt_len] = -100  # 不计算 prompt 部分 loss

                model_inputs = {"input_ids": full_ids[:, :-1].to(model.device), "labels": labels[:, 1:].to(model.device)}
                for k, v in inp.items():
                    if k in ("input_ids", "labels"):
                        continue
                    if hasattr(v, "to"):
                        t = v.to(model.device)
                        if "pixel" in k.lower():
                            t = t.to(dtype=DTYPE)
                        model_inputs[k] = t
                out = model(**model_inputs)

                loss = out.loss / batch_size
                loss.backward()
                total_loss += loss.item()
            opt.step()
            n_batches += 1
        print(f"Epoch {ep+1} avg loss: {total_loss / max(n_batches, 1):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    proc.save_pretrained(output_dir)
    print(f"已保存至 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MedGemma 蒸馏（233 清理数据）")
    parser.add_argument("--csv", default="mimic_eval_single_image_final_233.csv", help="233 样本 CSV")
    parser.add_argument("--output", default="./medgemma_distill_lora", help="输出目录")
    parser.add_argument("--no_csv_teacher", action="store_true", help="不用 CSV 中 Generated_Report，改用原模型生成")
    parser.add_argument("--num_samples", type=int, default=233)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    run_distill(
        csv_path=args.csv,
        output_dir=args.output,
        use_csv_teacher=not args.no_csv_teacher,
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
