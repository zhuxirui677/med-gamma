# W4A8 与 W4A4 演算逻辑与运用方法

## 一、符号说明

| 符号 | 含义 |
|------|------|
| W4 | 权重 4-bit 量化 |
| A8 | 激活 8-bit 量化 |
| A4 | 激活 4-bit 量化 |
| A16 | 激活 16-bit（全精度，不量化） |

---

## 二、W4 权重量化

### 2.1 AWQ（本项目 Mistral 使用）

- 激活感知权重量化，保护约 1% 显著权重
- Per-channel 缩放：$s^* = \arg\min \|Q(W \cdot s)(s^{-1} \cdot X) - WX\|$
- 量化：$Q(w) = \Delta \cdot \text{Round}(w/\Delta)$，$\Delta = \max(|w|)/(2^{N-1})$，$N=4$

### 2.2 BitsAndBytes NF4（MedGemma 使用，因无官方 AWQ）

- 4-bit Normalized Float，针对权重分布优化
- MedGemma 无 AutoAWQ 支持，用 `load_in_4bit=True` 近似 W4

---

## 三、A8 激活量化（8-bit）

### 3.1 演算逻辑

对每层 Linear 的**输入**（即上一层的激活）做 per-tensor 对称量化：

```
范围: [-128, 127]（有符号 8-bit）
scale = 127 / max(|x|)
q = round(x * scale).clamp(-128, 127)
x' = q / scale   # 反量化，用于后续计算
```

### 3.2 代码实现（参考 PrefixQuant）

```python
def _fake_quant_activation_8bit(x: torch.Tensor) -> torch.Tensor:
    max_val = x.abs().max().clamp(min=1e-8)
    scale = 127.0 / max_val
    q = (x * scale).round().clamp(-128, 127)
    return q / scale
```

### 3.3 运用方法

- **Kaggle Notebook**：用 `forward_pre_hook` 在每层 Linear 前对输入做 fake 量化
- **真实加速**：需 QServe、PrefixQuant 等 INT4×INT8 GEMM kernel，当前 PyTorch hook 仅为精度评估

---

## 四、A4 激活量化（4-bit）

### 4.1 演算逻辑

```
范围: [-8, 7]（有符号 4-bit）
scale = 7 / max(|x|)
q = round(x * scale).clamp(-8, 7)
x' = q / scale
```

### 4.2 代码实现

```python
def _fake_quant_activation_4bit(x: torch.Tensor) -> torch.Tensor:
    max_val = x.abs().max().clamp(min=1e-8)
    scale = 7.0 / max_val
    q = (x * scale).round().clamp(-8, 7)
    return q / scale
```

### 4.3 与 A8 对比

| 比特 | 范围 | 精度 | 显存 | 预期 F1 |
|------|------|------|------|---------|
| A8 | [-128, 127] | 高 | 中 | 接近原始 |
| A4 | [-8, 7] | 低 | 低 | 略降 |

---

## 五、Kaggle 运行顺序

1. **01 原始模型**：加载 MedGemma 全精度 → 生成报告 → F1 评估 → 保存 `original_scores.json` → **del 模型 + gc + empty_cache**
2. **02 W4A4**：清空 GPU → 加载 4-bit 权重 + 4-bit 激活 → 生成 → F1 → 对比原始
3. **03 W4A8**：清空 GPU → 加载 4-bit 权重 + 8-bit 激活 → 生成 → F1 → 对比原始

**重要**：每次只保留一个模型在 GPU，跑完删除再加载下一个，才能准确对比显存占用。

---

## 六、参考

- AWQ: https://arxiv.org/abs/2306.00978
- PrefixQuant: https://github.com/ChenMnZ/PrefixQuant
- Redgraph-F1score-calculator: https://github.com/sx2660-png/Redgraph-F1score-calculator
