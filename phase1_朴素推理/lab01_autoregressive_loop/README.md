# Lab 01：手写朴素自回归生成循环

原教程的 `step05_naive/engine.py` 只有很少代码，但它决定了推理系统最重要的控制流。本 lab 使用一个可记录调用历史的 toy model，因此测试关注的是**你的引擎是否按正确的序列长度调用模型**，不依赖随机模型或计时结果。

## 你要填的 TODO

文件：`naive_engine_lab.py`

| TODO | 函数 | 要证明的概念 |
|---|---|---|
| 1 | `attention_score_elements` | 长度为 `n` 的单头 Attention score 矩阵有 `n²` 个元素 |
| 2 | `total_naive_attention_work` | 连续生成时，每步完整重算，工作量为 `Σ(p+i)²` |
| 3 | `decode_one_step` | `model(input_ids)` → `logits[-1]` → `argmax` |
| 4 | `generate` | clone prompt，循环解码，使用 `torch.cat` 追加标量 token |

## 先在纸上推一遍

若 `prompt_ids=[3, 4]`，模型规则是“下一个 token 等于最后一个 token + 1”，生成 3 个 token：

```text
调用 1：model([3, 4])       -> 5  -> [3, 4, 5]
调用 2：model([3, 4, 5])    -> 6  -> [3, 4, 5, 6]
调用 3：model([3, 4, 5, 6]) -> 7  -> [3, 4, 5, 6, 7]
```

模型接收的长度依次为 `2, 3, 4`。若生成 3 个 token，Attention score 元素量为 `2² + 3² + 4² = 29`。注意：这不是运行时间的精确模型，但很好地刻画了“历史越长、重算越多”的趋势。

## 运行

```powershell
cd C:\Users\coley\Desktop\mini-vllm\phase1_朴素推理\lab01_autoregressive_loop
$env:PYTHONIOENCODING='utf-8'; python test_naive_engine_lab.py
```

全部通过后再跑上一级的 capstone。
