# Lab 01：从 logits 采样 next token

文件 `sampler_lab.py` 将所有“候选集过滤”和“随机抽样”拆开，使每个步骤可单独检验。与原教程一致，函数处理一维 `[vocab_size]` logits，返回一个标量 token id。

## TODO 清单

| TODO | 函数 | 提示 |
|---|---|---|
| 1 | `greedy_sample` | `argmax`，完全确定性 |
| 2 | `temperature_probabilities` | `softmax(logits / T)`，T 必须大于 0 |
| 3 | `temperature_sample` | 从上面的概率分布 `multinomial` 抽 1 个 |
| 4 | `filter_top_k` | 保留**恰好** top-k 的 logits，其余 `-inf` |
| 5 | `top_k_sample` | 过滤后调用 temperature sample |
| 6 | `filter_top_p` | 排序、累积概率、保留首次达到 p 的 token |
| 7 | `top_p_sample` | 过滤后调用 temperature sample |
| 8 | `gumbel_max_sample` | `argmax(logits/T + gumbel_noise)` |

## Top-p 的边界规则

例如概率已排序为 `[0.60, 0.30, 0.10]`：

| p | 保留 token | 原因 |
|---|---|---|
| 0.60 | 第 1 个 | 第 1 个已经使累积概率达到 0.60 |
| 0.85 | 前 2 个 | 第 2 个使累积概率从 0.60 达到 0.90 |
| 1.00 | 全部 | 不截断 |

先决定“保留哪些 token”，再从保留后的分布采样。未保留项用 `-inf`，因为 `softmax(-inf)=0`。

## 运行

```powershell
cd C:\Users\coley\Desktop\mini-vllm\phase2_采样算法\lab01_sampling_primitives
$env:PYTHONIOENCODING='utf-8'; python test_sampler_lab.py
```
