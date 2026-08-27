# Phase 2 Coding Lab — 采样算法

> 对应原教程 `mini-vllm-tutorial/step06_sampler`。Phase 1 解决了“怎样反复调用模型”，这里解决“拿到最后一个位置 logits 后，怎样选择 next token”。

## 目标

你将实现五种策略：Greedy、Temperature、Top-k、Top-p（Nucleus）和 Gumbel-Max，并把它们接到 Phase 1 同样的自回归循环中。

```text
last logits
  -> (可选) / temperature
  -> (可选) 截断候选集为 Top-k 或 Top-p
  -> multinomial，或 logits + Gumbel noise 后 argmax
  -> next token
```

采样策略影响输出多样性，**不会**解决 Phase 1 的重算问题；KV Cache 仍是下一阶段的重点。

## 目录和顺序

| 顺序 | 目录 | 对应教程 | 你将掌握 | 验收命令 |
|---|---|---|---|---|
| 1 | `lab01_sampling_primitives` | step06 `sampler.py` | 分布变换、top-k/top-p 过滤、Gumbel-Max | `python lab01_sampling_primitives/test_sampler_lab.py` |
| 2 | `lab02_sampling_engine` | step06 `engine.py` | 采样参数分派与带策略的生成循环 | `python lab02_sampling_engine/test_sampling_engine_lab.py` |
| 3 | `capstone_sampling_comparison` | step06 `run.py` 的扩展 | 同一 logits 下比较各种策略的输出 | `python capstone_sampling_comparison/compare_strategies.py` |

## Windows 命令

```powershell
cd C:\Users\coley\Desktop\mini-vllm\phase2_采样算法
$env:PYTHONIOENCODING='utf-8'; python lab01_sampling_primitives/test_sampler_lab.py
$env:PYTHONIOENCODING='utf-8'; python lab02_sampling_engine/test_sampling_engine_lab.py
$env:PYTHONIOENCODING='utf-8'; python capstone_sampling_comparison/compare_strategies.py
```

## 自查清单

- [ ] logits 和概率的区别是什么？为什么采样前要 softmax？
- [ ] 为什么 `temperature=0` 不能直接传给 `logits / temperature`，而应分派为 greedy？
- [ ] `T < 1`、`T > 1` 分别怎样改变概率分布的熵？
- [ ] Top-k 为什么应该在 logits 空间将未保留项设为 `-inf`？
- [ ] Top-p 如何做到候选数随分布尖锐程度自适应？为什么必须保留“首次达到 p 的边界 token”？
- [ ] Gumbel-Max 为什么可以不显式调用 softmax 和 multinomial？
- [ ] 这五种策略中，哪些会被随机种子影响？

卡住时对照 `mini-vllm-tutorial/step06_sampler/sampler.py` 和 `engine.py`，理解后再独立完成。下一步是 `step07_kvcache_for_single_request`。
