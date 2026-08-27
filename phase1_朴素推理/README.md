# Phase 1 Coding Lab — 朴素推理

> 对应原教程 `mini-vllm-tutorial/step05_naive`。这一阶段先实现一个**故意低效**的推理引擎；它是后续 KV Cache 优化的性能基线。

## 目标

完成后，你应能独立解释并写出以下生成闭环：

```text
prompt token ids
  -> model(完整历史序列)
  -> 只取最后一个位置的 logits
  -> greedy / argmax 选出 next token
  -> 把 next token 追加回历史序列
  -> 重复
```

关键是：第 `i` 步不是只送入新 token，而是把从 prompt 到当前位置的**完整历史**重新送进模型。Attention 每一步都重新构建 `seq_len × seq_len` 的分数矩阵，这正是重复计算的来源。

## 目录和顺序

| 顺序 | 目录 | 对应教程 | 重点 | 验收命令 |
|---|---|---|---|---|
| 1 | `lab01_autoregressive_loop` | step05 `engine.py` | 完整历史前向、最后 logits、追加 token、重算工作量 | `python lab01_autoregressive_loop/test_naive_engine_lab.py` |
| 2 | `capstone_recompute_profiler` | step05 的性能观察 | 可视化每步输入长度，并比较朴素重算与理想 KV Cache 的工作量估计 | `python capstone_recompute_profiler/recompute_profile.py` |

在 Windows 上若中文输出报 `UnicodeEncodeError`，使用：

```powershell
$env:PYTHONIOENCODING='utf-8'; python lab01_autoregressive_loop/test_naive_engine_lab.py
```

## 学习规则

1. 先读 `lab01_autoregressive_loop/README.md`，再填 `naive_engine_lab.py` 中的 TODO。
2. 测试通过后，运行 capstone；不要用它替代单元测试。
3. 卡住时再对照原教程的 `step05_naive/engine.py`。理解后回到自己的文件重写，不要直接复制。

## 完成后自查

- [ ] 为什么预测 next token 时取的是 `logits[-1]`，不是 `logits[0]`？
- [ ] `generate` 为什么必须 clone prompt，而不是直接向 prompt 追加？
- [ ] prompt 长度为 `p`、生成 `m` 个 token 时，朴素 Attention 分数元素总量为何是 `Σ(p+i)²`？
- [ ] 为什么“每步耗时线性变慢”和“整个生成过程的总工作量超线性增长”可以同时成立？
- [ ] KV Cache 将复用哪一部分中间结果？它不会缓存什么？

完成后继续 [`../phase2_采样算法/README.md`](../phase2_采样算法/README.md)。
