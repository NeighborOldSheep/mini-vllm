# Capstone：同一 logits，不同采样策略

此程序使用一个固定 logits 的 toy model，因此输出差异只能来自采样策略。每种随机策略前都会重设种子，方便重复观察。

它依赖已完成的 Lab 01 和 Lab 02：

```powershell
cd C:\Users\coley\Desktop\mini-vllm\phase2_采样算法\capstone_sampling_comparison
$env:PYTHONIOENCODING='utf-8'; python compare_strategies.py
```

观察：greedy 每次都选同一 token；低温通常更集中；top-k/top-p 不会选择过滤掉的 token；Gumbel-Max 也会产生随机选择。
