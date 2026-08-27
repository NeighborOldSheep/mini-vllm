# Capstone：重算成本账本

这个小程序不追求不稳定的 wall-clock benchmark，而是记录朴素引擎每次真正送入模型的序列长度，并计算 Attention score 元素数。

它还给出一个**理想单请求 KV Cache** 的对照估计：prompt 只 prefill 一次，之后每步只计算一个新 Query 对全部历史 K/V 的注意力。它不是 KV Cache 实现；真正实现留给 step07。

```powershell
cd C:\Users\coley\Desktop\mini-vllm\phase1_朴素推理\capstone_recompute_profiler
$env:PYTHONIOENCODING='utf-8'; python recompute_profile.py
```
