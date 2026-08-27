# Lab 02：把采样策略接入生成引擎

原教程 `step06_sampler/engine.py` 的核心是分派顺序。这个 lab 显式提供 `_select_next_token`，测试通过替身函数确认你选中了正确策略，而不是只碰巧生成出了一个合法 token。

## TODO

1. `_select_next_token`：按下列优先级选择策略。
2. `generate`：重复“完整历史前向 → last logits → `_select_next_token` → append”。

优先级与原教程保持一致：

```text
temperature == 0  -> greedy
use_gumbel         -> gumbel-max
top_k > 0          -> top-k
top_p < 1          -> top-p
otherwise          -> temperature sample
```

这意味着同时设置 `top_k` 和 `top_p` 时，此教学引擎优先 `top_k`。生产引擎也可选择允许二者叠加，但必须明确其语义。

```powershell
cd C:\Users\coley\Desktop\mini-vllm\phase2_采样算法\lab02_sampling_engine
$env:PYTHONIOENCODING='utf-8'; python test_sampling_engine_lab.py
```
