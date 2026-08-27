"""Phase 1 capstone：用确定性模型观察朴素重算。"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

LAB_DIR = Path(__file__).resolve().parents[1] / "lab01_autoregressive_loop"
sys.path.insert(0, str(LAB_DIR))
from naive_engine_lab import NaiveEngine, attention_score_elements, total_naive_attention_work


class CountingNextTokenModel(nn.Module):
    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.lengths: list[int] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.lengths.append(input_ids.numel())
        logits = torch.full((input_ids.numel(), self.vocab_size), -100.0)
        logits[-1, (input_ids[-1].item() + 1) % self.vocab_size] = 100.0
        return logits


def ideal_kv_cache_work(prompt_len: int, max_new_tokens: int) -> int:
    """prompt prefill 一次，之后仅为第 2..m 个生成 token 计算新 Query。"""
    if max_new_tokens == 0:
        return 0
    return prompt_len**2 + sum(prompt_len + i for i in range(1, max_new_tokens))


def main() -> None:
    prompt = torch.tensor([2, 3, 4])
    new_tokens = 5
    model = CountingNextTokenModel()
    output = NaiveEngine(model).generate(prompt, new_tokens)

    print(f"prompt: {prompt.tolist()}\n生成: {new_tokens} token\n输出: {output.tolist()}\n")
    print("step | 输入长度 | 本步 Attention score 元素数")
    print("-----|----------|---------------------------")
    for step, length in enumerate(model.lengths, start=1):
        print(f"{step:>4} | {length:>8} | {attention_score_elements(length):>25}")

    naive = total_naive_attention_work(len(prompt), new_tokens)
    cached = ideal_kv_cache_work(len(prompt), new_tokens)
    print(f"\n朴素完整重算估计: {naive} 个 score 元素")
    print(f"理想 KV Cache 估计: {cached} 个 score 元素")
    print(f"差异倍数: {naive / cached:.2f}x")
    print("\n✅ Phase 1 capstone 跑通；step07 将把这个估计变成真正的 KV Cache。")


if __name__ == "__main__":
    main()
