"""不依赖 pytest 的 Lab 01 自测。"""

import torch
import torch.nn as nn

from naive_engine_lab import (
    NaiveEngine,
    attention_score_elements,
    total_naive_attention_work,
)


class CountingNextTokenModel(nn.Module):
    """确定性 toy model：预测 (最后 token + 1) % vocab_size，并记录每次输入。"""

    def __init__(self, vocab_size: int = 10):
        super().__init__()
        self.vocab_size = vocab_size
        self.call_inputs: list[torch.Tensor] = []
        self.grad_enabled_during_call: list[bool] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.call_inputs.append(input_ids.detach().clone())
        self.grad_enabled_during_call.append(torch.is_grad_enabled())
        logits = torch.full((input_ids.numel(), self.vocab_size), -100.0)
        next_id = (input_ids[-1].item() + 1) % self.vocab_size
        logits[-1, next_id] = 100.0
        return logits


passed = 0
total = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, total
    total += 1
    if condition:
        passed += 1
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}  {detail}")


def test_work_accounting() -> None:
    check("n=0 的 score 元素数为 0", attention_score_elements(0) == 0)
    check("n=7 的 score 元素数为 49", attention_score_elements(7) == 49)
    check(
        "prompt=2、生成3步的重算工作量为 2²+3²+4²=29",
        total_naive_attention_work(2, 3) == 29,
    )
    check("不生成时总工作量为 0", total_naive_attention_work(9, 0) == 0)


def test_decode_and_generate() -> None:
    model = CountingNextTokenModel()
    engine = NaiveEngine(model)
    prompt = torch.tensor([3, 4])

    next_id = engine.decode_one_step(prompt)
    check("decode_one_step 返回标量", next_id.shape == ())
    check("decode_one_step 取最后 logits 的 argmax", next_id.item() == 5)
    check("decode_one_step 在 no_grad 下运行", model.grad_enabled_during_call == [False])

    model.call_inputs.clear()
    model.grad_enabled_during_call.clear()
    original_prompt = prompt.clone()
    result = engine.generate(prompt, max_new_tokens=3)
    check("generate 保留原 prompt 不变", torch.equal(prompt, original_prompt))
    check("generate 返回 prompt + 3 个 token", result.tolist() == [3, 4, 5, 6, 7])
    check(
        "每步都将完整且递增的历史送入模型",
        [x.tolist() for x in model.call_inputs] == [[3, 4], [3, 4, 5], [3, 4, 5, 6]],
    )


def main() -> None:
    print("=" * 60)
    print("Phase 1 / Lab 01: 朴素自回归引擎自测")
    print("=" * 60)
    test_work_accounting()
    test_decode_and_generate()
    print("=" * 60)
    print(f"通过 {passed} / {total}")
    print("✅ Lab 01 全部通过，可以运行 capstone_recompute_profiler" if passed == total else "❌ 请检查 TODO")


if __name__ == "__main__":
    main()
