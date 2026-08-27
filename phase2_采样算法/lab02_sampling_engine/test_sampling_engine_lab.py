"""不依赖 pytest 的 Lab 02 自测。"""

import torch
import torch.nn as nn

import sampling_engine_lab as engine_module
from sampling_engine_lab import SamplingEngine


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


class FixedLogitsModel(nn.Module):
    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.call_lengths: list[int] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.call_lengths.append(len(input_ids))
        logits = torch.zeros((len(input_ids), self.vocab_size))
        logits[-1, 6] = 10.0
        return logits


def test_dispatch_priority() -> None:
    originals = {
        name: getattr(engine_module, name)
        for name in ("greedy_sample", "gumbel_max_sample", "top_k_sample", "top_p_sample", "temperature_sample")
    }
    calls: list[tuple[str, tuple]] = []

    def marker(name: str, value: int):
        def fn(*args):
            calls.append((name, args[1:]))
            return torch.tensor(value)
        return fn

    try:
        engine_module.greedy_sample = marker("greedy", 1)
        engine_module.gumbel_max_sample = marker("gumbel", 2)
        engine_module.top_k_sample = marker("top_k", 3)
        engine_module.top_p_sample = marker("top_p", 4)
        engine_module.temperature_sample = marker("temperature", 5)
        engine = SamplingEngine(FixedLogitsModel())
        logits = torch.tensor([0.0, 1.0])
        cases = [
            ("temperature=0 优先 greedy", dict(temperature=0, top_k=2, top_p=0.5, use_gumbel=True), 1, "greedy"),
            ("gumbel 优先于 top-k/top-p", dict(temperature=1, top_k=2, top_p=0.5, use_gumbel=True), 2, "gumbel"),
            ("top-k 优先于 top-p", dict(temperature=1, top_k=2, top_p=0.5), 3, "top_k"),
            ("top-p 在没有 top-k 时生效", dict(temperature=1, top_k=0, top_p=0.5), 4, "top_p"),
            ("默认使用 temperature", dict(temperature=0.8), 5, "temperature"),
        ]
        for label, kwargs, expected_id, expected_call in cases:
            calls.clear()
            result = engine._select_next_token(logits, **kwargs)
            check(label, result.item() == expected_id and calls[0][0] == expected_call, f"calls={calls}")
    finally:
        for name, fn in originals.items():
            setattr(engine_module, name, fn)


def test_generate() -> None:
    model = FixedLogitsModel()
    engine = SamplingEngine(model)
    prompt = torch.tensor([1, 2])
    output = engine.generate(prompt, 3, temperature=0)
    check("generate 保留原 prompt", prompt.tolist() == [1, 2])
    check("greedy 生成 3 个最高 logit token", output.tolist() == [1, 2, 6, 6, 6])
    check("每步对完整历史执行前向", model.call_lengths == [2, 3, 4], f"got={model.call_lengths}")


def main() -> None:
    print("=" * 60)
    print("Phase 2 / Lab 02: SamplingEngine 自测")
    print("=" * 60)
    test_dispatch_priority()
    test_generate()
    print("=" * 60)
    print(f"通过 {passed} / {total}")
    print("✅ Lab 02 全部通过，可以运行 capstone_sampling_comparison" if passed == total else "❌ 请检查 TODO")


if __name__ == "__main__":
    main()
