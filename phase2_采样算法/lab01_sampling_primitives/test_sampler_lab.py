"""不依赖 pytest 的 Lab 01 自测。"""

import torch

from sampler_lab import (
    filter_top_k,
    filter_top_p,
    greedy_sample,
    gumbel_max_sample,
    temperature_probabilities,
    temperature_sample,
    top_k_sample,
    top_p_sample,
)


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


def raises_value_error(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def test_greedy_and_temperature() -> None:
    logits = torch.tensor([-1.0, 0.2, 4.0, 1.0])
    check("greedy 选择最大 logit", greedy_sample(logits).item() == 2)
    check("greedy 返回标量", greedy_sample(logits).shape == ())
    low = temperature_probabilities(logits, 0.4)
    high = temperature_probabilities(logits, 2.0)
    check("temperature probabilities 和为 1", torch.allclose(low.sum(), torch.tensor(1.0)))
    check("低温使最大概率更集中", low.max() > high.max())
    torch.manual_seed(7)
    check("temperature_sample 返回词表内标量", 0 <= temperature_sample(logits, 1.0).item() < len(logits))
    check("temperature=0 被拒绝", raises_value_error(lambda: temperature_sample(logits, 0.0)))


def test_top_k() -> None:
    logits = torch.tensor([0.0, 5.0, 2.0, 4.0, 1.0])
    filtered = filter_top_k(logits, 2)
    kept = torch.isfinite(filtered).nonzero().flatten().tolist()
    check("top-k 只保留分数最高的两个 token", kept == [1, 3], f"got={kept}")
    check("top-k 不修改输入", torch.equal(logits, torch.tensor([0.0, 5.0, 2.0, 4.0, 1.0])))
    torch.manual_seed(0)
    draws = {top_k_sample(logits, 2).item() for _ in range(100)}
    check("top-k sample 永不从候选集外抽样", draws.issubset({1, 3}), f"got={draws}")
    check("k 越界被拒绝", raises_value_error(lambda: filter_top_k(logits, 0)))


def test_top_p() -> None:
    logits = torch.log(torch.tensor([0.60, 0.30, 0.10]))
    kept_60 = torch.isfinite(filter_top_p(logits, 0.60)).nonzero().flatten().tolist()
    kept_85 = torch.isfinite(filter_top_p(logits, 0.85)).nonzero().flatten().tolist()
    kept_100 = torch.isfinite(filter_top_p(logits, 1.0)).nonzero().flatten().tolist()
    check("top-p 保留首次达到 p 的边界 token", kept_60 == [0], f"got={kept_60}")
    check("top-p=0.85 保留前两个 token", kept_85 == [0, 1], f"got={kept_85}")
    check("top-p=1 保留全部 token", kept_100 == [0, 1, 2])
    torch.manual_seed(1)
    draws = {top_p_sample(logits, 0.60).item() for _ in range(100)}
    check("top-p sample 永不从候选集外抽样", draws == {0}, f"got={draws}")
    check("非法 p 被拒绝", raises_value_error(lambda: filter_top_p(logits, 0.0)))


def test_gumbel_distribution() -> None:
    logits = torch.tensor([1.0, 0.0, -1.0])
    expected = temperature_probabilities(logits, 1.0)
    torch.manual_seed(123)
    draws = torch.tensor([gumbel_max_sample(logits).item() for _ in range(2000)])
    observed = torch.bincount(draws, minlength=3).float() / len(draws)
    check("Gumbel-Max 返回标量", gumbel_max_sample(logits).shape == ())
    check(
        "Gumbel-Max 的经验分布接近 softmax 分布",
        torch.allclose(observed, expected, atol=0.05),
        f"observed={observed.tolist()}, expected={expected.tolist()}",
    )


def main() -> None:
    print("=" * 60)
    print("Phase 2 / Lab 01: 采样原语自测")
    print("=" * 60)
    test_greedy_and_temperature()
    test_top_k()
    test_top_p()
    test_gumbel_distribution()
    print("=" * 60)
    print(f"通过 {passed} / {total}")
    print("✅ Lab 01 全部通过，可以进入 lab02_sampling_engine" if passed == total else "❌ 请检查 TODO")


if __name__ == "__main__":
    main()
