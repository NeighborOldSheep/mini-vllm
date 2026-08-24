"""
Lab 03 自测脚本。完成 attention_lab.py 里的全部 TODO 后运行：
    python test_attention_lab.py
"""

import torch
from attention_lab import scaled_dot_product_attention, MultiHeadAttention

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"[PASS] {name}")
    else:
        _failed += 1
        print(f"[FAIL] {name}  {detail}")


def test_shapes_and_softmax():
    torch.manual_seed(0)
    seq_len, d_head = 5, 4
    Q = torch.randn(seq_len, d_head)
    K = torch.randn(seq_len, d_head)
    V = torch.randn(seq_len, d_head)

    output, weights = scaled_dot_product_attention(Q, K, V, causal=True)
    check("output shape 正确", output.shape == (seq_len, d_head), f"got {output.shape}")
    check("weights shape 正确", weights.shape == (seq_len, seq_len), f"got {weights.shape}")

    row_sums = weights.sum(dim=-1)
    check(
        "softmax 归一化：每行权重和 ≈ 1",
        torch.allclose(row_sums, torch.ones(seq_len), atol=1e-5),
        f"got {row_sums.tolist()}",
    )


def test_causal_mask():
    torch.manual_seed(1)
    seq_len, d_head = 4, 4
    Q = torch.randn(seq_len, d_head)
    K = torch.randn(seq_len, d_head)
    V = torch.randn(seq_len, d_head)

    _, weights_causal = scaled_dot_product_attention(Q, K, V, causal=True)
    future_is_zero = all(
        weights_causal[i, j].item() < 1e-6
        for i in range(seq_len)
        for j in range(i + 1, seq_len)
    )
    check("causal=True 时未来位置权重为 0", future_is_zero)

    _, weights_noncausal = scaled_dot_product_attention(Q, K, V, causal=False)
    any_future_nonzero = any(
        weights_noncausal[i, j].item() > 1e-6
        for i in range(seq_len)
        for j in range(i + 1, seq_len)
    )
    check("causal=False 时未来位置权重不再强制为 0", any_future_nonzero)


def test_multihead_shape():
    torch.manual_seed(42)
    seq_len, d_model, num_heads = 6, 16, 4
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(seq_len, d_model)
    out = mha(x)
    check("MultiHeadAttention 输出 shape 正确", out.shape == (seq_len, d_model), f"got {out.shape}")


def test_single_head_consistency():
    """num_heads=1 时，MultiHeadAttention 内部应退化为对整个 d_model 做一次单头注意力。"""
    torch.manual_seed(7)
    seq_len, d_model = 5, 8
    mha = MultiHeadAttention(d_model=d_model, num_heads=1)
    x = torch.randn(seq_len, d_model)

    Q = mha.W_q(x)
    K = mha.W_k(x)
    V = mha.W_v(x)
    manual_out, _ = scaled_dot_product_attention(Q, K, V, causal=True)
    manual_out = mha.W_o(manual_out)

    mha_out = mha(x)
    check(
        "num_heads=1 时与手动单头计算结果一致",
        torch.allclose(mha_out, manual_out, atol=1e-5),
        f"max diff = {(mha_out - manual_out).abs().max().item()}",
    )


def main():
    print("=" * 60)
    print("Lab 03: Attention 自测")
    print("=" * 60)
    test_shapes_and_softmax()
    test_causal_mask()
    test_multihead_shape()
    test_single_head_consistency()
    print("=" * 60)
    print(f"通过 {_passed} / {_passed + _failed}")
    if _failed == 0:
        print("✅ Lab 03 全部通过，可以进入 lab04_transformer")
    else:
        print("❌ 还有未通过的测试，检查 attention_lab.py 里对应的 TODO")


if __name__ == "__main__":
    main()
