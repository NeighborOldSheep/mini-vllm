"""
Lab 04 自测脚本。完成 transformer_lab.py 里的全部 TODO 后运行：
    python test_transformer_lab.py

前置：确保 lab02_embedding / lab03_attention 的自测已经全部通过。
"""

import torch
from transformer_lab import RMSNorm, MLP, TransformerDecoderLayer, TinyTransformer

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


def test_rmsnorm():
    torch.manual_seed(0)
    d_model = 16
    norm = RMSNorm(d_model)  # weight 初始化为全 1

    x = torch.randn(5, d_model) * 10  # 放大数值，确保不是碰巧接近1
    out = norm(x)
    rms_out = out.pow(2).mean(-1).sqrt()
    check(
        "weight=全1 时，输出的 RMS ≈ 1",
        torch.allclose(rms_out, torch.ones(5), atol=1e-3),
        f"got {rms_out.tolist()}",
    )

    # 手算验证: x = [3, 4], RMS(x) = sqrt((9+16)/2) = sqrt(12.5)
    norm2 = RMSNorm(2)
    x2 = torch.tensor([[3.0, 4.0]])
    expected_rms = (12.5) ** 0.5
    expected = x2 / expected_rms
    out2 = norm2(x2)
    check(
        "RMSNorm 手算数值一致",
        torch.allclose(out2, expected, atol=1e-4),
        f"got {out2.tolist()}, expected {expected.tolist()}",
    )


def test_mlp_shape():
    torch.manual_seed(0)
    d_model, d_ff = 16, 64
    mlp = MLP(d_model, d_ff)
    x = torch.randn(5, d_model)
    out = mlp(x)
    check("MLP 输出 shape 与输入一致", out.shape == (5, d_model), f"got {out.shape}")


def test_decoder_layer_shape():
    torch.manual_seed(0)
    seq_len, d_model, num_heads, d_ff = 6, 16, 4, 64
    layer = TransformerDecoderLayer(d_model, num_heads, d_ff)
    x = torch.randn(seq_len, d_model)
    out = layer(x)
    check("DecoderLayer 输出 shape 与输入一致", out.shape == x.shape, f"got {out.shape}")


def test_residual_connection():
    """把 attn/mlp 输出强制置零，此时 layer(x) 应该恰好等于 x（验证残差加回原始 x）。"""
    torch.manual_seed(0)
    seq_len, d_model, num_heads, d_ff = 4, 8, 2, 32
    layer = TransformerDecoderLayer(d_model, num_heads, d_ff)

    layer.attn.forward = lambda t: torch.zeros_like(t)
    layer.mlp.forward = lambda t: torch.zeros_like(t)

    x = torch.randn(seq_len, d_model)
    out = layer(x)
    check(
        "attn/mlp 输出为 0 时，残差应使 layer(x) == x",
        torch.allclose(out, x, atol=1e-6),
        f"max diff = {(out - x).abs().max().item()}",
    )


def test_tiny_transformer_forward_and_causality():
    torch.manual_seed(42)
    vocab_size, d_model, num_heads, num_layers = 64, 32, 4, 2
    model = TinyTransformer(vocab_size, d_model, num_heads, num_layers)

    seq_len = 6
    token_ids = torch.randint(0, vocab_size, (seq_len,))
    logits = model(token_ids)
    check(
        "TinyTransformer 输出 shape 为 [seq_len, vocab_size]",
        logits.shape == (seq_len, vocab_size),
        f"got {logits.shape}",
    )

    token_ids2 = token_ids.clone()
    token_ids2[-1] = (token_ids[-1] + 1) % vocab_size
    logits2 = model(token_ids2)
    check(
        "因果性（多层堆叠后依然成立）：修改最后一个 token 不影响之前位置的 logits",
        torch.allclose(logits[:-1], logits2[:-1], atol=1e-5),
        f"max diff = {(logits[:-1] - logits2[:-1]).abs().max().item()}",
    )


def main():
    print("=" * 60)
    print("Lab 04: Transformer 自测")
    print("=" * 60)
    test_rmsnorm()
    test_mlp_shape()
    test_decoder_layer_shape()
    test_residual_connection()
    test_tiny_transformer_forward_and_causality()
    print("=" * 60)
    print(f"通过 {_passed} / {_passed + _failed}")
    if _failed == 0:
        print("✅ Lab 04 全部通过，Phase 0 核心 Lab 完成！去跑 capstone_end_to_end")
    else:
        print("❌ 还有未通过的测试，检查 transformer_lab.py 里对应的 TODO")
        print("   （如果错误来自 embedding_lab / attention_lab，回去检查 lab02 / lab03）")


if __name__ == "__main__":
    main()
