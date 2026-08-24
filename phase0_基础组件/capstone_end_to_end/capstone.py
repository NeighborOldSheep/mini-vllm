"""
Phase 0 Capstone: 串联 lab01(Tokenizer) → lab02(Embedding) → lab03(Attention)
                  → lab04(TransformerDecoderLayer/TinyTransformer)

没有新的 TODO。前置：lab01~lab04 的 test_*.py 必须全部先通过。
运行: python capstone.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab01_tokenizer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab04_transformer"))

from tokenizer_lab import SimpleBPETokenizer  # noqa: E402
from transformer_lab import TinyTransformer   # noqa: E402


def main():
    torch.manual_seed(42)

    # 1. Tokenizer：文本 → token id（对应 lab01）
    tok = SimpleBPETokenizer()
    text = "你好，世界！Hello!"
    token_ids = tok.encode(text)
    print(f"输入文本: {text!r}")
    print(f"Tokenizer 词表大小: {tok.vocab_size}")
    print(f"Token IDs: {token_ids}")
    print()

    # 2. TinyTransformer：token id → 向量(lab02) → 注意力(lab03) → logits(lab04)
    #    vocab_size 必须和 tokenizer 训练出的词表大小一致，否则 Embedding 查表会越界
    d_model, num_heads, num_layers = 32, 4, 2
    model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    print(
        f"TinyTransformer 配置: vocab_size={tok.vocab_size}, "
        f"d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}"
    )

    input_ids = torch.tensor(token_ids)
    logits = model(input_ids)
    print(f"输入 shape: {list(input_ids.shape)}  →  输出 logits shape: {list(logits.shape)}")

    expected_shape = (len(token_ids), tok.vocab_size)
    assert tuple(logits.shape) == expected_shape, (
        f"logits shape {tuple(logits.shape)} != 期望 {expected_shape}"
    )

    top1 = logits.argmax(dim=-1)
    print(f"\n每个位置 top-1 预测 token id: {top1.tolist()}")
    print(
        "（注意：TinyTransformer 权重是随机初始化的，预测结果没有语义，\n"
        "  这里只验证数据能不能在四层组件之间正确流动，形状对不对）"
    )

    print("\n✅ capstone 跑通，Phase 0 全部完成")


if __name__ == "__main__":
    main()
