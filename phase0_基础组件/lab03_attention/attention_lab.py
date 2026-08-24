"""
Lab 03: Scaled Dot-Product Attention + Multi-Head Attention（挖空版）

参考: mini-vllm-tutorial/step03_attention/attention.py
完成所有 TODO 后运行 test_attention_lab.py 自测。
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


def scaled_dot_product_attention(
    Q: Tensor,  # [seq_len, d_head]
    K: Tensor,  # [seq_len, d_head]
    V: Tensor,  # [seq_len, d_head]
    causal: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Scaled Dot-Product Attention（单头版本，教学用）。

    计算流程：
      1. scores = Q·Kᵀ / √d_head      # [seq_len, seq_len]
      2. 应用因果 mask（可选）
      3. weights = softmax(scores)      # [seq_len, seq_len]
      4. output = weights · V           # [seq_len, d_head]

    Returns:
        output:  [seq_len, d_head]
        weights: [seq_len, seq_len]  （注意力权重）
    """
    d_head = Q.size(-1)

    # ------------------------------------------------------------
    # TODO 1: scores = Q·Kᵀ / √d_head
    # ------------------------------------------------------------
    scores = None
    raise NotImplementedError("TODO 1: 计算 scores = Q·Kᵀ / √d_head")

    # ------------------------------------------------------------
    # TODO 2: 应用因果 mask（causal=True 时，token i 不能看到 j>i）
    #   1. mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    #   2. scores = scores.masked_fill(mask, float("-inf"))
    # ------------------------------------------------------------
    if causal:
        raise NotImplementedError("TODO 2: 应用因果 mask")

    # ------------------------------------------------------------
    # TODO 3: softmax 归一化（沿最后一维）
    # ------------------------------------------------------------
    weights = None
    raise NotImplementedError("TODO 3: weights = softmax(scores, dim=-1)")

    # ------------------------------------------------------------
    # TODO 4: 加权求和 output = weights · V
    # ------------------------------------------------------------
    output = None
    raise NotImplementedError("TODO 4: output = weights · V")

    return output, weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力（Multi-Head Attention）。

    将 d_model 维向量切成 num_heads 份（每份 d_head = d_model/num_heads），
    每个"头"在自己的子空间独立做注意力，最后拼回 d_model 维。
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        输入: x  形状 [seq_len, d_model]
        输出:    形状 [seq_len, d_model]
        """
        seq_len, d_model = x.shape

        # ------------------------------------------------------------
        # TODO 6a: 投影为 Q、K、V（各自过 W_q / W_k / W_v）
        # ------------------------------------------------------------
        Q, K, V = None, None, None
        raise NotImplementedError("TODO 6a: Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)")

        # ------------------------------------------------------------
        # TODO 5: 实现 split_heads，并用它把 Q/K/V 切成多头
        #   [seq_len, d_model] → [seq_len, num_heads, d_head] → [num_heads, seq_len, d_head]
        # ------------------------------------------------------------
        def split_heads(t: Tensor) -> Tensor:
            raise NotImplementedError("TODO 5: 实现 split_heads")

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        # 现在 shape: [num_heads, seq_len, d_head]

        # ------------------------------------------------------------
        # TODO 6b: 每个头独立调用 scaled_dot_product_attention，
        #          再把 num_heads 个 [seq_len, d_head] 输出拼接回 [seq_len, d_model]
        # ------------------------------------------------------------
        outputs = []
        raise NotImplementedError("TODO 6b: 遍历 num_heads，逐头计算注意力并收集输出")
        concat = torch.cat(outputs, dim=-1)  # [seq_len, d_model]

        # ------------------------------------------------------------
        # TODO 6c: 输出投影
        # ------------------------------------------------------------
        raise NotImplementedError("TODO 6c: return self.W_o(concat)")
