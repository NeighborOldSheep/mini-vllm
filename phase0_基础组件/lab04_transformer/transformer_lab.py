"""
Lab 04: 完整 Transformer Decoder 层 + TinyTransformer（挖空版）

参考: mini-vllm-tutorial/step04_transformer/transformer.py
完成所有 TODO 后运行 test_transformer_lab.py 自测。

依赖你在 lab02_embedding / lab03_attention 里写的代码——先确保那两个 Lab 的
自测全部通过，否则这里的报错可能来自之前没修完的 TODO。
"""

import sys
import os
import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab02_embedding"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab03_attention"))
from embedding_lab import Embedding          # noqa: E402
from attention_lab import MultiHeadAttention  # noqa: E402


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization。

    比 LayerNorm 更简单：不减均值，只除均方根。
    公式：output = x / RMS(x) * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    # ------------------------------------------------------------
    # TODO 1
    # ------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """
        RMS(x) = sqrt(mean(x², dim=-1, keepdim=True) + eps)
        output = x / RMS(x) * weight
        """
        rms = x.pow(2).mean(-1,keepdim=True).add(self.eps).sqrt()

        return x / rms * self.weight


class MLP(nn.Module):
    """
    Transformer MLP 层（SwiGLU 版本）。

    结构：
      gate = SiLU(x · W_gate)
      up   = x · W_up
      output = (gate * up) · W_down
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()

    # ------------------------------------------------------------
    # TODO 2
    # ------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """
        gate = self.act(self.W_gate(x))
        up   = self.W_up(x)
        return self.W_down(gate * up)
        """

        return self.W_down(self.act(self.W_gate(x)) * self.W_up(x))


class TransformerDecoderLayer(nn.Module):
    """
    单个 Transformer Decoder 层（Pre-Norm 结构）。

    数据流：
      x → norm1 → MultiHeadAttention → + x  (残差)
        → norm2 → MLP               → + x  (残差)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    # ------------------------------------------------------------
    # TODO 3
    # ------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """
        x = x + self.attn(self.norm1(x))   # 注意力子层，残差加回原始 x
        x = x + self.mlp(self.norm2(x))    # MLP 子层，残差加回原始 x
        return x
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class TinyTransformer(nn.Module):
    """
    用于教学的小型 Transformer 语言模型。

    结构：Embedding → N × DecoderLayer → RMSNorm → LM Head
    随机初始化权重（不会生成有意义的文字，用于演示推理流程）。
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        d_ff = d_model * 4

        self.embed = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    # ------------------------------------------------------------
    # TODO 4
    # ------------------------------------------------------------
    def forward(self, token_ids: Tensor) -> Tensor:
        """
        输入: token_ids  形状 [seq_len]
        输出: logits     形状 [seq_len, vocab_size]

        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
        """
        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits