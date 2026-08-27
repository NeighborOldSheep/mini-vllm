"""Phase 1 Lab：朴素自回归推理。

这个类接收任意满足 ``model(input_ids) -> logits`` 的模型，以便测试能够精确
检查控制流。真实模型输出形状为 [seq_len, vocab_size]。
"""

import torch
from torch import Tensor


def attention_score_elements(seq_len: int) -> int:
    """返回长度为 seq_len 的单头 Attention score 矩阵元素数。"""
    if seq_len < 0:
        raise ValueError("seq_len 必须 >= 0")
    # TODO 1：返回 n²。
    raise NotImplementedError("TODO 1: 实现 attention_score_elements")


def total_naive_attention_work(prompt_len: int, max_new_tokens: int) -> int:
    """估计朴素 generate 的 Attention score 元素总量。

    第 i 次解码（i 从 0 开始）会输入 prompt_len + i 个 token。
    因此结果是 sum((prompt_len + i)² for i in range(max_new_tokens))。
    """
    if prompt_len < 0 or max_new_tokens < 0:
        raise ValueError("长度必须 >= 0")
    # TODO 2：调用 attention_score_elements，不要手写另一份公式。
    raise NotImplementedError("TODO 2: 实现 total_naive_attention_work")


class NaiveEngine:
    """每次 decode 都把完整历史序列重新送入模型的基线引擎。"""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def decode_one_step(self, input_ids: Tensor) -> Tensor:
        """从完整历史 input_ids 预测一个 next token 的标量 Tensor。"""
        # TODO 3：完整前向；只取最后位置 logits；返回 argmax。
        raise NotImplementedError("TODO 3: 实现 decode_one_step")

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, max_new_tokens: int) -> Tensor:
        """生成并返回 [prompt..., generated...]，且绝不修改 prompt_ids。"""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 必须 >= 0")
        # TODO 4：clone prompt；循环 decode_one_step；把标量 next_id 变成 [1] 后追加。
        raise NotImplementedError("TODO 4: 实现 generate")
