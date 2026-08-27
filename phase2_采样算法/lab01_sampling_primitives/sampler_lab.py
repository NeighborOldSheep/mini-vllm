"""Phase 2 Lab：logits -> next token 的采样原语。"""

import torch
import torch.nn.functional as F
from torch import Tensor


def _check_temperature(temperature: float) -> None:
    if temperature <= 0:
        raise ValueError("temperature 必须 > 0；temperature=0 应由引擎分派为 greedy")


def greedy_sample(logits: Tensor) -> Tensor:
    """选择最大 logit，对随机种子不敏感。"""
    # TODO 1
    raise NotImplementedError("TODO 1: 实现 greedy_sample")


def temperature_probabilities(logits: Tensor, temperature: float) -> Tensor:
    """返回 softmax(logits / temperature) 概率，用于检查温度的分布效应。"""
    _check_temperature(temperature)
    # TODO 2
    raise NotImplementedError("TODO 2: 实现 temperature_probabilities")


def temperature_sample(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """按温度缩放后的分类分布抽取一个标量 token id。"""
    # TODO 3：复用 temperature_probabilities；torch.multinomial(..., 1) 的结果是 [1]，需压成标量。
    raise NotImplementedError("TODO 3: 实现 temperature_sample")


def filter_top_k(logits: Tensor, k: int) -> Tensor:
    """返回仅 top-k 位置有限、其余为 -inf 的新 logits，且不修改输入。"""
    if not 1 <= k <= logits.numel():
        raise ValueError("k 必须在 [1, vocab_size] 内")
    # TODO 4：torch.topk 得到 values 和 indices；使用 full_like + scatter 保留恰好 k 项。
    raise NotImplementedError("TODO 4: 实现 filter_top_k")


def top_k_sample(logits: Tensor, k: int, temperature: float = 1.0) -> Tensor:
    """先 top-k 截断，再按温度采样。"""
    # TODO 5：复用 filter_top_k 和 temperature_sample。
    raise NotImplementedError("TODO 5: 实现 top_k_sample")


def filter_top_p(logits: Tensor, p: float) -> Tensor:
    """返回最小的、累计原始概率质量至少为 p 的候选集。"""
    if not 0 < p <= 1:
        raise ValueError("p 必须在 (0, 1] 内")
    # TODO 6：按降序 sort logits；对 softmax 后的 sorted logits 求 cumsum。
    # 保留条件可写作：前一个累计概率 < p。它会保留使累计值第一次达到 p 的边界 token。
    # 最后将 sorted 空间的结果 scatter 回原 token id 顺序。
    raise NotImplementedError("TODO 6: 实现 filter_top_p")


def top_p_sample(logits: Tensor, p: float, temperature: float = 1.0) -> Tensor:
    """先 nucleus / top-p 截断，再按温度采样。"""
    # TODO 7
    raise NotImplementedError("TODO 7: 实现 top_p_sample")


def gumbel_max_sample(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """使用 Gumbel-Max trick 采样，分布等价于 temperature sampling。"""
    _check_temperature(temperature)
    # TODO 8：u = rand_like(logits)，为避免 log(0) 可 clamp；
    # g = -log(-log(u))；返回 argmax(logits / temperature + g)。
    raise NotImplementedError("TODO 8: 实现 gumbel_max_sample")
