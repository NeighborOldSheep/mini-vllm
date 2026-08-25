"""
Lab 02: Embedding — token_id 到向量的查表过程（挖空版）

参考: mini-vllm-tutorial/step02_embedding/embedding.py
完成所有 TODO 后运行 test_embedding_lab.py 自测。
"""

import torch
import torch.nn as nn
from torch import Tensor


class Embedding(nn.Module):
    """
    词嵌入层：将 token_id 映射为 d_model 维向量。

    内部是一个矩阵 weight: [vocab_size, d_model]
    调用时 weight[token_id] 取出对应行。
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model))
        self.vocab_size = vocab_size
        self.d_model = d_model

    # ------------------------------------------------------------------
    # TODO 1
    # ------------------------------------------------------------------
    def forward(self, token_ids: Tensor) -> Tensor:
        """
        输入: token_ids  形状 [seq_len]，值域 [0, vocab_size)
        输出: 向量矩阵   形状 [seq_len, d_model]

        操作本质：output[i] = self.weight[token_ids[i]]
        """
        return self.weight[token_ids]


# ------------------------------------------------------------------
# TODO 2
# ------------------------------------------------------------------
def cosine_similarity(a: Tensor, b: Tensor) -> float:
    """
    计算两个向量（或 [1, d_model] 矩阵）的余弦相似度。

    公式：cos(θ) = (a·b) / (‖a‖ × ‖b‖)
    取值：[-1, 1]，越接近 1 越相似
    """
    a = a.flatten().float()
    b = b.flatten().float()

    return (
        torch.dot(a,b) / (torch.norm(a) * torch.norm(b))
    ).item()


# ------------------------------------------------------------------
# TODO 3（原教程没有，本 Lab 新增练习）
# ------------------------------------------------------------------
def most_similar(query_vec: Tensor, weight: Tensor, top_k: int = 3) -> Tensor:
    """
    在整张 Embedding 权重表里，找出与 query_vec 余弦相似度最高的 top_k 个 token id。

    输入:
        query_vec: [d_model]
        weight:    [vocab_size, d_model]
        top_k:     返回几个结果
    输出:
        token id 张量，形状 [top_k]，按相似度从高到低排序

    要求：不允许写 for 循环遍历 vocab，必须向量化实现。
    提示：
        1. 归一化 weight 的每一行、归一化 query_vec（除以各自范数）
        2. 归一化后的 weight 与归一化后的 query_vec 做矩阵-向量乘法
           一次得到 [vocab_size] 的相似度向量
        3. torch.topk(sims, k=top_k) 取前 k 个
    """
    query_norm = torch.norm(query_vec)
    weight_norm = torch.norm(weight,dim=1,keepdim=True)

    query_normalization = query_vec / query_norm 
    weight_normalization = weight / weight_norm

    sim = weight_normalization @ query_normalization 

    return torch.topk(sim,k=top_k).indices