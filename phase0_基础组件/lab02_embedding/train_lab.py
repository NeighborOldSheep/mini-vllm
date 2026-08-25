"""
Lab 02 附加练习: 训练循环（挖空版）

参考: mini-vllm-tutorial/step02_embedding/train.py
完成 TODO 4 后，test_embedding_lab.py 会用它验证"梯度只更新出现过的行"。
"""

import torch
import torch.nn as nn
from embedding_lab import Embedding,cosine_similarity


# ------------------------------------------------------------------
# TODO 4
# ------------------------------------------------------------------
def train_step(emb: Embedding, head: nn.Linear, x, y, optimizer, loss_fn) -> float:
    """
    执行一次训练迭代，返回本次 loss（float）。

    步骤：
      1. logits = head(emb(x))       # [batch, vocab_size]
      2. loss = loss_fn(logits, y)
      3. optimizer.zero_grad()
      4. loss.backward()
      5. optimizer.step()
      6. return loss.item()
    """
    
    logits = head(emb(x))
    loss = loss_fn(logits,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def build_toy_model(vocab_size: int = 256, d_model: int = 16):
    """已提供：构建一个极简 next-token 预测模型（Embedding + LM head）。"""
    emb = Embedding(vocab_size, d_model)
    head = nn.Linear(d_model, vocab_size, bias=False)
    optimizer = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    return emb, head, optimizer, loss_fn
