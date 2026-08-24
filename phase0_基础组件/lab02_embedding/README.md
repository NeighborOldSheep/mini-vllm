# Lab 02 — Embedding：查找表 + 可学习性

对应：[`mini-vllm-tutorial/step02_embedding`](../../mini-vllm-tutorial/step02_embedding/README.md)（原理讲解请先读那份 README）

## 目标

1. 实现 Embedding 的 `forward`：本质是"用 token_id 做行索引"，而不是矩阵乘法
2. 实现余弦相似度，用它验证向量方向关系
3. 实现一个"批量找最相似 token"的向量化函数（不写 for 循环）
4. 完成一个极简训练循环，亲眼验证"梯度只更新出现过的行"

## 文件

- `embedding_lab.py` — 待完成代码，4 处 `TODO`
- `test_embedding_lab.py` — 自测脚本

## TODO 清单

### TODO 1：`Embedding.forward`

```
输入: token_ids  形状 [seq_len]，值域 [0, vocab_size)
输出: 向量矩阵   形状 [seq_len, d_model]
操作: output[i] = self.weight[token_ids[i]]
```

一行代码：用 `token_ids` 直接对 `self.weight` 做行索引（PyTorch 的花式索引，等价于 `nn.Embedding`）。

### TODO 2：`cosine_similarity(a, b)`

```
cos(θ) = (a·b) / (‖a‖ × ‖b‖)
```

用 `torch.dot` 和 `torch.norm`，返回 Python `float`（用 `.item()`）。输入可能是 `[1, d_model]` 的矩阵（batch=1），先 `.flatten()` 再算。

### TODO 3：`most_similar(query_vec, weight, top_k)`

**原教程没有这个函数，这是本 Lab 新增的练习**：给定一个查询向量 `query_vec: [d_model]` 和整张 Embedding 权重表 `weight: [vocab_size, d_model]`，返回与 `query_vec` 余弦相似度最高的 `top_k` 个 token id（按相似度降序）。

要求：**不允许写 for 循环遍历 vocab**，必须用向量化操作一次算出所有 token 的相似度。

提示：
```python
# 1. 对 weight 的每一行和 query_vec 分别做归一化（除以各自的范数）
# 2. 归一化后的 weight 与归一化后的 query_vec 做矩阵-向量乘法，一次得到 vocab_size 个相似度
# 3. torch.topk 取前 k 个
```

这个练习是为后面 step14/step15 里"批量算注意力，不逐请求 for 循环"做铺垫——向量化思维会反复出现。

### TODO 4：`train_step(emb, head, x, y, optimizer, loss_fn)`

在 `train_lab.py` 里，实现一次训练迭代：

```
1. logits = head(emb(x))       # [batch, vocab_size]
2. loss = loss_fn(logits, y)
3. optimizer.zero_grad()
4. loss.backward()
5. optimizer.step()
6. 返回 loss.item()
```

## 验收标准

```bash
python test_embedding_lab.py
```

包含以下验证：
- 单个/批量 token_id 查表后的 shape 正确
- `cosine_similarity(v, v) ≈ 1.0`（自相似）
- `most_similar` 能在人工构造的小词表里找出真正最相似的 top-k（构造一个已知答案的场景来验证，而不是靠随机初始化）
- **梯度局部更新验证**：跑一步 `train_step` 后，只有 batch 中出现过的 token 行的权重发生了变化，未出现的行权重完全不变（`torch.allclose`）

## 想深入一步（选做）

- `most_similar` 里如果不做归一化，直接用点积排序，结果会不会不同？为什么点积和余弦相似度在向量模长不一致时会给出不同排序？
- 训练时如果两个 batch 之间共享某个 token（比如都出现了 "the"），这个 token 的行会被两次梯度更新，是否会互相干扰？这对应真实 LLM 训练里"高频词收敛更快"的现象。
