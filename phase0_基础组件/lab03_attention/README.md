# Lab 03 — Scaled Dot-Product Attention / Multi-Head Attention

对应：[`mini-vllm-tutorial/step03_attention`](../../mini-vllm-tutorial/step03_attention/README.md)（原理讲解请先读那份 README）

## 目标

1. 手写 Scaled Dot-Product Attention 的四步计算：打分 → 因果 mask → softmax → 加权求和
2. 实现多头拆分 / 拼接，理解"多个子空间独立算注意力再合并"
3. 验证因果 mask、softmax 归一化这些"性质"，而不只是跑通一次前向

## 文件

- `attention_lab.py` — 待完成代码，6 处 `TODO`
- `test_attention_lab.py` — 自测脚本

## TODO 清单

### TODO 1：`scaled_dot_product_attention` — 计算相似度分数

```
scores = Q · Kᵀ / √d_head
```
`Q, K` 形状均为 `[seq_len, d_head]`，`scores` 形状 `[seq_len, seq_len]`。用 `torch.matmul` + `math.sqrt`。

### TODO 2：应用因果 mask

生成时 token `i` 不能看到 `j > i`（未来）。做法：

1. 构造上三角（不含对角线）为 `True` 的布尔矩阵：`torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()`
2. 用 `scores.masked_fill(mask, float("-inf"))` 把这些位置的分数设为 `-inf`

**为什么在 softmax 之前 mask，而不是之后？** 想清楚这个问题再动手——留到 README 末尾的思考题。

### TODO 3：softmax 归一化

```
weights = softmax(scores, dim=-1)
```
沿最后一维（每一行内部）做 softmax，保证每行权重和为 1。

### TODO 4：加权求和

```
output = weights · V
```
`output[i] = Σ_j weights[i,j] * V[j]`，用 `torch.matmul`。

### TODO 5：`MultiHeadAttention.split_heads`

把 `[seq_len, d_model]` 的 Q/K/V 拆成 `[num_heads, seq_len, d_head]`：

```python
def split_heads(t):
    # [seq_len, d_model] → [seq_len, num_heads, d_head] → [num_heads, seq_len, d_head]
    return t.view(seq_len, self.num_heads, self.d_head).transpose(0, 1)
```

### TODO 6：`MultiHeadAttention.forward` 主流程

```
1. Q, K, V = W_q(x), W_k(x), W_v(x)          # [seq_len, d_model]
2. Q, K, V = split_heads(Q), split_heads(K), split_heads(V)   # [num_heads, seq_len, d_head]
3. 对每个 head h，调用 TODO1~4 实现的 scaled_dot_product_attention(Q[h], K[h], V[h], causal=True)
4. 把 num_heads 个 [seq_len, d_head] 输出沿最后一维拼接回 [seq_len, d_model]
5. 过输出投影 W_o
```

## 验收标准

```bash
python test_attention_lab.py
```

包含：
- `scaled_dot_product_attention` 输出 / 权重矩阵 shape 正确
- 因果 mask 生效：`weights[i, j] ≈ 0` 对所有 `j > i`
- softmax 归一化：每一行权重和 ≈ 1
- `causal=False` 时上三角权重不再为 0（验证 mask 是可选项，不是硬编码行为）
- `MultiHeadAttention` 输出 shape 正确
- **head 数=1 时的一致性检查**：当 `num_heads=1`，`MultiHeadAttention` 内部的注意力计算应该退化为对整个 `d_model` 做一次单头注意力（拼接和拆分不引入任何数值变化）

## 想深入一步（选做）

- **为什么必须在 softmax 之前应用因果 mask？** 如果改成"softmax 之后把对应位置权重设为 0"，会破坏什么性质？（提示：想想每行权重和还等不等于 1）
- 缩放因子 `1/√d_head` 去掉会发生什么？构造一个 `d_head` 较大（比如 512）的例子，观察 `scores` 的数值范围和 softmax 输出的"尖锐程度"（可以在 `attention_lab.py` 外面自己写段代码试验，不需要放进测试）。
- 本 Lab 的多头实现是"for 循环遍历每个 head"，原教程 `step14_3_batched_attention`（Phase 6）会把它改成一次批量矩阵运算。提前想一下：`Q, K, V` 已经是 `[num_heads, seq_len, d_head]`，怎样用一次 `torch.matmul`（而不是循环）算出所有 head 的 `scores`？
