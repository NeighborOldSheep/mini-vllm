# Lab 04 — RMSNorm + MLP + Decoder 层 + TinyTransformer

对应：[`mini-vllm-tutorial/step04_transformer`](../../mini-vllm-tutorial/step04_transformer/README.md)（原理讲解请先读那份 README）

## 前置

这个 Lab 会 `import` 你在前两个 Lab 里写的代码：

```python
from lab02_embedding.embedding_lab import Embedding
from lab03_attention.attention_lab import MultiHeadAttention
```

**先确保 `lab02` 和 `lab03` 的自测全部通过**，否则这里的错误可能来自之前没修完的 TODO，而不是本 Lab 新引入的问题。

## 目标

把已经实现的 Embedding、Attention 组装成一个完整的 Transformer Decoder 层，再叠加成一个可以做前向推理的小模型，理解：

1. RMSNorm 相比 LayerNorm 少了什么步骤
2. SwiGLU 结构的 MLP：为什么有两条并行的线性变换（gate 和 up）
3. Pre-Norm + 残差连接：`x = x + F(norm(x))`，为什么这样组织子层
4. 因果性在多层堆叠后依然成立：修改最后一个 token，不影响之前所有位置的输出

## 文件

- `transformer_lab.py` — 待完成代码，4 处 `TODO`
- `test_transformer_lab.py` — 自测脚本

## TODO 清单

### TODO 1：`RMSNorm.forward`

```
RMS(x) = sqrt(mean(x², dim=-1))
output = x / RMS(x) * weight
```

注意 `mean` 要在最后一维上算（`dim=-1, keepdim=True`），并加一个很小的 `eps` 防止除零：`.add(self.eps)`（在开根号前加）。

### TODO 2：`MLP.forward`（SwiGLU）

```
gate = SiLU(x · W_gate)
up   = x · W_up
output = (gate * up) · W_down
```

`self.act` 已经是 `nn.SiLU()`，直接调用。逐元素相乘 `gate * up`，不是矩阵乘法。

### TODO 3：`TransformerDecoderLayer.forward`（Pre-Norm + 残差）

```
x = x + self.attn(self.norm1(x))   # 注意力子层
x = x + self.mlp(self.norm2(x))    # MLP 子层
return x
```

先 norm 再变换，变换结果加回原始输入（不是加回 norm 后的值）。

### TODO 4：`TinyTransformer.forward`

```
x = self.embed(token_ids)      # [seq_len, d_model]
for layer in self.layers:
    x = layer(x)
x = self.norm(x)                # 最终归一化
logits = self.lm_head(x)        # [seq_len, vocab_size]
return logits
```

## 验收标准

```bash
python test_transformer_lab.py
```

包含：
- `RMSNorm` 输出的均方根 ≈ 1（在 `weight` 全 1 的情况下，输出的 RMS 应该被归一化到 1）
- `RMSNorm` 与手算结果数值一致（构造一个具体向量手工验证）
- `MLP` 输出 shape 正确
- `TransformerDecoderLayer` 输出 shape 与输入一致（`[seq_len, d_model]` 不变，这是残差结构的要求）
- `TinyTransformer` 前向输出 shape 为 `[seq_len, vocab_size]`
- **因果性验证**：修改 `token_ids` 最后一个位置的值，重新前向，前面所有位置的 logits 必须完全不变——这个性质要在**多层堆叠**之后依然成立，是对 Lab03 因果 mask 是否正确实现的间接检验
- **残差连接检验**：把某一层的 attn/mlp 输出强制设为全 0（monkeypatch），此时该层输出应该等于输入（验证残差是"加回原始 x"而不是"加回 norm(x)"）

## 想深入一步（选做）

- Pre-Norm（`x + F(norm(x))`）和 Post-Norm（`norm(x + F(x))`）的区别？为什么现代大模型几乎都用 Pre-Norm？（提示：想想深层网络里梯度反传路径上有没有被 Norm "打断"）
- `MLP` 里 `d_ff` 通常是 `d_model` 的 4 倍（`TinyTransformer` 里 `d_ff = d_model * 4`），如果去掉 gate 分支，变成普通的两层 MLP（`W_down(SiLU(W_gate(x)))`，没有 `up` 分支），参数量会怎么变？为什么 SwiGLU 仍然是当前主流选择？
