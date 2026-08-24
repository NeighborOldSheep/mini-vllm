# Phase 0 Coding Lab — 基础组件

> 对应学习大纲 [`学习大纲.md`](../学习大纲.md) 第 2 节 Phase 0（`step01_tokenizer` ~ `step04_transformer`）。
> 目的不是抄一遍代码，而是**独立实现**这四个组件，用测试脚本验证自己真的理解了每一处计算。

## 这个 Lab 和原教程的区别

`mini-vllm-tutorial/step01~04` 已经给出完整实现，直接读代码容易"看懂了"但"写不出来"。
本 Lab 把关键计算逻辑挖空成 `TODO`，你需要自己填上，再用 `test_*.py` 自测——全部测试通过才算掌握。

如果卡住超过 15 分钟，去对照 `mini-vllm-tutorial/stepXX_xxx/` 下的对应文件，理解后**关掉参考、自己重写**，不要复制粘贴。

## 目录结构

```
phase0_基础组件/
├── lab01_tokenizer/       ← 对应 step01：字节级 BPE
│   ├── README.md          ← 任务说明、TODO 清单、验收标准
│   ├── tokenizer_lab.py   ← 你要填的代码
│   └── test_tokenizer_lab.py
├── lab02_embedding/       ← 对应 step02：Embedding 查找表 + 可学习性
│   ├── README.md
│   ├── embedding_lab.py
│   └── test_embedding_lab.py
├── lab03_attention/       ← 对应 step03：Scaled Dot-Product / Multi-Head Attention
│   ├── README.md
│   ├── attention_lab.py
│   └── test_attention_lab.py
├── lab04_transformer/     ← 对应 step04：RMSNorm + MLP + Decoder Layer + TinyTransformer
│   ├── README.md
│   ├── transformer_lab.py
│   └── test_transformer_lab.py
└── capstone_end_to_end/   ← 串联四步：一段中文文本 → token → 向量 → 注意力 → logits
    ├── README.md
    └── capstone.py
```

`lab04` 会 `import` 你在 `lab02`（Embedding）和 `lab03`（MultiHeadAttention）里写的代码——前面没做对，后面会直接报错。这是刻意设计的：Phase 0 里这四个组件本来就是逐层依赖的。

## 环境

```bash
pip install torch>=2.4.0   # 与 mini-vllm-tutorial/requirements-cpu.txt 一致，纯 CPU 可跑
```

## 学习顺序与验收方式

| 顺序 | Lab | 你要实现 | 自测命令 | 通过标准 |
|---|---|---|---|---|
| 1 | `lab01_tokenizer` | BPE 训练循环、pair 合并、encode/decode | `python lab01_tokenizer/test_tokenizer_lab.py` | 全部 `PASS`，含中文 round-trip |
| 2 | `lab02_embedding` | 查表 forward、余弦相似度、训练循环 | `python lab02_embedding/test_embedding_lab.py` | 全部 `PASS`，含梯度局部更新验证 |
| 3 | `lab03_attention` | Scaled Dot-Product Attention、多头拆分/拼接 | `python lab03_attention/test_attention_lab.py` | 全部 `PASS`，含因果 mask 与 softmax 归一化验证 |
| 4 | `lab04_transformer` | RMSNorm、SwiGLU MLP、Decoder 层、TinyTransformer | `python lab04_transformer/test_transformer_lab.py` | 全部 `PASS`，含因果性验证 |
| 5 | `capstone_end_to_end` | 无新增 TODO，跑通全链路 | `python capstone_end_to_end/capstone.py` | 打印出 `logits` 形状与 top-1 预测 token，无报错 |

每个 `test_*.py` 都是独立脚本（不依赖 pytest，风格与原教程 `run.py` 一致），跑起来会逐项打印 `[PASS]`/`[FAIL]`，并在最后给出通过计数。

## 自查清单（完成本 Lab 后应该能回答）

- [ ] BPE 训练时，为什么要按"频率最高的相邻 pair"合并，而不是随机合并？
- [ ] `count_pairs` 和 `apply_merge` 为什么要分开成两个函数？各自的时间复杂度是什么？
- [ ] Embedding 的 `forward` 为什么是"取行"而不是矩阵乘法？两者在数学上等价吗？
- [ ] 为什么 `loss.backward()` 之后，Embedding 权重矩阵里**只有出现过的 token 行**梯度非零？
- [ ] `scores / sqrt(d_head)` 这个缩放为什么必要？如果去掉会发生什么？（提示：softmax 输入方差）
- [ ] 因果 mask 是加在 softmax **之前**还是**之后**？为什么？
- [ ] 多头注意力里，为什么切成多个头之后拼接的效果，通常比一个大头更好？
- [ ] RMSNorm 和 LayerNorm 差在哪一步？为什么现代 LLM 更常用 RMSNorm？
- [ ] Pre-Norm（`x + F(norm(x))`）和 Post-Norm（`norm(x + F(x))`）的区别？残差连接解决了什么训练问题？
- [ ] `capstone_end_to_end` 里，为什么 `TinyTransformer` 的 `vocab_size` 必须等于 tokenizer 的 `vocab_size`？如果不等会发生什么？

完成 Phase 0 Lab 后，回到 [`学习大纲.md`](../学习大纲.md) 继续 Phase 1（`step05_naive`）。
