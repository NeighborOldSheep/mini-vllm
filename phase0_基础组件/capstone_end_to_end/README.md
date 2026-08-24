# Capstone — 端到端串联 Phase 0 四个组件

> 这里没有新的 TODO。目的是让你**亲眼看到**四个 Lab 拼在一起的完整数据流：
> 一段中文/英文文本 → tokenizer 编码成 token id → TinyTransformer（内部用你写的 Embedding + Attention）
> → 输出每个位置的下一 token 概率分布。

## 前置条件

`lab01` ~ `lab04` 的自测必须**全部通过**，否则这里会直接报错（错误信息会指向具体是哪个 Lab 的问题）。

## 运行

```bash
python capstone.py
```

## 会看到什么

```
输入文本: '你好，世界！Hello!'
Tokenizer 词表大小: <你训练出来的实际大小，取决于 lab01 的实现>
Token IDs: [...]

TinyTransformer 配置: vocab_size=<与 tokenizer 一致>, d_model=32, num_heads=4, num_layers=2
输入 shape: [seq_len]  →  输出 logits shape: [seq_len, vocab_size]

每个位置 top-1 预测 token id: [...]
（注意：TinyTransformer 权重是随机初始化的，预测结果没有语义，
  这里只验证"数据能不能在四层组件之间正确流动，形状对不对"）

✅ capstone 跑通，Phase 0 全部完成
```

## 为什么 `vocab_size` 必须对齐

`TinyTransformer` 的 `lm_head` 输出维度是 `vocab_size`，如果这个数字和 tokenizer 实际训练出的词表大小不一致：

- 传入 tokenizer 编码出的 token id（比如某个 id=400）
- 但 `TinyTransformer` 的 `Embedding` 权重表只有 300 行

`self.weight[token_ids]` 会直接越界报错。这个 capstone 会用 `tok.vocab_size` 动态构造 `TinyTransformer`，而不是像 `step04_transformer/run.py` 里那样写死 `vocab_size=256`——这是刻意设计，逼你注意到"分词器词表大小"和"模型 embedding 表大小"必须是同一个数字，这个约束在后面 Phase 6 接入真实 Qwen3 模型时会以更严格的形式出现（权重文件的 shape 决定一切，对不上就是 `size mismatch` 报错）。

## 下一步

Phase 0 全部完成后，回到 [`../../学习大纲.md`](../../学习大纲.md)，进入 Phase 1（`step05_naive`）：让 TinyTransformer 真正跑起来做自回归生成，并亲眼看到 O(n²) 问题有多严重。
