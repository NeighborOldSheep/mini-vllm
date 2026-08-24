# Lab 01 — 字节级 BPE Tokenizer

对应：[`mini-vllm-tutorial/step01_tokenizer`](../../mini-vllm-tutorial/step01_tokenizer/README.md)（原理讲解请先读那份 README）

## 目标

独立实现一个字节级 BPE（Byte Pair Encoding）分词器，掌握：

1. 训练阶段如何从"相邻字节对频率"里学出合并规则
2. 编码阶段如何贪心应用合并规则把字节序列压缩成更短的 token 序列
3. 解码阶段如何无损还原原文（任意 UTF-8 文本，不出现 `<unk>`）

## 文件

- `tokenizer_lab.py` — 待完成的代码，5 处 `TODO`，每处对应 `raise NotImplementedError`
- `test_tokenizer_lab.py` — 自测脚本，`python test_tokenizer_lab.py` 运行

## TODO 清单

### TODO 1：`count_pairs(corpus)`

统计一批 token id 序列（`corpus: List[List[int]]`）里，所有**相邻二元组**出现的次数。

```
输入 corpus = [[116, 104, 101, 32], [116, 104, 101]]
输出：Counter({(116,104): 2, (104,101): 2, (101,32): 1})
```

提示：对每条序列用 `zip(seq, seq[1:])` 遍历相邻对。

### TODO 2：`apply_merge(seq, pair, new_id)`

把序列 `seq` 中所有**相邻且等于 `pair`** 的两个元素替换为 `new_id`。

```
apply_merge([116, 104, 101, 32, 116, 104], (116, 104), 256)
→ [256, 101, 32, 256]
```

注意：替换后跳过被合并的两个位置（不能重叠匹配），用双指针或 `while` 循环实现，不要用递归正则。

### TODO 3：`_train_merges(self)`

用 TODO1/TODO2 实现完整训练循环：

```
for 每一轮（最多 MERGE_COUNT 轮）:
    1. 统计当前 corpus 里所有相邻 pair 的频率（用 count_pairs）
    2. 如果没有任何 pair 了，提前结束
    3. 取频率最高的 pair
    4. 给它分配一个新 token id（从 BASE_VOCAB_SIZE 开始递增）
    5. 更新 self.vocab / self._bytes_to_id / self.merges
    6. 用 apply_merge 把这个 pair 在整个 corpus 里替换掉，供下一轮统计使用
```

训练语料已经在 `__init__` 附近以 `self._get_train_corpus()` 提供，不需要你自己写。

### TODO 4：`encode(self, text)`

1. 把 `text` 转成 UTF-8 字节，得到初始 token id 列表（每个字节一个 id，0~255）
2. 按 `self.merges` **训练时产生的顺序**依次应用合并规则（`self.merges` 是按插入顺序遍历的 dict，直接 `for pair, new_id in self.merges.items()` 即可）

### TODO 5：`decode(self, ids)`

1. 用 `self.vocab` 把每个 id 查回对应的 `bytes`
2. 拼接所有 bytes
3. UTF-8 解码回字符串（用 `errors="replace"` 避免因非法字节序列崩溃）

## 验收标准

```bash
python test_tokenizer_lab.py
```

全部用例打印 `[PASS]`，包括：
- 英文文本 encode→decode round-trip 一致
- 中文文本 encode→decode round-trip 一致（验证字节级覆盖，无 `<unk>`）
- 训练后词表大小 = `256 + 实际 merges 数量`（注意：训练语料只有 4 行固定文本，相邻 pair 会在凑满 `MERGE_COUNT=256` 之前被穷尽，所以 `_train_merges` 必须正确处理"提前 break"，不要期望 merges 数量恰好等于 256）
- 高频组合（如训练语料里反复出现的词）编码后 token 数量明显少于原始字节数
- `count_pairs` / `apply_merge` 的独立单元测试

## 想深入一步（选做，不计入通过标准）

- 现在的 `encode` 是"按训练顺序依次替换"，如果换成"每轮都重新找当前最高频 pair 再合并"（更接近真实 tokenizer 的贪心策略），结果会不会不同？为什么两种策略在实践中通常等价？
- 试着解释：为什么词表大小固定后，"预分词"（先按空格/标点切开再做 BPE）能提升语义质量？（原教程 README 末尾有提示）
