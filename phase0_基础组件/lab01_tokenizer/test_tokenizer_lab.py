"""
Lab 01 自测脚本。完成 tokenizer_lab.py 里的全部 TODO 后运行：
    python test_tokenizer_lab.py
"""

from collections import Counter
from tokenizer_lab import SimpleBPETokenizer

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"[PASS] {name}")
    else:
        _failed += 1
        print(f"[FAIL] {name}  {detail}")


def test_count_pairs():
    corpus = [[116, 104, 101, 32], [116, 104, 101]]
    result = SimpleBPETokenizer.count_pairs(corpus)
    expected = Counter({(116, 104): 2, (104, 101): 2, (101, 32): 1})
    check("count_pairs 基础统计", result == expected, f"got {dict(result)}")

    check("count_pairs 空输入返回空", SimpleBPETokenizer.count_pairs([]) == Counter())
    check(
        "count_pairs 单元素序列无 pair",
        SimpleBPETokenizer.count_pairs([[1]]) == Counter(),
    )


def test_apply_merge():
    result = SimpleBPETokenizer.apply_merge([116, 104, 101, 32, 116, 104], (116, 104), 256)
    check("apply_merge 基础替换", result == [256, 101, 32, 256], f"got {result}")

    # 不应重叠匹配：[1,1,1] 合并 (1,1) → 应该是 [merged, 1]，不是 [merged, merged部分]
    result2 = SimpleBPETokenizer.apply_merge([1, 1, 1], (1, 1), 99)
    check("apply_merge 不重叠匹配", result2 == [99, 1], f"got {result2}")

    result3 = SimpleBPETokenizer.apply_merge([5, 6, 7], (1, 2), 99)
    check("apply_merge 无匹配时原样返回", result3 == [5, 6, 7], f"got {result3}")


def test_training_and_vocab():
    tok = SimpleBPETokenizer()
    # 注意：训练语料很小（4 行固定文本），相邻 pair 会在达到 MERGE_COUNT 之前就被穷尽，
    # 训练循环应提前 break，所以 merges 数量 <= MERGE_COUNT，不一定恰好等于它。
    check(
        "词表大小 = 256 + 实际 merges 数量",
        tok.vocab_size == SimpleBPETokenizer.BASE_VOCAB_SIZE + len(tok.merges),
        f"vocab_size={tok.vocab_size}, merges={len(tok.merges)}",
    )
    check(
        "merges 数量在 (0, MERGE_COUNT] 之间",
        0 < len(tok.merges) <= SimpleBPETokenizer.MERGE_COUNT,
        f"got {len(tok.merges)}",
    )
    check(
        "训练语料耗尽时应提前停止（不强行凑满 MERGE_COUNT）",
        len(tok.merges) < SimpleBPETokenizer.MERGE_COUNT,
        "如果你的实现在 pair_counts 为空时没有 break，这里会不通过",
    )


def test_roundtrip_and_compression():
    tok = SimpleBPETokenizer()

    text = "Hello, world!"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    check("英文 round-trip", decoded == text, f"{decoded!r} != {text!r}")

    text_cn = "你好世界"
    ids_cn = tok.encode(text_cn)
    decoded_cn = tok.decode(ids_cn)
    check("中文 round-trip（字节级无 <unk>）", decoded_cn == text_cn, f"{decoded_cn!r} != {text_cn!r}")

    # 训练语料里出现的高频词，编码后应比原始字节数明显短
    frequent_text = "the quick brown fox jumps over the lazy dog "
    raw_bytes_len = len(frequent_text.encode("utf-8"))
    compressed_len = len(tok.encode(frequent_text))
    check(
        "高频文本编码后 token 数 < 原始字节数（合并生效）",
        compressed_len < raw_bytes_len,
        f"compressed={compressed_len}, raw={raw_bytes_len}",
    )

    empty_ids = tok.encode("")
    check("空字符串 encode 返回空列表", empty_ids == [])
    check("空列表 decode 返回空字符串", tok.decode([]) == "")


def main():
    print("=" * 60)
    print("Lab 01: Tokenizer 自测")
    print("=" * 60)
    test_count_pairs()
    test_apply_merge()
    test_training_and_vocab()
    test_roundtrip_and_compression()
    print("=" * 60)
    print(f"通过 {_passed} / {_passed + _failed}")
    if _failed == 0:
        print("✅ Lab 01 全部通过，可以进入 lab02_embedding")
    else:
        print("❌ 还有未通过的测试，检查 tokenizer_lab.py 里对应的 TODO")


if __name__ == "__main__":
    main()
