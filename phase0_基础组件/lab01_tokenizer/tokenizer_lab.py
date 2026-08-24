"""
Lab 01: 字节级 BPE Tokenizer（挖空版，供你自己实现）

参考: mini-vllm-tutorial/step01_tokenizer/tokenizer.py
完成所有 TODO 后运行 test_tokenizer_lab.py 自测。
"""

from collections import Counter
from typing import Dict, List, Tuple



class SimpleBPETokenizer:
    """
    字节级 BPE Tokenizer（教学用简化版）。

    词表结构：
      0~255   → 单字节 token（基础词表，覆盖所有字节）
      256+    → BPE 合并产生的多字节 token
    """

    BASE_VOCAB_SIZE = 256
    MERGE_COUNT = 256  # 演示用，真实模型通常 50000+

    def __init__(self):
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(self.BASE_VOCAB_SIZE)}
        self.merges: Dict[Tuple[int, int], int] = {}
        self._bytes_to_id: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self._train_merges()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _get_train_corpus() -> List[List[int]]:
        """训练语料（已提供，不需要修改）：文本 → 字节序列列表。"""
        train_texts = [
            "the quick brown fox jumps over the lazy dog ",
            "hello world hello world hello world ",
            "Python is great for machine learning ",
            "large language model inference engine ",
        ]
        return [list(text.encode("utf-8")) for text in train_texts]

    # ------------------------------------------------------------------
    # TODO 1
    # ------------------------------------------------------------------
    @staticmethod
    def count_pairs(corpus: List[List[int]]) -> Counter:
        """
        统计 corpus（多条 token id 序列）里所有相邻二元组 (a, b) 的出现次数。

        输入: corpus = [[116, 104, 101, 32], [116, 104, 101]]
        输出: Counter({(116,104): 2, (104,101): 2, (101,32): 1})
        """
        raise NotImplementedError("TODO 1: 实现 count_pairs")

    # ------------------------------------------------------------------
    # TODO 2
    # ------------------------------------------------------------------
    @staticmethod
    def apply_merge(seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """
        把 seq 中所有相邻且等于 pair 的两个元素替换为 new_id。

        apply_merge([116, 104, 101, 32, 116, 104], (116, 104), 256)
        → [256, 101, 32, 256]

        注意：替换后要跳过被合并的两个位置，不能重叠匹配。
        """
        raise NotImplementedError("TODO 2: 实现 apply_merge")

    # ------------------------------------------------------------------
    # TODO 3
    # ------------------------------------------------------------------
    def _train_merges(self):
        """
        用 count_pairs + apply_merge 实现完整训练循环：

        corpus = self._get_train_corpus()
        next_id = self.BASE_VOCAB_SIZE
        for _ in range(self.MERGE_COUNT):
            1. 统计 corpus 中所有相邻 pair 的频率
            2. 若没有任何 pair，提前 break
            3. 取频率最高的 pair（Counter.most_common(1)）
            4. new_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
            5. 更新 self.vocab[next_id] = new_bytes
               更新 self._bytes_to_id[new_bytes] = next_id
               更新 self.merges[pair] = next_id
            6. 用 apply_merge 把这个 pair 在整个 corpus 里替换掉
            7. next_id += 1
        """
        raise NotImplementedError("TODO 3: 实现 _train_merges")

    # ------------------------------------------------------------------
    # TODO 4
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 token id 列表。

        1. text.encode("utf-8") 得到字节序列，转成 List[int]（每字节一个 base token）
        2. 按 self.merges 的插入顺序依次调用 apply_merge
        """
        raise NotImplementedError("TODO 4: 实现 encode")

    # ------------------------------------------------------------------
    # TODO 5
    # ------------------------------------------------------------------
    def decode(self, ids: List[int]) -> str:
        """
        将 token id 列表解码为文本。

        1. 用 self.vocab 把每个 id 查回 bytes
        2. b"".join(...) 拼接
        3. .decode("utf-8", errors="replace")
        """
        raise NotImplementedError("TODO 5: 实现 decode")
