"""
Lab 02 自测脚本。完成 embedding_lab.py / train_lab.py 里的全部 TODO 后运行：
    python test_embedding_lab.py
"""

import torch
from embedding_lab import Embedding, cosine_similarity, most_similar
from train_lab import build_toy_model, train_step

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


def test_forward_shape():
    torch.manual_seed(0)
    emb = Embedding(vocab_size=256, d_model=8)

    vec = emb(torch.tensor([65]))
    check("单 token 输出 shape 正确", vec.shape == (1, 8), f"got {vec.shape}")

    ids = torch.tensor([72, 101, 108, 108, 111])
    vecs = emb(ids)
    check("批量 token 输出 shape 正确", vecs.shape == (5, 8), f"got {vecs.shape}")

    check(
        "查表结果与直接索引 weight 一致",
        torch.allclose(vecs, emb.weight[ids]),
    )


def test_cosine_similarity():
    torch.manual_seed(0)
    emb = Embedding(vocab_size=256, d_model=8)

    sim_self = cosine_similarity(emb(torch.tensor([65])), emb(torch.tensor([65])))
    check("自相似度 ≈ 1.0", abs(sim_self - 1.0) < 1e-5, f"got {sim_self}")

    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    sim_orth = cosine_similarity(a, b)
    check("正交向量相似度 ≈ 0", abs(sim_orth - 0.0) < 1e-5, f"got {sim_orth}")

    c = torch.tensor([-1.0, 0.0])
    sim_opp = cosine_similarity(a, c)
    check("反向向量相似度 ≈ -1", abs(sim_opp - (-1.0)) < 1e-5, f"got {sim_opp}")


def test_most_similar():
    # 人工构造已知答案的小词表：token0 和 query 方向一致，token2 次相似，
    # token1 正交，token3 完全相反。
    weight = torch.tensor([
        [1.0, 0.0],   # token 0: 与 query 完全同向
        [0.0, 1.0],   # token 1: 正交
        [0.9, 0.1],   # token 2: 接近同向，次相似
        [-1.0, 0.0],  # token 3: 完全相反
    ])
    query = torch.tensor([1.0, 0.0])

    top2 = most_similar(query, weight, top_k=2)
    top2_list = top2.tolist()
    check(
        "most_similar 找出真正最相似的 top-2（按相似度降序）",
        top2_list == [0, 2],
        f"got {top2_list}, expected [0, 2]",
    )

    top1 = most_similar(query, weight, top_k=1)
    check("most_similar top_k=1 只返回最相似的那个", top1.tolist() == [0], f"got {top1.tolist()}")


def test_gradient_local_update():
    torch.manual_seed(42)
    vocab_size, d_model = 100, 16
    emb, head, optimizer, loss_fn = build_toy_model(vocab_size, d_model)

    before = emb.weight.detach().clone()

    x = torch.tensor([3, 7, 42])
    y = torch.tensor([4, 8, 43])
    loss = train_step(emb, head, x, y, optimizer, loss_fn)
    check("train_step 返回一个数值 loss", isinstance(loss, float), f"got {type(loss)}")

    after = emb.weight.detach()

    touched_ids = [3, 7, 42]
    untouched_ids = [0, 1, 50, 99]

    all_touched_changed = all(
        not torch.allclose(before[i], after[i]) for i in touched_ids
    )
    check("出现在 batch 中的 token 行权重发生了更新", all_touched_changed)

    all_untouched_same = all(
        torch.allclose(before[i], after[i]) for i in untouched_ids
    )
    check("未出现在 batch 中的 token 行权重完全不变（稀疏梯度）", all_untouched_same)


def main():
    print("=" * 60)
    print("Lab 02: Embedding 自测")
    print("=" * 60)
    test_forward_shape()
    test_cosine_similarity()
    test_most_similar()
    test_gradient_local_update()
    print("=" * 60)
    print(f"通过 {_passed} / {_passed + _failed}")
    if _failed == 0:
        print("✅ Lab 02 全部通过，可以进入 lab03_attention")
    else:
        print("❌ 还有未通过的测试，检查 embedding_lab.py / train_lab.py 里对应的 TODO")


if __name__ == "__main__":
    main()
