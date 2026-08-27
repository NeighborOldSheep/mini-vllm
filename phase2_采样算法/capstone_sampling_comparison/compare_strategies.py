"""Phase 2 capstone：在同一分布下比较采样策略。"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ENGINE_DIR = Path(__file__).resolve().parents[1] / "lab02_sampling_engine"
sys.path.insert(0, str(ENGINE_DIR))
from sampling_engine_lab import SamplingEngine


class FixedDistributionModel(nn.Module):
    """每一步给出相同的、非均匀的 6-token logits 分布。"""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
        return base.repeat(len(input_ids), 1)


def main() -> None:
    engine = SamplingEngine(FixedDistributionModel())
    prompt = torch.tensor([0])
    settings = [
        ("greedy", dict(temperature=0.0)),
        ("temperature=0.7", dict(temperature=0.7)),
        ("temperature=1.8", dict(temperature=1.8)),
        ("top-k=2", dict(temperature=1.0, top_k=2)),
        ("top-p=0.75", dict(temperature=1.0, top_p=0.75)),
        ("gumbel-max", dict(temperature=1.0, use_gumbel=True)),
    ]
    print("固定 logits: [4, 3, 2, 1, 0, -1]\n")
    for name, kwargs in settings:
        torch.manual_seed(42)
        output = engine.generate(prompt, max_new_tokens=10, **kwargs)
        print(f"{name:>16}: {output[1:].tolist()}")
    print("\n✅ Phase 2 capstone 跑通。下一步：step07 的单请求 KV Cache。")


if __name__ == "__main__":
    main()
