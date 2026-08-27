"""Phase 2 Lab：拥有可配置采样策略的朴素生成引擎。"""

import sys
from pathlib import Path

import torch
from torch import Tensor

SAMPLER_DIR = Path(__file__).resolve().parents[1] / "lab01_sampling_primitives"
sys.path.insert(0, str(SAMPLER_DIR))
from sampler_lab import (  # noqa: E402
    greedy_sample,
    gumbel_max_sample,
    temperature_sample,
    top_k_sample,
    top_p_sample,
)


class SamplingEngine:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def _select_next_token(
        self,
        logits: Tensor,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        use_gumbel: bool = False,
    ) -> Tensor:
        """根据参数分派到一种采样策略；见本目录 README 的优先级表。"""
        # TODO 1：实现完整的 if / elif 分派；每个函数返回标量 token id。
        raise NotImplementedError("TODO 1: 实现 _select_next_token")

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        use_gumbel: bool = False,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 必须 >= 0")
        # TODO 2：复用 _select_next_token。模型应每步收到完整历史；不要修改 prompt_ids。
        raise NotImplementedError("TODO 2: 实现 generate")
