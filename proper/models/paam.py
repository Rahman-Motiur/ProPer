from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PromptAwareAttentionMapping(nn.Module):
    def __init__(self, prompt_dim: int, feature_channels: int):
        super().__init__()
        self.output = nn.Linear(prompt_dim, feature_channels)

    def forward(self, query_prompts: torch.Tensor, prompt_bank: torch.Tensor, spatial_size: tuple[int, int]) -> torch.Tensor:
        q = F.normalize(query_prompts, dim=-1)
        k = F.normalize(prompt_bank, dim=-1)
        attn = torch.softmax((q @ k.T) / math.sqrt(q.shape[-1]), dim=-1)
        mapped = attn @ prompt_bank
        mapped = self.output(mapped)
        return mapped[:, :, None, None].expand(-1, -1, spatial_size[0], spatial_size[1])
