from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PromptProjector(nn.Module):
    def __init__(self, feature_channels: int, prompt_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_channels, prompt_dim, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).flatten(1)


class PromptBank(nn.Module):
    """Dynamic least-similar prompt-bank for rater-specific memory."""

    def __init__(self, num_raters: int, prompt_dim: int, initial_threshold: float = 0.95):
        super().__init__()
        self.num_raters = num_raters
        self.prompt_dim = prompt_dim
        self.register_buffer("prompts", F.normalize(torch.randn(num_raters, prompt_dim), dim=-1))
        self.register_buffer("thresholds", torch.full((num_raters,), initial_threshold))
        self.initialized = False

    @torch.no_grad()
    def update(self, new_prompts: torch.Tensor) -> None:
        new_prompts = F.normalize(new_prompts.detach().mean(dim=0), dim=-1)
        if not self.initialized:
            self.prompts.copy_(new_prompts)
            self.thresholds.fill_(0.95)
            self.initialized = True
            return

        for i in range(self.num_raters):
            others = [j for j in range(self.num_raters) if j != i]
            similarity = F.cosine_similarity(new_prompts[i].unsqueeze(0), self.prompts[others], dim=-1).min()
            if similarity < self.thresholds[i]:
                self.prompts[i].copy_(new_prompts[i])
                self.thresholds[i] = similarity

    def forward(self) -> torch.Tensor:
        return F.normalize(self.prompts, dim=-1)
