from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .encoder_decoder import Decoder, Encoder
from .paam import PromptAwareAttentionMapping
from .prompt_bank import PromptBank, PromptProjector


@dataclass
class ProPerConfig:
    image_channels: int = 1
    num_classes: int = 2
    num_raters: int = 4
    base_channels: int = 32
    prompt_dim: int = 64
    latent_dim: int = 64
    initial_prompt_threshold: float = 0.95


class ProPer(nn.Module):
    def __init__(self, config: ProPerConfig):
        super().__init__()
        self.config = config
        ref_in = config.image_channels + config.num_raters
        feature_channels = config.base_channels * 4
        decoded_channels = config.base_channels

        self.reference_encoder = Encoder(ref_in, config.base_channels, config.latent_dim)
        self.main_encoder = Encoder(config.image_channels, config.base_channels, config.latent_dim)
        self.reference_decoder = Decoder(feature_channels, decoded_channels, config.latent_dim)
        self.main_decoder = Decoder(feature_channels, decoded_channels, config.latent_dim)

        self.reference_prompt_heads = nn.ModuleList(
            [PromptProjector(decoded_channels, config.prompt_dim) for _ in range(config.num_raters)]
        )
        self.main_prompt_heads = nn.ModuleList(
            [PromptProjector(decoded_channels, config.prompt_dim) for _ in range(config.num_raters)]
        )
        self.prompt_bank = PromptBank(config.num_raters, config.prompt_dim, config.initial_prompt_threshold)
        self.paam = PromptAwareAttentionMapping(config.prompt_dim, decoded_channels)
        self.shared_head = nn.Sequential(
            nn.Conv2d(decoded_channels * 2, decoded_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoded_channels, config.num_classes, 1),
        )
        self.reference_head = nn.Sequential(
            nn.Conv2d(decoded_channels * 2, decoded_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoded_channels, config.num_classes, 1),
        )

    def _prompt_stack(self, heads: nn.ModuleList, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(features) for head in heads], dim=1)

    def forward(self, images: torch.Tensor, annotations: torch.Tensor | None = None, update_prompt_bank: bool = False) -> dict:
        b, _, h, w = images.shape
        main_features, main_mu_e, main_logvar_e = self.main_encoder(images)
        main_decoded, main_mu_d, main_logvar_d = self.main_decoder(main_features, (h, w))
        main_prompts = self._prompt_stack(self.main_prompt_heads, main_decoded)

        if annotations is not None:
            reference_input = torch.cat([images, annotations.float()], dim=1)
            ref_features, ref_mu_e, ref_logvar_e = self.reference_encoder(reference_input)
            ref_decoded, ref_mu_d, ref_logvar_d = self.reference_decoder(ref_features, (h, w))
            ref_prompts = self._prompt_stack(self.reference_prompt_heads, ref_decoded)
            if update_prompt_bank:
                self.prompt_bank.update(ref_prompts)
        else:
            ref_decoded = ref_prompts = ref_mu_e = ref_logvar_e = ref_mu_d = ref_logvar_d = None

        bank = self.prompt_bank()
        personalized_logits = []
        mappings = []
        for rater_idx in range(self.config.num_raters):
            mapping = self.paam(main_prompts[:, rater_idx], bank, (h, w))
            mappings.append(mapping)
            conditioned = torch.cat([main_decoded, mapping], dim=1)
            personalized_logits.append(self.shared_head(conditioned))
        logits = torch.stack(personalized_logits, dim=1)

        output = {
            "logits": logits,
            "main_prompts": main_prompts,
            "prompt_bank": bank,
            "main_encoder_dist": (main_mu_e, main_logvar_e),
            "main_decoder_dist": (main_mu_d, main_logvar_d),
            "reference_prompts": ref_prompts,
            "reference_decoded": ref_decoded,
            "reference_encoder_dist": (ref_mu_e, ref_logvar_e) if ref_mu_e is not None else None,
            "reference_decoder_dist": (ref_mu_d, ref_logvar_d) if ref_mu_d is not None else None,
        }

        if ref_decoded is not None:
            ref_idx = torch.randint(0, self.config.num_raters, (1,), device=images.device).item()
            ref_mapping = self.paam(ref_prompts[:, ref_idx], bank, (h, w))
            output["diversification_logits"] = self.reference_head(torch.cat([ref_decoded, ref_mapping], dim=1))
            output["diversification_rater_index"] = ref_idx
        return output
