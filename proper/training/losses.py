from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ProPerLossConfig:
    encoder_kl_weight: float = 1.0
    decoder_kl_weight: float = 1.0
    personalized_weight: float = 1.0
    mean_weight: float = 1.0
    diversification_weight: float = 1.0


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    target_1h = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_1h, dim=dims)
    denominator = torch.sum(probs + target_1h, dim=dims)
    return 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth)).mean()


def hybrid_segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return dice_loss(logits, target) + F.cross_entropy(logits, target.long())


def gaussian_kl(mu_ref: torch.Tensor, logvar_ref: torch.Tensor, mu_main: torch.Tensor, logvar_main: torch.Tensor) -> torch.Tensor:
    var_ref = torch.exp(logvar_ref)
    var_main = torch.exp(logvar_main)
    kl = 0.5 * (logvar_main - logvar_ref + (var_ref + (mu_ref - mu_main).pow(2)) / var_main.clamp_min(1e-6) - 1.0)
    return kl.mean()


def uncertainty_weighted_mean_loss(logits: torch.Tensor, annotations: torch.Tensor) -> torch.Tensor:
    mean_logits = logits.mean(dim=1)
    mean_annotations = annotations.float().mean(dim=1).round().long()
    probs = torch.softmax(logits, dim=2)[:, :, 1]
    uncertainty = (probs - probs.mean(dim=1, keepdim=True)).pow(2).mean(dim=1)
    pixel_loss = F.cross_entropy(mean_logits, mean_annotations, reduction="none")
    return ((1.0 + uncertainty.detach()) * pixel_loss).mean() + dice_loss(mean_logits, mean_annotations)


class ProPerLoss(nn.Module):
    def __init__(self, config: ProPerLossConfig = ProPerLossConfig()):
        super().__init__()
        self.config = config

    def forward(self, outputs: dict, annotations: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = outputs["logits"]
        personalized_terms = [
            hybrid_segmentation_loss(logits[:, i], annotations[:, i]) for i in range(annotations.shape[1])
        ]
        personalized = torch.stack(personalized_terms).mean()
        mean_loss = uncertainty_weighted_mean_loss(logits, annotations)

        enc_kl = logits.new_tensor(0.0)
        dec_kl = logits.new_tensor(0.0)
        if outputs.get("reference_encoder_dist") is not None:
            ref_mu, ref_logvar = outputs["reference_encoder_dist"]
            main_mu, main_logvar = outputs["main_encoder_dist"]
            enc_kl = gaussian_kl(ref_mu.detach(), ref_logvar.detach(), main_mu, main_logvar)
        if outputs.get("reference_decoder_dist") is not None:
            ref_mu, ref_logvar = outputs["reference_decoder_dist"]
            main_mu, main_logvar = outputs["main_decoder_dist"]
            dec_kl = gaussian_kl(ref_mu.detach(), ref_logvar.detach(), main_mu, main_logvar)

        diversification = logits.new_tensor(0.0)
        if "diversification_logits" in outputs:
            idx = outputs["diversification_rater_index"]
            diversification = hybrid_segmentation_loss(outputs["diversification_logits"], annotations[:, idx])

        total = (
            self.config.encoder_kl_weight * enc_kl
            + self.config.decoder_kl_weight * dec_kl
            + self.config.personalized_weight * personalized
            + self.config.mean_weight * mean_loss
            + self.config.diversification_weight * diversification
        )
        return {
            "loss": total,
            "encoder_kl_loss": enc_kl.detach(),
            "decoder_kl_loss": dec_kl.detach(),
            "personalized_loss": personalized.detach(),
            "mean_loss": mean_loss.detach(),
            "diversification_loss": diversification.detach(),
        }
