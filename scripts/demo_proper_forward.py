from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from proper import ProPer, ProPerConfig
from proper.training import ProPerLoss, ProPerLossConfig


def main() -> None:
    config = ProPerConfig(image_channels=1, num_classes=2, num_raters=4, base_channels=16, prompt_dim=32, latent_dim=32)
    model = ProPer(config)
    images = torch.randn(2, 1, 64, 64)
    annotations = torch.randint(0, 2, (2, 4, 64, 64))
    outputs = model(images, annotations, update_prompt_bank=True)
    losses = ProPerLoss(ProPerLossConfig())(outputs, annotations)
    print("personalized logits:", tuple(outputs["logits"].shape))
    print("prompt bank:", tuple(outputs["prompt_bank"].shape))
    print("loss:", float(losses["loss"].detach()))


if __name__ == "__main__":
    main()
