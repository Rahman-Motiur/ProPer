import torch

from proper import ProPer, ProPerConfig
from proper.training import ProPerLoss


def test_proper_forward_and_loss():
    model = ProPer(ProPerConfig(image_channels=1, num_classes=2, num_raters=3, base_channels=8, prompt_dim=16, latent_dim=16))
    images = torch.randn(2, 1, 32, 32)
    annotations = torch.randint(0, 2, (2, 3, 32, 32))
    outputs = model(images, annotations, update_prompt_bank=True)
    assert outputs["logits"].shape == (2, 3, 2, 32, 32)
    losses = ProPerLoss()(outputs, annotations)
    assert torch.isfinite(losses["loss"])
