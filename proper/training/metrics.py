from __future__ import annotations

import torch


def dice_score(logits: torch.Tensor, target: torch.Tensor, foreground_class: int = 1, smooth: float = 1e-6) -> float:
    pred = torch.argmax(logits, dim=1) == foreground_class
    target = target == foreground_class
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * (pred & target).sum().float() + smooth) / (denom.float() + smooth))


def soft_dice_score(predictions: torch.Tensor, annotations: torch.Tensor, smooth: float = 1e-6) -> float:
    probs = torch.softmax(predictions, dim=2)[:, :, 1]
    target = annotations.float()
    inter = (probs * target).sum()
    denom = probs.sum() + target.sum()
    return float((2.0 * inter + smooth) / (denom + smooth))


def generalized_energy_distance(predictions: torch.Tensor, annotations: torch.Tensor) -> float:
    probs = torch.softmax(predictions, dim=2)[:, :, 1]
    ann = annotations.float()
    pred_dist = torch.cdist(probs.flatten(2), probs.flatten(2)).mean()
    ann_dist = torch.cdist(ann.flatten(2), ann.flatten(2)).mean()
    cross = torch.cdist(probs.flatten(2), ann.flatten(2)).mean()
    return float(2.0 * cross - pred_dist - ann_dist)
