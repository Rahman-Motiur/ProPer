from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from proper import ProPer, ProPerConfig
from proper.data import MultiRaterSegmentationDataset
from proper.training import ProPerLoss, ProPerLossConfig
from proper.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42))
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MultiRaterSegmentationDataset(
        cfg["data"]["train_csv"],
        tuple(cfg["data"]["image_size"]),
        cfg["data"]["rater_columns"],
    )
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"])
    model = ProPer(ProPerConfig(**cfg["model"])).to(device)
    loss_keys = ProPerLossConfig.__dataclass_fields__.keys()
    criterion = ProPerLoss(ProPerLossConfig(**{k: cfg["training"][k] for k in loss_keys}))
    optimizer = Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        total = 0.0
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            annotations = batch["annotations"].to(device)
            outputs = model(images, annotations, update_prompt_bank=True)
            losses = criterion(outputs, annotations)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            total += float(losses["loss"].detach())
        print(f"epoch={epoch:03d} loss={total / len(loader):.4f}")
        torch.save({"model": model.state_dict(), "config": cfg}, run_dir / "last.pt")


if __name__ == "__main__":
    main()
