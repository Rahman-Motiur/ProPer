# ProPer PyTorch

PyTorch implementation of **ProPer: Prompt-aware Adaptive Personalization for Multi-rater Medical Image Segmentation**.

ProPer is a single-stage personalized segmentation framework for multi-rater medical images. It learns both annotation diversity and rater-specific preferences using:

- **Reference-guided diversification**: a reference encoder-decoder receives the image plus all rater annotations and supervises the main image-only encoder-decoder.
- **Dynamic rater prompt-bank**: rater prompts are generated from reference-decoded features and updated with least-similar prompt replacement.
- **Prompt-Aware Attention Mapping (PAAM)**: rater-specific query prompts attend to the prompt-bank to produce personalized feature conditioning.
- **Personalized, mean, and diversification losses**: supervise rater outputs, consensus behavior, and annotation diversity.

The repository includes a lightweight runnable implementation with professional hooks for LIDC-IDRI, RIGA, or other multi-rater segmentation datasets.

## Repository Layout

```text
proper-pytorch/
  proper/
    data/              CSV dataset loader for image and multi-rater masks
    models/            ProPer model, prompt-bank, PAAM, encoder-decoder blocks
    training/          Losses and metrics
    utils/             Reproducibility helpers
  configs/             Example experiment configs
  scripts/             Demo and training commands
  tests/               Forward and loss tests
```

## Install

```bash
git clone https://github.com/<your-user>/ProPer.git
cd ProPer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Demo Forward Pass

```bash
python scripts/demo_proper_forward.py
```

## Train

Prepare a CSV file:

```csv
image,rater_1,rater_2,rater_3,rater_4
data/images/case_001.npy,data/masks/case_001_r1.npy,data/masks/case_001_r2.npy,data/masks/case_001_r3.npy,data/masks/case_001_r4.npy
```

Run:

```bash
python scripts/train_proper.py --config configs/proper_lidc_idri.yaml
```

## Paper-to-Code Mapping

| Paper component | Code |
| --- | --- |
| Main/reference encoders and decoders | `proper/models/encoder_decoder.py` |
| Prompt-bank generation and least-similar update | `proper/models/prompt_bank.py` |
| Prompt-Aware Attention Mapping (PAAM) | `proper/models/paam.py` |
| ProPer forward workflow | `proper/models/proper.py` |
| KL, personalized, mean, and diversification losses | `proper/training/losses.py` |
| Dice, ASSD-style placeholders, GED, soft Dice | `proper/training/metrics.py` |

## Notes

During training, ProPer uses both reference and main pathways. During inference, only the image-only main pathway, prompt-bank, PAAM, and shared segmentation head are needed.
