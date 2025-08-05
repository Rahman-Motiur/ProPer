# ProPer: Prompt-aware Adaptive Personalization for Multi-rater Medical Image Segmentation

![ProPer Architecture](assets/proper_architecture.png)

**Authors**: Md Motiur Rahman, Saeka Rahman, Smriti Bhatt, Miad Faezipour  
**Published in**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025 (Submitted)  
**Paper**: [Link to paper (TBD)](https://arxiv.org/abs/XXXX.XXXXX)  
**Code**: Coming soon on [GitHub](https://github.com/XXXX.XXXXX)

---

## Overview

**ProPer** is a novel **single-stage, end-to-end deep learning model** for medical image segmentation that handles multi-rater annotation ambiguity using:

- Reference-Guided Diversification
- Prompt-Bank for Rater Expertise
- Prompt-Aware Attention Mapping (PAAM)
- Efficient Segmentation with Shared Heads

Unlike existing approaches that struggle with either inefficiency (two-stage) or lack of personalization (one-stage), ProPer effectively combines both diversification and personalization to deliver **state-of-the-art results**.

---

## Key Features

- **Dual Encoder-Decoder** setup to learn expert annotation variance.
- **Dynamic Prompt-Bank** to store and update rater-specific styles.
- **PAAM Module** for aligning prompts with feature space.
- **Shared Segmentation Head** for scalable personalization.
- **KL Loss Regularization** for effective supervision without collapse.

---

## Datasets Used

| Dataset     | Modality | Annotations | Tasks              |
|-------------|----------|-------------|--------------------|
| LIDC-IDRI   | CT       | 4 Raters    | Lung Nodule Segmentation |
| RIGA        | Fundus   | 6 Raters    | Optic Cup & Disc Segmentation |
| QUBIQ       | MRI/CT   | 3–7 Raters  | Brain, Kidney, Prostate Segmentation |

---

## Performance Highlights

| Dataset     | Metric           | ProPer (Ours) | Previous Best (D-Persona) |
|-------------|------------------|---------------|----------------------------|
| **LIDC-IDRI** | Dice (%)         | **91.90**      | 89.17                     |
|             | GED (↓)          | **0.1289**     | 0.1444                    |
|             | Dicesoft (↑)     | **92.65**      | 90.31                     |
| **RIGA**     | Dice (Disc, Cup) | **97.68, 86.86** | 95.89, 84.40            |
|             | ASSD (↓)         | **0.77, 4.02** | 1.14, 4.14                |

---

## Research Questions Answered

- **RQ1**: Can end-to-end modeling improve personalization? → ✅ Yes  
- **RQ2**: Does the reference encoder-decoder help? → ✅ Yes  
- **RQ3**: Do PAAM & prompt-bank updates help? → ✅ Yes  
- **RQ4**: Can ProPer outperform crowdsourcing methods? → ✅ Yes

---

## Ablation & Generalization

- Ablation confirms each module (KL loss, PAAM, λ-weighting) improves performance.
- Generalizes well across **new raters** and **unseen modalities** (QUBIQ).
- Robust against **annotation noise** (simulated poisoned raters).

---

## Implementation Details

- **Framework**: PyTorch  
- **Backbone**: ResNet18 encoder-decoder  
- **Input Sizes**:
  - LIDC-IDRI: 128×128
  - RIGA: 224×224  
- **Training**:
  - Optimizer: Adam
  - LR: 1e-4
  - Scheduler: ReduceLROnPlateau
  - Early stopping with patience = 20

---


### Project Structure (when code is released)

```text
ProPer/
│
├── model/                        # Core modules of the ProPer model
│   ├── encoder.py               # ResNet18-based encoder with KL regularization
│   ├── decoder.py               # Simple conv decoder for reference/main pathway
│   ├── paam.py                  # Prompt-aware attention mapping
│   └── prompt_bank.py           # Dynamic prompt-bank and update mechanism
│
├── datasets/                    # Dataset wrappers and loaders
│   ├── lidc.py                  # LIDC-IDRI dataset preprocessor/loader
│   └── riga.py                  # RIGA dataset preprocessor/loader
│
├── train.py                     # Model training script
├── test.py                      # Model evaluation and inference
├── utils.py                     # Loss functions, metrics, visualization
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview and usage
```
## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ProPer.git
cd ProPer
```
### 2. Create environment and install dependencies
conda create -n proper python=3.8
conda activate proper
pip install -r requirements.txt

### 3. Training
python train.py --dataset lidc --config configs/lidc_config.yaml

### 4. Testing
python test.py --checkpoint checkpoints/proper_lidc.pth --dataset lidc

## Citation
```text
@article{rahman2025proper,
  title={Prompt-aware Adaptive Personalization for Multi-rater Medical Image Segmentation},
  author={Rahman, Md Motiur and Rahman, Saeka and Bhatt, Smriti and Faezipour, Miad},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}



