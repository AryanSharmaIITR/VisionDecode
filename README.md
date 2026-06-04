<div align="center">

# 🔍 VisionDecode

### Distorted Visual Sequence Pattern Recognition

*Reading the unreadable — reconstructing ordered character sequences from noisy, distorted grayscale images.*

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/vision/)
[![timm](https://img.shields.io/badge/timm-ViT-blue)](https://github.com/huggingface/pytorch-image-models)
[![Best Acc](https://img.shields.io/badge/Full--code%20Acc-97.40%25-brightgreen)](#-results)
[![License](https://img.shields.io/badge/License-See%20LICENSE-green)](./LICENSE)

**📦 [Dataset + Pretrained Artifacts (Google Drive)](https://drive.google.com/drive/folders/1BmYQBdOoLxwWSHTTVDNipFGZD1N2IcfF?usp=sharing)**

</div>

---

## 📖 Overview

**VisionDecode** tackles the CIG AI/ML challenge **PS-1: Distorted Visual Sequence Pattern Recognition** — given a distorted grayscale image containing a fixed-length character code, predict the exact ordered sequence. The images are deliberately corrupted with **blur, occlusion, overlapping characters, warping, and visual noise**, making naive OCR useless.

The solution is a full **model zoo** (CNNs, CRNNs, and Vision Transformers) culminating in a **weighted soft-voting ensemble** that blends the three strongest models — reaching **97.40% full-code accuracy** on the validation split.

- **Task** — image → ordered 6-character sequence
- **Charset** — `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` 
- **Code length** — 6 characters
- **Metric** — **Character Error Rate (CER)**, Levenshtein-based (lower is better)

---

## 📊 Dataset

| Split | Count | Notes |
|-------|------:|-------|
| Train | 20,000 | grayscale `.png` + `train-labels.csv` (`image, text`) |
| Test  | 5,000  | grayscale `.png`, no labels |

> 💾 **The dataset and all pretrained model checkpoints are hosted on Google Drive (the repo ships code, not weights/images):**
>
> ### 👉 [**Download Dataset + Artifacts**](https://drive.google.com/drive/folders/1BmYQBdOoLxwWSHTTVDNipFGZD1N2IcfF?usp=sharing)
>
> After downloading, place the contents so the layout matches the [Project Structure](#-project-structure) below:
> - `data/train_images/`, `data/test_images/`, `data/train-labels.csv`
> - `Artifacts/*.pth`

---

## 🧠 Model Zoo

All models share a common training/eval pipeline (charset encoding, CER, dataloaders, classifier trainer) and are implemented in [`NoteBook/prototype_model.ipynb`](./NoteBook/prototype_model.ipynb). Every model outputs a `(B, 6, 33)` grid, so they are directly comparable.

| # | Model | Architecture | Idea |
|---|-------|--------------|------|
| 1 | **EfficientNet Baseline** | EfficientNet-B0 + 6 classification heads | One linear head per character position |
| 2 | **CRNN-EfficientNet** | EfficientNet-B0 → BiGRU → Linear | Convolutional features fed to a recurrent sequence decoder |
| 3 | **CRNN-MobileNet** | MobileNetV3-Small → BiGRU | Lightweight CRNN, fast and memory-friendly |
| 4 | **ViTCaptcha** | ViT-Small (timm) + 6 heads | Pure transformer vision backbone |
| 4b | **AdvancedViTCaptcha** | Vertical-strip tokenizer + TransformerEncoder | Crops 10px/side, treats each `30×100` strip as one token → per-token character prediction |
| 5 | **TinyCNN** | Custom CNN (<500k params) | Minimal-footprint, real-time CPU baseline |
| 6 | **Ensemble** | Weighted soft-voting | Position-wise softmax averaging across top models |
| 7 | **CRNN-ResNet** | ResNet → BiGRU | Deeper CNN backbone variant |

### 🏆 Final Ensemble

The production model ([`NoteBook/Final_Model.ipynb`](./NoteBook/Final_Model.ipynb)) is a **weighted soft-vote** over the three strongest classification models:

```
prediction = argmax( 0.25·EfficientNet  +  0.25·CRNN-BiGRU  +  0.50·CRNN-MobileNet )   # per character position
```

It is fully self-contained — it redefines the three model classes and all helpers so the checkpoints load cleanly, reports validation CER/accuracy per-model vs. the blend, and writes the final submission CSV.

---

## 🎯 Results

### Per-model — validation split (`prototype_model.ipynb`)

| # | Model | Val CER ↓ | Full-code Acc ↑ | Trainable Params |
|---|-------|----------:|----------------:|-----------------:|
| 1 | EfficientNet baseline | 0.0976 | 52.72% | 5.17M |
| 2 | CRNN-EfficientNet + BiGRU | 0.0727 | 63.73% | 6.72M |
| 3 | CRNN-MobileNet + GRU | 0.0161 | 90.69% | 1.77M |
| 4 | ViT-Small | 0.9648 | 0.00% | 4.42M |
| 4b | AdvancedViTCaptcha | — | — | ~4.4M |
| 5 | TinyCNN | 0.3740 | 2.00% | 0.25M |
| 6 | Ensemble (soft-vote, top-3) | **0.0089** | 94.80% | — |
| 7 | CRNN-ResNet + BiGRU | 0.2001 | 24.56% | 10.78M |

### 🥇 Final tuned ensemble (`Final_Model.ipynb`)

| Ensemble | Members (weights) | Full-code Acc ↑ |
|----------|-------------------|----------------:|
| **Final Weighted Soft-Vote** | EfficientNet (0.25) + CRNN-EfficientNet (0.25) + CRNN-MobileNet (0.50) | **🟢 97.40%** |

> The final ensemble — built from tuned ("Advanced") variants of Models 1–3 — is the strongest configuration and the basis for the submitted predictions. It generates **5,000 test predictions** written to [`Result/submission_AryanSharma_24113024.csv`](./Result/submission_AryanSharma_24113024.csv).

**Takeaways**
- 🪶 The **lightweight CRNN-MobileNet** is the single best model (93.69% acc, 0.0161 CER) at just **1.77M** trainable params — small *and* accurate, so it carries the largest ensemble weight.
- 🤝 **Ensembling lifts accuracy substantially** — from 90.69% (best single) → **97.40%** (tuned blend).
- 🧊 The generic **ViT-Small** struggled on this distorted-captcha distribution (data-hungry); the **custom strip-token AdvancedViTCaptcha** is the captcha-specific transformer alternative.

---

## 📁 Project Structure

```
VisionDecode/
├── data/
│   ├── train_images/              # 20,000 training images   (from Drive)
│   ├── test_images/               #  5,000 test images       (from Drive)
│   └── train-labels.csv           # image,text labels
├── Artifacts/                     # trained .pth checkpoints  (from Drive)
│   ├── model1_efficientnet.pth
│   ├── model2_crnn_bigru.pth
│   ├── model3_mobilenet_gru.pth
│   ├── model4b_advanced_vit.pth
│   ├── model7_resnet18.pth
│   ├── model_tinycnn.pth
│   └── FinalModel_*.pth           # tuned ensemble checkpoints
├── NoteBook/
│   ├── prototype_model.ipynb      # full model zoo + training pipeline
│   └── Final_Model.ipynb          # self-contained ensemble + submission
├── Result/
│   └── submission_AryanSharma_24113024.csv
├── ProjectDetails/                # challenge brief (PS-1)
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone <your-repo-url>
cd VisionDecode
pip install torch torchvision timm pandas pillow numpy
```

### 2. Download data + weights

Grab the **[Dataset + Artifacts from Google Drive](https://drive.google.com/drive/folders/1BmYQBdOoLxwWSHTTVDNipFGZD1N2IcfF?usp=sharing)** and unpack them into `data/` and `Artifacts/` as shown above.

### 3. Run

- **Train / explore the model zoo** → open [`NoteBook/prototype_model.ipynb`](./NoteBook/prototype_model.ipynb)
- **Generate the final submission** → open [`NoteBook/Final_Model.ipynb`](./NoteBook/Final_Model.ipynb), run all cells. It loads the trained checkpoints, validates the ensemble, and writes `Result/submission_<name>_<enroll>.csv`.

> 💡 **GPU note:** developed on a **4GB RTX 3050**. In Jupyter on Python 3.14, dataloaders use `num_workers=0` to avoid forkserver `BrokenPipeError`.

---

## 📤 Submission Format

A single CSV with the predicted code for every test image:

```csv
image,prediction
test-0.png,QVTQ8A
test-1.png,7PSW9D
```

---

## 🛠️ Tech Stack

`PyTorch` · `torchvision` (EfficientNet-B0, MobileNetV3, ResNet) · `timm` (ViT) · `pandas` · `Pillow` · `NumPy`

---

<div align="center">

*Built for the CIG AI/ML Challenge — PS-1.* ⚡

</div>
