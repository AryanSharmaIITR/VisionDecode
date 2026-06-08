<div align="center">

# 🔍 VisionDecode

### Distorted Visual Sequence Pattern Recognition

*Reading the unreadable — reconstructing ordered character sequences from noisy, distorted grayscale images.*

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/vision/)
[![timm](https://img.shields.io/badge/timm-ViT-blue)](https://github.com/huggingface/pytorch-image-models)
[![Best Acc](https://img.shields.io/badge/Full--code%20Acc-98.85%25-brightgreen)](#-results)
[![License](https://img.shields.io/badge/License-See%20LICENSE-green)](./LICENSE)

**📦 [Dataset + Pretrained Artifacts (Google Drive)](https://drive.google.com/drive/folders/1BmYQBdOoLxwWSHTTVDNipFGZD1N2IcfF?usp=sharing)**

</div>

---

## 📖 Overview

**VisionDecode** tackles the CIG AI/ML challenge **PS-3: Distorted Visual Sequence Pattern Recognition** — given a distorted grayscale image containing a fixed-length character code, predict the exact ordered sequence. The images are deliberately corrupted with **blur, occlusion, overlapping characters, warping, and visual noise**, making naive OCR useless.

The solution is a full **model zoo** (CNNs, CRNNs, and Vision Transformers) culminating in a **weighted soft-voting ensemble** that blends three diverse custom CRNN models — reaching **98.95% full-code accuracy** (val CER **0.00258**) on the validation split.

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

The production model ([`NoteBook/Final_Model.ipynb`](./NoteBook/Final_Model.ipynb)) is a **weighted soft-vote** over three diverse **custom CRNN (from-scratch CNN backbone → BiGRU/BiLSTM)** classification models. The members differ in backbone depth and recurrence so their errors decorrelate:

| Model | Class — backbone → head | Weight | Trainable Params |
|-------|--------------------------|-------:|-----------------:|
| Model 1 | `CRNN_CustomModel_light` — CNN(512) → 2-layer **BiGRU** | 0.40 | 2.37M |
| Model 2 | `CRNN_CustomModel` — CNN(1024) → 4-layer **BiGRU** | 0.20 | 6.24M |
| Model 3 | `CRNN_CustomModel_light_lstm` — CNN(512) → 4-layer **BiLSTM** | 0.40 | 3.42M |

```
prediction = argmax( 0.4·CRNN_CustomModel_light  +  0.2·CRNN_CustomModel  +  0.4·CRNN_CustomModel_light_lstm )   # per character position
```

It is fully self-contained — it redefines the three model classes and all helpers (charset, CER, dataloaders, classifier trainer), trains/loads the checkpoints, reports validation CER/accuracy per-model vs. the blend, and writes the final submission CSV.

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

### 🥇 Final ensemble (`Final_Model.ipynb`)

Per-member validation performance (best checkpoint) and the final weighted blend:

| Model | Class — backbone → head | Weight | Val CER ↓ | Full-code Acc ↑ | Params |
|-------|--------------------------|-------:|----------:|----------------:|-------:|
| Model 1 | `CRNN_CustomModel_light` — CNN(512) → BiGRU-2 | 0.40 | 0.0051 | ~97.7% | 2.37M |
| Model 2 | `CRNN_CustomModel` — CNN(1024) → BiGRU-4 | 0.20 | 0.0128 | ~92.9% | 6.24M |
| Model 3 | `CRNN_CustomModel_light_lstm` — CNN(512) → BiLSTM-4 | 0.40 | 0.0068 | ~96.4% | 3.42M |
| **Final Weighted Soft-Vote** | **All three (0.5 / 0.1 / 0.4)** | — | **0.00258** | **🟢 98.95%** |

> The final ensemble is the strongest configuration and the basis for the submitted predictions — reaching **98.85% full-code accuracy** at a validation CER of just **0.00267**. It generates **5,000 test predictions** written to [`Result/submission_AryanSharma_24113024.csv`](./Result/submission_AryanSharma_24113024.csv).

**Takeaways**
- 🪶 All three members are **lightweight custom CRNNs** (2.4M–6.2M params) — from-scratch CNN backbones paired with BiGRU/BiLSTM sequence decoders.
- 🏅 The **`CRNN_CustomModel_light`** is the single best member (~97.7% acc, 0.0051 CER); the weights down-weight the weakest member (Model 2) to 0.20.
- 🤝 **Ensembling beats every member** — the weighted soft-vote reaches **98.85% full-code accuracy** at a character error rate of only **0.267%**, better than any model alone.

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
│   ├── CRNN_CustomModel_light.pth        # final ensemble member 1 (BiGRU-2)
│   ├── CRNN_CustomModel.pth              # final ensemble member 2 (BiGRU-4)
│   └── CRNN_CustomModel_light_lstm.pth   # final ensemble member 3 (BiLSTM-4)
├── NoteBook/
│   ├── prototype_model.ipynb      # full model zoo + training pipeline
│   └── Final_Model.ipynb          # self-contained ensemble + submission
├── Result/
│   └── submission_AryanSharma_24113024.csv
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

`PyTorch` · `torchvision` (EfficientNet-B0, MobileNetV3-Small/Large, ShuffleNetV2, ResNet) · `timm` (ViT) · `pandas` · `Pillow` · `NumPy`

---

<div align="center">

*Built for the CIG AI/ML Challenge — PS-3.* ⚡

</div>
