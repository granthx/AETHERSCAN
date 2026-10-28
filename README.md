# COVID-19 CT-Scan Image Classifier

[![CI](https://github.com/example/covid-ct-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/example/covid-ct-classifier/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ### ⚠️ MEDICAL SAFETY NOTICE
> **Research and education prototype only.** This software is **NOT a medical device** and is **NOT intended for clinical diagnosis, patient screening, or triage**. Chest CT is not a primary screening tool for COVID-19.

---

## 1. Overview

This repository is a production-grade, reproducible implementation of a **COVID-19 chest CT-scan image classifier** using **transfer learning with MobileNetV3-Large (PyTorch)**, reconstructing the workflow demonstrated in the popular tutorial video.

- **Input:** A single 2D axial chest CT slice (`.jpg` / `.png`).
- **Output:** One of 3 classes with calibrated probabilities:
  - **`nCT`**: Normal CT (no findings).
  - **`pCT`**: COVID-19 positive (ground-glass opacities / consolidation).
  - **`NiCT`**: Non-informative slice (cuts outside lung parenchyma, e.g., neck or abdomen).
- **Target Video Quality:** **99.51% test accuracy** on the Kaggle CT-Scan dataset.
- **Headline Safety Metric:** Minimizing **Type II Error (`pCT` false negatives = missed COVID cases)**.

---

## 3. Results & Reproduction

| Metric | Video Tutorial Benchmark (Real Kaggle Data) | Synthetic Smoke Test | Clinical Implication |
|---|:---:|:---:|---|
| **Test Accuracy** | **99.51%** (2,941 / 2,956) | 100.0% | Slices correctly categorized |
| **Validation Accuracy** | **99.40%** | 100.0% | Held-out validation split |
| **Train Accuracy** | **98.60%** | 99.1% | Feature extraction + fine-tuning |
| **Test Loss** | **0.0152** | 0.0210 | CrossEntropyLoss with AdamW |
| **pCT Sensitivity (Recall)** | **99.67%** (599 / 601) | 100.0% | Real COVID cases detected |
| **pCT False Negatives** | **2 slices** | **0 slices** | **Headline Safety Metric (Missed COVID)** |
| **nCT Specificity** | **99.79%** | 100.0% | Healthy controls cleared |

*Full evaluation metrics, confusion matrix, and calibration curves are detailed in [docs/RESULTS.md](docs/RESULTS.md).*

> **⚠️ Leakage Disclosure:** The real Kaggle dataset consists of preprocessed 2D slices without patient IDs. An image-wise 70/15/15 split leaks slices of the same patient between train and test sets, which inflates test accuracy from ~90% to 99.5%. See [docs/LEAKAGE.md](docs/LEAKAGE.md).

---

## 4. Architecture & Pipeline

```
Raw Slices ──► Stratified 70/15/15 Split (seed 42)
           ──► Transforms (Resize 256 -> RandomResizedCrop 224 -> Flips -> ImageNet Norm)
           ──► MobileNetV3-Large (Pretrained ImageNet Backbone)
           ──► Swapped Linear Classifier Head (1280 -> 3 classes)
           ──► Stage 1: Freeze Backbone, Train Head Only
               Stage 2: Unfreeze Backbone, Fine-Tune All Layers
           ──► Save Portable Checkpoint (outputs/model.pt)
           ──► Consumers: CLI, Grad-CAM, Streamlit UI, FastAPI REST
```

*Full data-flow diagram and module specifications are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).*

---

## 5. Key Features

- **Transfer Learning (MobileNetV3-Large)**: Balances low parameter count (~4.2M params) with high feature extraction capacity.
- **Two-Stage Fine-Tuning**: Feature extraction stage trains the swapped head, followed by gentle end-to-end fine-tuning.
- **Visual Explainability (Grad-CAM)**: Generates heatmaps from the final convolutional layer (`model.features[-1]`) to verify the model focuses on lung opacities rather than scanner beds or text labels.
- **Uncertainty & Calibration (ECE)**: Calculates Expected Calibration Error; automatically marks predictions with confidence below threshold as **"Uncertain"**.
- **Robustness Suite**: Evaluates accuracy under JPEG compression, brightness/exposure shifts, and ±10° patient rotation.
- **Cinematic Web Experience (`web/`)**: Awwwards-grade, futuristic medical AI research console with 3D Three.js volumetric particle simulation, dual-focus CT/Grad-CAM stage, specimen vault presets, real-time probability matrix, and scanner artifact stress-testing.
- **Production REST API (`api/main.py`)**: High-performance FastAPI endpoint (`POST /predict` and `POST /explain`) with multipart image handling and Base64 thermal heatmap generation.
- **Educational Notebook (`notebooks/classifier.ipynb`)**: Self-contained walkthrough mirroring the video's sequence.

---

## 7. Working with the Real Kaggle Dataset

1. Place your `kaggle.json` API token at `~/.kaggle/kaggle.json`.
2. Run the automated acquisition script:
   ```bash
   python scripts/download_real_dataset.py --dest data/real
   ```
3. Verify target counts: **nCT = 9,979**, **NiCT = 5,705**, **pCT = 4,001** (Total = 19,685).
4. Run leakage audit:
   ```bash
   python scripts/download_real_dataset.py --check-leakage --dest data/real
   ```

---

## 9. Ethics, Limitations & Licensing

- **License**: MIT License. See [LICENSE](LICENSE).
- **Dataset License**: Sourced from Kaggle (`azaemon/preprocessed-ct-scans-for-covid19`, derived from HUST-19/iCTCF). Slices are for educational and academic research.
- **Ethical Limitation**: CT imaging entails ionizing radiation and is not a first-line diagnostic method for COVID-19. AI tools must never replace RT-PCR or board-certified radiologist reviews.
