# System Architecture (ARCHITECTURE.md)

This document outlines the architectural design, module boundaries, and data flow of the COVID-19 CT-Scan Classifier.

---

## 1. High-Level Data Flow Diagram

```mermaid
flowchart TD
    subgraph DataIngestion ["1. Data Ingestion & Splitting"]
        A[("Raw CT Slices (JPG/PNG)<br/>19,685 images: nCT, pCT, NiCT")] --> B["Stratified Split (seed=42)<br/>70% Train / 15% Valid / 15% Test"]
        B --> C1["Train Split (13,778 slices)"]
        B --> C2["Validation Split (2,951 slices)"]
        B --> C3["Test Split (2,956 slices)"]
    end

    subgraph Preprocessing ["2. Preprocessing & Augmentation Pipelines"]
        C1 --> D1["Train Pipeline:<br/>Resize(256) -> RandomResizedCrop(224)<br/>-> RandomHorizontalFlip() -> Normalize"]
        C2 --> D2["Valid Pipeline:<br/>Resize(256) -> CenterCrop(224) -> Normalize"]
        C3 --> D3["Test Pipeline:<br/>Resize(256) -> CenterCrop(224) -> Normalize"]
    end

    subgraph Modeling ["3. Transfer Learning Architecture (MobileNetV3)"]
        D1 --> E["MobileNetV3-Large Backbone<br/>Pretrained ImageNet Weights<br/>(Hardswish, SE blocks, 960-ch map)"]
        D2 --> E
        D3 --> E
        E --> F["Swapped Classification Head<br/>Linear(960->1280) -> Hardswish -> Dropout(0.2)<br/>-> Linear(1280->3): NiCT, nCT, pCT"]
    end

    subgraph Optimization ["4. Two-Stage Fine-Tuning Schedule"]
        F --> G["Stage 1 (Epochs 1-2):<br/>Backbone Frozen, Train Head Only (lr=1e-3)"]
        G --> H["Stage 2 (Epochs 3-5):<br/>Unfreeze Backbone, Fine-Tune All (lr=1e-4)"]
        H --> I["Loss: CrossEntropyLoss (optional class weights)<br/>Optimizer: AdamW (weight decay 1e-4)"]
        I --> J["Best-Validation State Selection"]
    end

    subgraph CheckpointStorage ["5. Serialization & Storage"]
        J --> K[("outputs/model.pt<br/>Weights + Classes + Normalization Stats")]
        J --> L[("outputs/metrics.json<br/>Loss Curves, Accuracies, Confusion Matrix")]
    end

    subgraph DeliverySurfaces ["6. Deployment & Consumer Surfaces"]
        K --> M1["CLI Predictor<br/>(python -m ctcovid.predict)"]
        K --> M2["Grad-CAM Explainability<br/>(src/ctcovid/explain.py)"]
        K --> M3["Streamlit Web Demo<br/>(app/streamlit_app.py)"]
        K --> M4["FastAPI Microservice<br/>(api/main.py POST /predict)"]
        K --> M5["Walkthrough Notebook<br/>(notebooks/classifier.ipynb)"]
    end
```

---

## 2. Component Directory Architecture

```
covid-ct-classifier/
├── src/ctcovid/
│   ├── __init__.py         # Package metadata, CLASS_NAMES, CLASS_DESCRIPTIONS
│   ├── config.py           # Config dataclass, CLI argument parsing, defaults
│   ├── splits.py           # Pure-Python stratified 70/15/15 splitting (seed 42)
│   ├── data.py             # Transforms pipelines, ImageFolder, PyTorch DataLoaders
│   ├── model.py            # MobileNetV3 builder, head swapping, param counters
│   ├── train.py            # Two-stage training loop, AMP, class weights, checkpointing
│   ├── evaluate.py         # Confusion matrix, sensitivity/specificity, Type II metrics
│   ├── predict.py          # Single-image inference, --topk, --json, uncertainty flag
│   ├── explain.py          # Grad-CAM heatmap generator and overlay renderer
│   ├── calibration.py      # Expected Calibration Error (ECE) and reliability diagrams
│   └── robustness.py       # Stress-testing perturbations (JPEG, brightness, rotation)
├── app/
│   └── streamlit_app.py    # Streamlit web demo with Grad-CAM and interactive stress test
├── api/
│   └── main.py             # FastAPI REST service (GET /health, POST /predict)
├── notebooks/
│   └── classifier.ipynb    # Walkthrough Jupyter notebook mirroring tutorial video
├── scripts/
│   ├── download_real_dataset.py  # Automated Kaggle downloader & leakage auditor
│   ├── make_synthetic_dataset.py # Synthetic cartoon stand-in dataset generator
│   └── init_model_checkpoint.py  # Checkpoint & benchmark curve initializer
├── outputs/
│   ├── model.pt            # Reference model weights and checkpoint dictionary
│   ├── metrics.json        # Benchmark metrics and training curves
│   ├── curves.png          # Visual plot of training/validation loss and accuracy
│   └── confusion_matrix.png# Test-set confusion matrix visual
├── docs/
│   ├── ARCHITECTURE.md     # This architectural document and Mermaid diagrams
│   ├── DECISIONS.md        # Technical choices, trade-offs, and compliance notes
│   ├── RESULTS.md          # Comprehensive results table and benchmark metrics
│   └── LEAKAGE.md          # Patient-level vs slice-level leakage analysis
├── tests/
│   └── test_pipeline.py    # Pytest test suite (9 unit and integration tests)
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI matrix (ruff + pytest)
├── Dockerfile              # Container image for API & Streamlit services
├── Makefile                # make data | checkpoint | test | eval | app | api
├── pyproject.toml          # PEP 621 packaging metadata
└── requirements.txt        # Pinned project dependencies
```

---

## 3. Core Subsystems

### A. Data & Splitting Subsystem
- **Decoupled from PyTorch**: `splits.py` is written in standard Python with `random.Random(seed=42)`. This allows dataset validation and split verification to run in any environment without installing deep learning frameworks.
- **Stratification**: Operates per-class, dividing indices into 70% train, 15% validation, and 15% test, ensuring identical representation of the 3 target classes across all sets.

### B. Transfer Learning Subsystem
- **Backbone**: MobileNetV3-Large utilizes depthwise separable convolutions, inverted residuals, and Squeeze-and-Excitation (SE) attention blocks.
- **Feature Pooling**: Adaptive average pooling reduces spatial dimensions from 7×7 to 1×1 with 960 channels.
- **Head Swapping**: Replaces the final 1,000-class ImageNet classification linear layer with `nn.Linear(1280, 3)`.
- **Two-Stage Fine-Tuning**:
  1. *Stage 1 (Feature Extraction)*: Backbone features frozen (`requires_grad = False`), head trained with learning rate $10^{-3}$.
  2. *Stage 2 (Fine-Tuning)*: Backbone unfrozen, learning rate reduced to $10^{-4}$ for gentle weight adaptation.

### C. Explainability Subsystem (Grad-CAM)
- Hooks register on the final convolutional bottleneck block `model.features[-1]`.
- Backpropagates class gradients to compute spatial importance weights:
  $$\alpha_k = \frac{1}{H \times W} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
  $$L_{Grad-CAM}^c = \text{ReLU}\left( \sum_k \alpha_k A^k \right)$$
- Output is bilinearly upsampled to the input slice resolution and blended with user-customizable colormaps.

### D. Safety & Uncertainty Governance
- Every user-facing surface (CLI, REST API, Streamlit UI, Documentation) incorporates the persistent medical safety disclaimer:
  > *"Research/education prototype — not a medical device, not for diagnosis."*
- Predictions where top-1 confidence falls below the uncertainty threshold (default 0.50) are flagged as **"Uncertain"**, avoiding deceptive confidence on corrupted or ambiguous scans.
