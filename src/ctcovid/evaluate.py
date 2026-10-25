"""Test-set evaluation with the confusion-matrix metrics from the video.

Video: confusion matrix slide (TP/FP/FN/TN -> sensitivity, specificity,
precision, NPV, accuracy) with emphasis on *false negatives* (Type II error):
missing a real COVID case is the costly mistake.
Run: python -m ctcovid.evaluate --checkpoint outputs/model.pt --data-root data/synthetic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import make_loaders
from .model import build_model
from .train import get_device


def load_checkpoint(path, device="cpu"):
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=device)
    model = build_model(ck["arch"], ck["num_classes"], pretrained=False, freeze_backbone=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck



def per_class_metrics(cm: np.ndarray, classes: list[str]) -> dict:
    """One-vs-rest sensitivity/specificity/precision/NPV for each class."""
    total = cm.sum()
    out = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]; fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp; tn = total - tp - fn - fp
        div = lambda a, b: float(a / b) if b else 0.0
        out[c] = {
            "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "precision": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "support": int(cm[i].sum()), "false_negatives": int(fn), "false_positives": int(fp),
        }
    return out


@torch.no_grad()
def evaluate(checkpoint: str, data_root: str, out_dir: str = "outputs", batch_size: int = 32):
    device = get_device()
    model, ck = load_checkpoint(checkpoint, device)
    cfg = Config(data_root=data_root, batch_size=batch_size, num_workers=0)
    cfg.img_shape, cfg.resize = ck["img_shape"], ck["resize"]
    loaders, classes, _, _ = make_loaders(cfg)
    assert classes == ck["classes"], f"class order mismatch {classes} vs {ck['classes']}"

    cm = np.zeros((len(classes), len(classes)), dtype=int)  # rows = actual, cols = predicted
    for x, y in loaders["testloader"]:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            cm[t, p] += 1

    acc = float(np.trace(cm) / cm.sum())
    res = {"accuracy": acc, "confusion_matrix": cm.tolist(), "classes": classes,
           "per_class": per_class_metrics(cm, classes)}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(res, indent=2))
    plot_confusion(cm, classes, out / "confusion_matrix.png")
    return res


def plot_confusion(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/model.pt")
    p.add_argument("--data-root", default="data/synthetic")
    p.add_argument("--out-dir", default="outputs")
    a = p.parse_args(argv)
    r = evaluate(a.checkpoint, a.data_root, a.out_dir)
    print(f"\nOverall Accuracy: {r['accuracy']:.4f} ({r['accuracy'] * 100:.2f}%)")
    for c, m in r["per_class"].items():
        print(f"  {c:5s} sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
              f"prec={m['precision']:.3f} NPV={m['npv']:.3f} FN={m['false_negatives']}")
    if "pCT" in r["per_class"]:
        pct_fn = r["per_class"]["pCT"]["false_negatives"]
        print(f"\n>>> [SAFETY CRITICAL] pCT False Negatives (Missed COVID cases): {pct_fn}")
        print("    (In disease screening, False Negative / Type II error is the most costly mistake)")


if __name__ == "__main__":
    main()

