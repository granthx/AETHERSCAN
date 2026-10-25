"""Single-image inference (video: 'Inference for classification' notebook section).
Run: python -m ctcovid.predict path/to/slice.jpg --checkpoint outputs/model.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .evaluate import load_checkpoint
from .train import get_device

DISCLAIMER = "Research/education prototype - NOT a medical device. Not for diagnosis."


def process_image(img: Image.Image, ck: dict) -> torch.Tensor:
    """Return a [1,3,H,W] batched tensor (matches the video's process_image)."""
    tf = transforms.Compose([
        transforms.Resize(ck["resize"]), transforms.CenterCrop(ck["img_shape"]),
        transforms.ToTensor(), transforms.Normalize(ck["mean"], ck["std"]),
    ])
    return tf(img.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict_image(img: Image.Image, checkpoint: str = "outputs/model.pt",
                  threshold: float = 0.50, topk: int = 3) -> dict:
    device = get_device()
    model, ck = load_checkpoint(checkpoint, device)
    tensor = process_image(img, ck).to(device)
    raw_probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    scores = dict(zip(ck["classes"], raw_probs))
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_conf = sorted_items[0]

    top_predictions = [
        {"class": c, "confidence": float(prob)}
        for c, prob in sorted_items[:topk]
    ]

    is_uncertain = best_conf < threshold

    return {
        "label": best_label if not is_uncertain else "Uncertain",
        "predicted_class": best_label,
        "confidence": float(best_conf),
        "is_uncertain": is_uncertain,
        "threshold": threshold,
        "probs": scores,
        "top_k": top_predictions,
        "disclaimer": DISCLAIMER,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Predict COVID-19 status from a single CT scan slice")
    p.add_argument("image", help="Path to input CT image (JPG/PNG)")
    p.add_argument("--checkpoint", default="outputs/model.pt", help="Path to model checkpoint")
    p.add_argument("--topk", type=int, default=3, help="Number of top predictions to display")
    p.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold below which prediction is marked uncertain")
    p.add_argument("--json", action="store_true", help="Output result as JSON")
    a = p.parse_args(argv)

    img_path = Path(a.image)
    if not img_path.exists():
        p.error(f"Image file not found: {a.image}")

    r = predict_image(Image.open(img_path), a.checkpoint, threshold=a.threshold, topk=a.topk)

    if a.json:
        print(json.dumps(r, indent=2))
    else:
        status_tag = f" [UNCERTAIN (< {a.threshold:.0%})]" if r["is_uncertain"] else ""
        print(f"Prediction: {r['label']}{status_tag} (confidence: {r['confidence']:.1%})")
        print("\nTop predictions:")
        for item in r["top_k"]:
            print(f"  {item['class']:5s} {item['confidence']:6.2%}")
        print("\nAll class probabilities:")
        for k, v in r["probs"].items():
            print(f"  {k:5s} {v:.4f}")
        print(f"\nNotice: {r['disclaimer']}")


if __name__ == "__main__":
    main()

