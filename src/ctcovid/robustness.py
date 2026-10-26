"""Robustness evaluation against real-world CT distortions and acquisition shifts.

Tests model degradation under:
- JPEG compression artifacts (e.g., DICOM-to-JPEG conversion)
- Brightness / windowing variations (scanner calibration variations)
- Slices rotated by ±10° (patient positioning tilt)
- High-frequency Gaussian noise (sensor noise / low-dose CT)
"""
from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance

from .evaluate import load_checkpoint
from .predict import process_image
from .train import get_device


def perturb_jpeg(img: Image.Image, quality: int = 30) -> Image.Image:
    """Simulate lossy image compression artifacts."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


def perturb_brightness(img: Image.Image, factor: float = 1.3) -> Image.Image:
    """Simulate scanner windowing/exposure level shift."""
    enhancer = ImageEnhance.Brightness(img.convert("RGB"))
    return enhancer.enhance(factor)


def perturb_rotation(img: Image.Image, degrees: float = 10.0) -> Image.Image:
    """Simulate patient tilt or scanner bed misalignment."""
    return img.convert("RGB").rotate(degrees, resample=Image.Resampling.BILINEAR)


def perturb_noise(img: Image.Image, std: float = 15.0) -> Image.Image:
    """Simulate low-dose CT sensor noise."""
    arr = np.array(img.convert("RGB"), dtype=float)
    noise = np.random.normal(0, std, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)


@torch.no_grad()
def evaluate_robustness(
    sample_images: Sequence[tuple[Image.Image, int]],
    checkpoint: str = "outputs/model.pt",
) -> dict:
    """Evaluate model accuracy under multiple perturbation regimes."""
    device = get_device()
    model, ck = load_checkpoint(checkpoint, device)

    perturbation_suite = {
        "baseline (clean)": lambda im: im,
        "jpeg_q50 (moderate compression)": lambda im: perturb_jpeg(im, 50),
        "jpeg_q20 (heavy compression)": lambda im: perturb_jpeg(im, 20),
        "brightness_high (+30%)": lambda im: perturb_brightness(im, 1.3),
        "brightness_low (-30%)": lambda im: perturb_brightness(im, 0.7),
        "rotation_plus_10deg": lambda im: perturb_rotation(im, 10.0),
        "rotation_minus_10deg": lambda im: perturb_rotation(im, -10.0),
        "gaussian_noise (sensor noise)": lambda im: perturb_noise(im, 15.0),
    }

    results = {}
    for test_name, transform_fn in perturbation_suite.items():
        correct = 0
        total = len(sample_images)

        for img, true_label in sample_images:
            distorted = transform_fn(img)
            tensor = process_image(distorted, ck).to(device)
            pred = model(tensor).argmax(dim=1).item()
            if pred == true_label:
                correct += 1

        acc = correct / max(total, 1)
        results[test_name] = {
            "correct": correct,
            "total": total,
            "accuracy": acc,
        }

    return results
