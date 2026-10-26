"""Generate a SYNTHETIC stand-in for the Kaggle CT dataset (same folder layout).

!! These are procedurally drawn cartoons, NOT medical images. They exist only so the
!! pipeline (data -> train -> evaluate -> predict -> app) runs end-to-end offline.
!! Swap in the real dataset via scripts/download_real_dataset.md for real results.

Classes (same names as Kaggle):
  nCT  - normal: two clean dark lungs with fine vessels
  pCT  - COVID-positive: lungs with bright patchy 'ground-glass' opacities
  NiCT - non-informative: slice outside the chest (neck/head-like, no big lungs)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

S = 256


def _grid():
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    return x, y


def _ellipse(cx, cy, rx, ry, x, y):
    return (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2) <= 1.0


def _base(rng, kind: str):
    x, y = _grid()
    img = np.full((S, S), 0.04, np.float32)
    j = lambda a: a + rng.uniform(-6, 6)
    if kind == "NiCT":  # head/neck-like: small body, bone ring, no lungs
        body = _ellipse(j(128), j(128), rng.uniform(55, 75), rng.uniform(60, 80), x, y)
        ring = body & ~_ellipse(128, 128, 45, 50, x, y)
        img[body] = 0.45
        img[ring] = 0.85
        for _ in range(3):  # small side structures (shoulders / jaw)
            img[_ellipse(j(rng.choice([55, 200])), j(150), 18, 25, x, y)] = 0.6
        lungs = np.zeros_like(img, bool)
    else:
        body = _ellipse(j(128), j(125), rng.uniform(105, 118), rng.uniform(80, 92), x, y)
        img[body] = 0.40
        img[_ellipse(128, 125, 106, 82, x, y) & ~_ellipse(128, 125, 100, 76, x, y)] = 0.7  # ribs/skin
        lw, lh = rng.uniform(34, 42), rng.uniform(52, 62)
        left = _ellipse(j(88), j(122), lw, lh, x, y)
        right = _ellipse(j(168), j(122), lw, lh, x, y)
        lungs = left | right
        img[lungs] = 0.08
        img[_ellipse(128, 108, 20, 24, x, y)] = 0.62  # heart/mediastinum
        for _ in range(60):  # vessels
            vx, vy = rng.integers(60, 196), rng.integers(80, 170)
            if lungs[vy, vx]:
                img[max(vy - 1, 0):vy + 1, max(vx - 1, 0):vx + 2] = 0.35
    img[(y > 205) & (y < 212) & (np.abs(x - 128) < 85)] = 0.55  # CT table
    return img, lungs


def _opacities(img, lungs, rng):
    """Add prominent multifocal ground-glass opacities and peripheral consolidations (pCT)."""
    noise = np.zeros_like(img)
    # Create characteristic peripheral subpleural opacities
    for _ in range(rng.integers(6, 12)):
        cx = rng.choice([rng.integers(68, 108), rng.integers(146, 188)])
        cy = rng.integers(95, 155)
        rad = rng.uniform(12, 26)
        x, y = _grid()
        blob = ((x - cx) ** 2 + (y - cy) ** 2) < rad ** 2
        noise[blob] += rng.uniform(0.38, 0.58)

    # Infiltrates in bilateral mid-to-lower zones
    tex = rng.normal(0, 0.04, img.shape).astype(np.float32)
    img = img + (noise + tex * (noise > 0)) * lungs
    return np.clip(img, 0, 1)


def render(kind: str, rng) -> Image.Image:
    img, lungs = _base(rng, kind)
    if kind == "pCT":
        img = _opacities(img, lungs, rng)
    img = img + rng.normal(0, 0.02, img.shape)  # subtle scanner noise
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.1)))
    return im.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=100, help="images per class")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    counts = {"nCT": a.n, "NiCT": a.n, "pCT": a.n}
    for k, n in counts.items():
        d = Path(a.out) / k
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            render(k, rng).save(d / f"{k}_{i:04d}.jpg", quality=92)
        print(f"{k}: {n} images -> {d}")


if __name__ == "__main__":
    main()

