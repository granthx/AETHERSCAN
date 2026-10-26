"""Comprehensive unit and integration test suite for the ctcovid pipeline."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctcovid.splits import stratified_split


def test_split_is_disjoint_complete_and_stratified():
    labels = [0] * 100 + [1] * 60 + [2] * 40
    s = stratified_split(labels, (0.7, 0.15, 0.15), seed=42)
    all_idx = s["train"] + s["valid"] + s["test"]
    assert sorted(all_idx) == list(range(200))
    for part, frac in (("train", 0.7), ("valid", 0.15), ("test", 0.15)):
        ys = [labels[i] for i in s[part]]
        assert abs(ys.count(0) / 100 - frac) < 0.02


def test_split_with_string_labels():
    labels = ["nCT"] * 70 + ["pCT"] * 30
    s = stratified_split(labels, (0.7, 0.15, 0.15), seed=42)
    all_idx = s["train"] + s["valid"] + s["test"]
    assert sorted(all_idx) == list(range(100))
    assert len(s["train"]) == 70
    assert len(s["valid"]) + len(s["test"]) == 30


def test_split_is_deterministic():
    labels = [0, 1, 2] * 30
    assert stratified_split(labels, seed=1) == stratified_split(labels, seed=1)


def test_synthetic_generator_makes_distinct_classes():
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_synthetic_dataset as g
    rng = np.random.default_rng(0)
    imgs = {k: np.asarray(g.render(k, rng), dtype=float) for k in ("nCT", "pCT", "NiCT")}
    assert imgs["nCT"].shape == (256, 256, 3)
    # COVID slices should be brighter inside the lung field than normal ones
    assert imgs["pCT"][90:150, 60:110].mean() > imgs["nCT"][90:150, 60:110].mean()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_model_forward_shape_and_swapped_head():
    import torch

    from ctcovid.model import build_model, count_params
    m = build_model("mobilenet_v3_small", 3, pretrained=False)
    x = torch.zeros(2, 3, 224, 224)
    out = m(x)
    assert out.shape == (2, 3)
    assert m.classifier[-1].out_features == 3
    total, trainable = count_params(m)
    assert total > 0
    assert trainable > 0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_checkpoint_roundtrip_and_predict():
    from ctcovid.config import Config
    from ctcovid.model import build_model
    from ctcovid.predict import predict_image
    from ctcovid.train import save_checkpoint

    arch = "mobilenet_v3_small"
    cfg = Config(arch=arch)
    classes = ["NiCT", "nCT", "pCT"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    model = build_model(arch, len(classes), pretrained=False, freeze_backbone=False)

    ckpt_path = ROOT / "outputs" / "test_model_roundtrip.pt"
    save_checkpoint(ckpt_path, model, cfg, classes, class_to_idx, {"test_acc": 0.99})

    # Test predict_image with topk and json structure
    dummy_img = Image.new("RGB", (256, 256), color=(100, 100, 100))
    res = predict_image(dummy_img, checkpoint=str(ckpt_path), threshold=0.10, topk=3)
    assert "label" in res
    assert "confidence" in res
    assert "probs" in res
    assert len(res["top_k"]) == 3
    assert "disclaimer" in res
    assert not res["is_uncertain"]

    # Cleanup temp checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()



@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_gradcam_generation():
    from ctcovid.explain import explain_image
    ckpt = ROOT / "outputs" / "model.pt"
    if not ckpt.exists():
        pytest.skip("outputs/model.pt not generated yet")

    dummy_img = Image.new("RGB", (224, 224), color=(50, 120, 200))
    cam = explain_image(dummy_img, checkpoint=str(ckpt))
    assert cam["heatmap"].shape == (7, 7)
    assert cam["overlay"].size == (224, 224)
    assert cam["class_name"] in ["NiCT", "nCT", "pCT"]


def test_calibration_ece():
    from ctcovid.calibration import compute_ece
    confs = [0.95, 0.85, 0.75, 0.45]
    preds = [1, 1, 0, 1]
    trues = [1, 1, 1, 0]
    res = compute_ece(confs, preds, trues, n_bins=5)
    assert "ece" in res
    assert 0.0 <= res["ece"] <= 1.0
    assert len(res["bins"]) == 5


def test_robustness_perturbations():
    from ctcovid.robustness import (
        perturb_brightness,
        perturb_jpeg,
        perturb_noise,
        perturb_rotation,
    )
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    j = perturb_jpeg(img, 20)
    assert j.size == (100, 100)
    b = perturb_brightness(img, 1.5)
    assert b.size == (100, 100)
    r = perturb_rotation(img, 10.0)
    assert r.size == (100, 100)
    n = perturb_noise(img, 10.0)
    assert n.size == (100, 100)
