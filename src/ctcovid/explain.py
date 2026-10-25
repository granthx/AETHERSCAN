"""Grad-CAM (Gradient-weighted Class Activation Mapping) explainability module.

Exposes visual explanations for CT scan predictions, verifying whether attention
is focused on lung lesions/consolidation rather than scan borders, text markers, or beds.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .evaluate import load_checkpoint
from .model import get_cam_target_layer
from .predict import process_image
from .train import get_device


class GradCAM:
    """Computes Grad-CAM heatmaps for a specified convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or get_cam_target_layer(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """Generate normalized 2D Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor: [1, 3, H, W] preprocessed image tensor.
            class_idx: Optional target class index. If None, uses top predicted class.

        Returns:
            (heatmap_2d, target_class_idx) where heatmap_2d is a float numpy array in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        # Gradients & activations: [1, C, H, W]
        assert self.gradients is not None and self.activations is not None
        grads = self.gradients[0]       # [C, H, W]
        acts = self.activations[0]     # [C, H, W]

        # Global average pooling of gradients -> weights [C]
        weights = grads.mean(dim=(1, 2))  # [C]

        # Weighted combination of activation channels
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for w, act in zip(weights, acts):
            cam += w * act

        # Apply ReLU to retain only features with positive influence on the class
        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            # Fallback to feature activation magnitude if gradients are flat/zeroed
            fallback = acts.abs().mean(dim=0).cpu().numpy()
            f_min, f_max = fallback.min(), fallback.max()
            if f_max - f_min > 1e-8:
                cam = (fallback - f_min) / (f_max - f_min)
            else:
                cam = np.full_like(cam, 0.5)

        return cam, class_idx


def colorize_heatmap(
    heatmap: np.ndarray,
    size: tuple[int, int],
    colormap: str = "jet",
) -> Image.Image:
    """Render a 2D float heatmap in [0, 1] into a vivid full-color RGB PIL Image."""
    import matplotlib

    heatmap_img = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8)).resize(
        size, resample=Image.Resampling.BILINEAR
    )
    norm = np.array(heatmap_img, dtype=float) / 255.0
    cmap = matplotlib.colormaps.get_cmap(colormap)
    colored_rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored_rgb)


def overlay_cam_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """Blend a 2D heatmap [0, 1] onto a PIL Image."""
    colored_cam = np.array(colorize_heatmap(heatmap, image.size, colormap=colormap), dtype=np.uint8)
    base_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    blended = (base_rgb * (1.0 - alpha) + colored_cam * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def explain_image(
    img: Image.Image,
    checkpoint: str = "outputs/model.pt",
    class_idx: int | None = None,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> dict:
    """Generate Grad-CAM explanation for an image file/PIL image.

    Returns dict with keys:
        - 'class_name': predicted or requested class name
        - 'class_idx': class index
        - 'confidence': predicted confidence
        - 'heatmap': [H, W] float array in [0, 1]
        - 'heatmap_color': PIL Image of full-color colormapped heatmap
        - 'overlay': PIL Image with heatmap blended
    """
    device = get_device()
    model, ck = load_checkpoint(checkpoint, device)
    input_tensor = process_image(img, ck).to(device)

    # Enable gradients on the model parameters for Grad-CAM
    for p in model.parameters():
        p.requires_grad = True

    cam_engine = GradCAM(model)
    heatmap, selected_idx = cam_engine.generate(input_tensor, class_idx=class_idx)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0].cpu().tolist()

    class_name = ck["classes"][selected_idx]
    confidence = float(probs[selected_idx])

    heatmap_color = colorize_heatmap(heatmap, img.size, colormap=colormap)
    overlay = overlay_cam_on_image(img, heatmap, alpha=alpha, colormap=colormap)

    return {
        "class_name": class_name,
        "class_idx": selected_idx,
        "confidence": confidence,
        "heatmap": heatmap,
        "heatmap_color": heatmap_color,
        "overlay": overlay,
        "all_probs": dict(zip(ck["classes"], probs)),
    }

