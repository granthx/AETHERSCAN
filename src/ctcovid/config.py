"""Central configuration. Values mirror what is shown in the source video."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass
class Config:
    data_root: str = "data/synthetic"      # folder containing nCT/ pCT/ NiCT/ (ImageFolder layout)
    out_dir: str = "outputs"
    arch: str = "mobilenet_v3_large"       # 960-ch final feature map, as in the video's diagram
    img_shape: int = 224                   # IMG_SHAPE in the video
    resize: int = 256                      # transforms.Resize(256)
    mean: tuple = (0.485, 0.456, 0.406)    # ImageNet normalisation
    std: tuple = (0.229, 0.224, 0.225)
    ratios: tuple = (0.70, 0.15, 0.15)     # train / valid / test (2800/600/601 for pCT)
    seed: int = 42
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    freeze_backbone: bool = True           # stage 1: train head only (feature extraction)
    unfreeze_at_epoch: int = 3             # stage 2: fine-tune whole net (0 = never)
    pretrained: bool = True                # ImageNet weights (falls back to random if offline)
    num_workers: int = 2
    print_every: int = 10
    use_amp: bool = False                  # mixed precision (cuda/mps)
    class_weights: bool = False            # class-weighted loss (for imbalanced classes)
    uncertainty_threshold: float = 0.60    # confidence threshold for 'uncertain' prediction

    def to_dict(self) -> dict:
        return asdict(self)


def add_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    d = Config()
    p.add_argument("--data-root", default=d.data_root)
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--arch", default=d.arch, choices=["mobilenet_v3_large", "mobilenet_v3_small"])
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--unfreeze-at-epoch", type=int, default=d.unfreeze_at_epoch)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--class-weights", action="store_true", help="Use inverse frequency class weights in CE loss")
    p.add_argument("--uncertainty-threshold", type=float, default=d.uncertainty_threshold)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    return p


def from_args(a: argparse.Namespace) -> Config:
    c = Config()
    c.data_root, c.out_dir, c.arch = a.data_root, a.out_dir, a.arch
    c.epochs, c.batch_size, c.lr, c.seed = a.epochs, a.batch_size, a.lr, a.seed
    c.unfreeze_at_epoch, c.pretrained = a.unfreeze_at_epoch, not a.no_pretrained
    c.use_amp, c.num_workers = a.amp, a.num_workers
    c.class_weights = getattr(a, "class_weights", False)
    c.uncertainty_threshold = getattr(a, "uncertainty_threshold", 0.60)
    return c

