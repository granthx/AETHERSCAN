"""Datasets, transforms and dataloaders (mirrors the video's 'Load the data' cell)."""
from __future__ import annotations

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import Config
from .splits import stratified_split


def build_transforms(cfg: Config) -> dict:
    norm = transforms.Normalize(list(cfg.mean), list(cfg.std))
    return {
        "train": transforms.Compose([
            transforms.Resize(cfg.resize),
            transforms.RandomResizedCrop(cfg.img_shape),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ]),
        "valid": transforms.Compose([
            transforms.Resize(cfg.resize),
            transforms.CenterCrop(cfg.img_shape),
            transforms.ToTensor(),
            norm,
        ]),
    }


def make_loaders(cfg: Config):
    """Return (loaders, classes, class_to_idx, sizes).

    One ImageFolder per transform pipeline, indexed with the *same* stratified
    indices, so train gets augmentation while valid/test get deterministic crops.
    """
    tf = build_transforms(cfg)
    base = datasets.ImageFolder(cfg.data_root)
    train_ds = datasets.ImageFolder(cfg.data_root, transform=tf["train"])
    eval_ds = datasets.ImageFolder(cfg.data_root, transform=tf["valid"])

    labels = [y for _, y in base.samples]
    idx = stratified_split(labels, cfg.ratios, cfg.seed)

    def dl(ds, key, shuffle):
        return DataLoader(
            Subset(ds, idx[key]), batch_size=cfg.batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, pin_memory=False,
        )

    loaders = {
        "trainloader": dl(train_ds, "train", True),
        "validloader": dl(eval_ds, "valid", False),
        "testloader": dl(eval_ds, "test", False),
    }
    sizes = {k: len(v) for k, v in idx.items()}
    return loaders, base.classes, base.class_to_idx, sizes
