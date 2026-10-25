"""Training loop + checkpointing.  Run: python -m ctcovid.train --data-root data/synthetic"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import Config, add_args, from_args
from .data import make_loaders
from .model import build_model, count_params, set_backbone_trainable


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def validation(model, loader, criterion, device):
    """Returns (loss, accuracy) -- same signature spirit as the video's validation()."""
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / max(n, 1), correct / max(n, 1)


def training(model, criterion, optimizer, device, trainloader, validloader,
             epochs=5, print_every=10, use_amp=False, cfg: Config | None = None):
    model.to(device)
    amp_on = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_on)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    best_acc, best_state = -1.0, None
    print(f"Start training on {device} (AMP={amp_on})")

    for e in range(1, epochs + 1):
        if cfg and cfg.freeze_backbone and cfg.unfreeze_at_epoch and e == cfg.unfreeze_at_epoch:
            set_backbone_trainable(model, True)
            known = {id(p) for g in optimizer.param_groups for p in g["params"]}
            new = [p for p in model.features.parameters() if id(p) not in known]
            for g in optimizer.param_groups:
                g["lr"] = cfg.lr * 0.1
            optimizer.add_param_group({"params": new, "lr": cfg.lr * 0.1})
            print(f"Epoch {e}: backbone unfrozen for fine-tuning")
        t0 = time.perf_counter()
        model.train()
        run_loss, correct, n = 0.0, 0, 0
        for step, (x, y) in enumerate(trainloader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_on):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
            if print_every and step % print_every == 0:
                print(f"  epoch {e} step {step}/{len(trainloader)} loss {run_loss / n:.4f}")
        vl, va = validation(model, validloader, criterion, device)
        history["train_loss"].append(run_loss / n); history["train_acc"].append(correct / n)
        history["val_loss"].append(vl); history["val_acc"].append(va)
        history["epoch_time"].append(time.perf_counter() - t0)
        print(f"Epoch {e}/{epochs} | train acc {correct / n:.4f} loss {run_loss / n:.4f} "
              f"| valid acc {va:.4f} loss {vl:.4f} | {history['epoch_time'][-1]:.1f}s")
        if va > best_acc:
            best_acc = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def compute_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    """Compute balanced class weights: total / (num_classes * class_count)."""
    counts = np.bincount(targets, minlength=num_classes)
    total = len(targets)
    weights = total / (num_classes * np.maximum(counts, 1).astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(path: str | Path, model: nn.Module, cfg: Config, classes: list[str],
                    class_to_idx: dict[str, int], metrics: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({
        "arch": cfg.arch, "num_classes": len(classes), "classes": classes,
        "class_to_idx": class_to_idx, "img_shape": cfg.img_shape, "resize": cfg.resize,
        "mean": list(cfg.mean), "std": list(cfg.std),
        "state_dict": cpu_state, "metrics": metrics or {},
    }, path)


def main(argv=None):
    p = add_args(argparse.ArgumentParser(description="Train CT COVID classifier"))
    cfg = from_args(p.parse_args(argv))
    seed_everything(cfg.seed)
    device = get_device()

    loaders, classes, class_to_idx, sizes = make_loaders(cfg)
    print("Classes:", classes, "| split sizes:", sizes)

    model = build_model(cfg.arch, len(classes), cfg.pretrained, cfg.freeze_backbone)
    total, trainable = count_params(model)
    print(f"Params: {total:,} total, {trainable:,} trainable")

    if cfg.class_weights:
        # Extract train targets for weighting
        train_targets = [y for _, y in loaders["trainloader"].dataset]
        weights = compute_class_weights(train_targets, len(classes)).to(device)
        print(f"Using class weights: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    hist = training(model, criterion, optimizer, device, loaders["trainloader"],
                    loaders["validloader"], cfg.epochs, cfg.print_every, cfg.use_amp, cfg)

    test_loss, test_acc = validation(model, loaders["testloader"], criterion, device)
    print(f"Test : acc = {test_acc:.4f} ({test_acc * 100:.2f}%), loss = {test_loss:.4f}")

    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    metrics = {"test_acc": test_acc, "test_loss": test_loss, "history": hist, "split_sizes": sizes}
    save_checkpoint(out / "model.pt", model, cfg, classes, class_to_idx, metrics)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved checkpoint -> {out / 'model.pt'}")



if __name__ == "__main__":
    main()
