"""YOLOatr training pipeline.

Implements:
- SGD optimizer with cosine LR schedule + warmup
- Mixed precision training (fp16)
- Checkpoint management (top-2 by val_mAP)
- Early stopping
- TensorBoard logging

Paper hyperparameters: SGD lr=0.01, momentum=0.937, wd=0.0005,
batch=32, epochs=100, from scratch.

Paper: arxiv 2507.11267
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from anima_yoloatr.dataset import YOLODataset, collate_fn
from anima_yoloatr.evaluate import evaluate_model
from anima_yoloatr.losses import ComputeLoss
from anima_yoloatr.model import build_model
from anima_yoloatr.utils import load_config, set_seed

PROJECT = "project_yoloatr"
ARTIFACTS = "/mnt/artifacts-datai"


class CheckpointManager:
    """Manages model checkpoints, keeping only top-K by metric."""

    def __init__(
        self,
        save_dir: str,
        keep_top_k: int = 2,
        metric: str = "val_map50",
        mode: str = "max",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(
        self,
        state: dict,
        metric_value: float,
        epoch: int,
    ) -> Path:
        """Save checkpoint and prune old ones."""
        path = self.save_dir / f"yoloatr_epoch{epoch:03d}_map{metric_value:.4f}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))

        # Sort: best first
        reverse = self.mode == "max"
        self.history.sort(key=lambda x: x[0], reverse=reverse)

        # Prune
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            if old_path.exists():
                old_path.unlink()

        # Copy best
        best_val, best_path = self.history[0]
        best_dest = self.save_dir / "best.pth"
        shutil.copy2(best_path, best_dest)

        return path


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            return True
        return False


class WarmupCosineScheduler:
    """Linear warmup + cosine annealing LR scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_bias_lr: float = 0.1,
        warmup_momentum: float = 0.8,
        min_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.base_momentum = [
            pg.get("momentum", 0.937) for pg in optimizer.param_groups
        ]
        self.current_epoch = 0

    def step(self, epoch: int) -> None:
        """Update LR for given epoch."""
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            xi = epoch / self.warmup_epochs
            for i, pg in enumerate(self.optimizer.param_groups):
                pg["lr"] = self.base_lrs[i] * xi
                if "momentum" in pg:
                    pg["momentum"] = (
                        self.warmup_momentum
                        + (self.base_momentum[i] - self.warmup_momentum) * xi
                    )
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
            for i, pg in enumerate(self.optimizer.param_groups):
                pg["lr"] = self.base_lrs[i] * scale

    def state_dict(self) -> dict:
        return {"current_epoch": self.current_epoch}

    def load_state_dict(self, state: dict) -> None:
        self.current_epoch = state["current_epoch"]
        self.step(self.current_epoch)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    compute_loss: ComputeLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_items = {"box": 0.0, "obj": 0.0, "cls": 0.0}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        targets = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast("cuda", enabled=use_amp):
            predictions = model(images)
            loss, loss_dict = compute_loss(predictions, targets)

        if torch.isnan(loss):
            print(f"[FATAL] NaN loss at epoch {epoch}, batch {batch_idx}")
            print("[FIX] Reduce lr, check data, check gradient clipping")
            break

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            if k in loss_items:
                loss_items[k] += v
        num_batches += 1

    if num_batches > 0:
        total_loss /= num_batches
        loss_items = {k: v / num_batches for k, v in loss_items.items()}

    return {
        "train_loss": total_loss,
        **{f"train_{k}": v for k, v in loss_items.items()},
    }


def train(config: dict, resume_path: str | None = None, max_steps: int | None = None) -> None:
    """Main training function."""
    # Setup
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # Config values
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    loss_cfg = config.get("loss", {})
    ckpt_cfg = config.get("checkpoint", {})
    log_cfg = config.get("logging", {})
    aug_cfg = config.get("augmentation", {})
    es_cfg = config.get("early_stopping", {})

    epochs = train_cfg.get("epochs", 100)
    batch_size = train_cfg.get("batch_size", 32)
    lr = train_cfg.get("learning_rate", 0.01)
    momentum = train_cfg.get("momentum", 0.937)
    weight_decay = train_cfg.get("weight_decay", 0.0005)
    num_workers = train_cfg.get("num_workers", 4)
    use_amp = train_cfg.get("precision", "fp16") in ("fp16", "bf16")
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    # Model
    num_classes = model_cfg.get("num_classes", 4)
    model = build_model(num_classes=num_classes)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] YOLOatr | {param_count:,} parameters")

    # Dataset
    train_dataset = YOLODataset(
        data_root=data_cfg.get("train_path", ""),
        img_size=model_cfg.get("input_size", 640),
        augment=True,
        hyp=aug_cfg,
    )
    val_dataset = YOLODataset(
        data_root=data_cfg.get("val_path", ""),
        img_size=model_cfg.get("input_size", 640),
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=train_cfg.get("pin_memory", True),
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    print(f"[DATA] train={len(train_dataset)} val={len(val_dataset)}")
    print(f"[TRAIN] {epochs} epochs, lr={lr}, batch_size={batch_size}")

    # Optimizer (SGD as per paper)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=train_cfg.get("warmup_epochs", 3),
        total_epochs=epochs,
        warmup_bias_lr=train_cfg.get("warmup_bias_lr", 0.1),
        warmup_momentum=train_cfg.get("warmup_momentum", 0.8),
    )

    # Loss
    compute_loss = ComputeLoss(
        model,
        box_gain=loss_cfg.get("box_gain", 0.05),
        obj_gain=loss_cfg.get("obj_gain", 1.0),
        cls_gain=loss_cfg.get("cls_gain", 0.5),
        focal_gamma=loss_cfg.get("focal_gamma", 0.3),
        anchor_threshold=loss_cfg.get("anchor_threshold", 4.0),
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
    )

    # Scaler for mixed precision
    scaler = GradScaler("cuda", enabled=use_amp)

    # Checkpoint manager
    ckpt_manager = CheckpointManager(
        save_dir=ckpt_cfg.get("output_dir", f"{ARTIFACTS}/checkpoints/{PROJECT}"),
        keep_top_k=ckpt_cfg.get("keep_top_k", 2),
        metric=ckpt_cfg.get("metric", "val_map50"),
        mode=ckpt_cfg.get("mode", "max"),
    )

    # Early stopping
    early_stopper = None
    if es_cfg.get("enabled", True):
        early_stopper = EarlyStopping(
            patience=es_cfg.get("patience", 10),
            min_delta=es_cfg.get("min_delta", 0.0001),
            mode="max",
        )

    # TensorBoard
    tb_dir = log_cfg.get("tensorboard_dir", f"{ARTIFACTS}/tensorboard/{PROJECT}")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    # Log dir
    log_dir = log_cfg.get("log_dir", f"{ARTIFACTS}/logs/{PROJECT}")
    os.makedirs(log_dir, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    best_map = 0.0
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_map = ckpt.get("best_map", 0.0)
        print(f"[RESUME] From epoch {start_epoch}, best_map={best_map:.4f}")

    # Print config
    print(f"[CONFIG] {json.dumps(config.get('training', {}), indent=2)}")
    print(f"[CKPT] save_dir={ckpt_manager.save_dir}")

    # Training loop
    global_step = 0
    for epoch in range(start_epoch, epochs):
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, compute_loss, optimizer, scaler,
            device, epoch, max_grad_norm, use_amp,
        )
        t1 = time.time()
        global_step += len(train_loader)

        # Max steps check (for smoke tests)
        if max_steps is not None and global_step >= max_steps:
            print(f"[MAX STEPS] Reached {global_step} steps, stopping.")
            # Save before exit
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
                "config": config,
            }
            ckpt_manager.save(state, 0.0, epoch)
            break

        # Validate
        eval_cfg = config.get("evaluation", {})
        val_metrics = evaluate_model(
            model, val_loader, device,
            conf_threshold=eval_cfg.get("conf_threshold", 0.001),
            iou_threshold=eval_cfg.get("iou_threshold", 0.5),
            nms_threshold=eval_cfg.get("nms_threshold", 0.6),
        )

        val_map = val_metrics.get("map50", 0.0)

        # Log
        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"val_mAP={val_map:.4f} "
            f"lr={current_lr:.6f} "
            f"time={t1 - t0:.1f}s"
        )

        writer.add_scalar("train/loss", train_metrics["train_loss"], epoch)
        writer.add_scalar("train/box_loss", train_metrics.get("train_box", 0), epoch)
        writer.add_scalar("train/obj_loss", train_metrics.get("train_obj", 0), epoch)
        writer.add_scalar("train/cls_loss", train_metrics.get("train_cls", 0), epoch)
        writer.add_scalar("val/mAP50", val_map, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        # Checkpoint
        save_every = ckpt_cfg.get("save_every_n_epochs", 10)
        if (epoch + 1) % save_every == 0 or val_map > best_map:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_map": max(val_map, best_map),
                "config": config,
            }
            ckpt_manager.save(state, val_map, epoch)
            if val_map > best_map:
                best_map = val_map
                print(f"  -> New best mAP: {best_map:.4f}")

        # Early stopping
        if early_stopper and early_stopper.step(val_map):
            print(f"[EARLY STOP] No improvement for {early_stopper.patience} epochs")
            break

    writer.close()
    print(f"\n[DONE] Best mAP@0.5: {best_map:.4f}")
    print(f"[CKPT] Best model: {ckpt_manager.save_dir / 'best.pth'}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train YOLOatr")
    parser.add_argument(
        "--config", type=str, default="configs/paper.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Max training steps (for smoke tests)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_path=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
