"""Training loss, scheduler, and segmentation metrics for HEFGNet."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class OHEMCrossEntropy(nn.Module):
    """Online hard-example mining cross-entropy used in the experiments."""

    def __init__(
        self,
        ignore_index: int = 255,
        threshold: float = 0.7,
        min_kept: int = 20,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.min_kept = min_kept

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch, classes, height, width = logits.shape
        flat_target = target.reshape(-1)
        valid = flat_target.ne(self.ignore_index)
        safe_target = torch.where(valid, flat_target, torch.zeros_like(flat_target))

        probabilities = F.softmax(logits, dim=1)
        probabilities = probabilities.permute(1, 0, 2, 3).reshape(classes, -1)
        target_probability = probabilities.gather(
            0, safe_target.unsqueeze(0)
        ).squeeze(0)
        target_probability = target_probability.masked_fill(~valid, 1.0)

        num_valid = int(valid.sum().item())
        if num_valid >= self.min_kept and self.min_kept > 0:
            sorted_probability = target_probability[valid].sort().values
            dynamic_threshold = sorted_probability[
                min(self.min_kept, sorted_probability.numel()) - 1
            ]
            threshold = max(self.threshold, float(dynamic_threshold.item()))
            valid = valid & target_probability.le(threshold)

        mined_target = flat_target.masked_fill(~valid, self.ignore_index)
        mined_target = mined_target.view(batch, height, width)
        return F.cross_entropy(logits, mined_target, ignore_index=self.ignore_index)


@dataclass
class SegmentationMetrics:
    pixel_accuracy: float
    mean_accuracy: float
    mean_iou: float
    per_class_accuracy: list[float]
    per_class_iou: list[float]

    def as_dict(self) -> dict[str, float | list[float]]:
        return {
            "pixel_accuracy": self.pixel_accuracy,
            "mean_accuracy": self.mean_accuracy,
            "mean_iou": self.mean_iou,
            "per_class_accuracy": self.per_class_accuracy,
            "per_class_iou": self.per_class_iou,
        }


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, target: torch.Tensor, logits: torch.Tensor) -> None:
        prediction = logits.argmax(dim=1).detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        valid = (
            target.ne(self.ignore_index)
            & target.ge(0)
            & target.lt(self.num_classes)
        )
        indices = self.num_classes * target[valid].to(torch.int64) + prediction[valid]
        self.matrix += torch.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> SegmentationMetrics:
        matrix = self.matrix.float()
        diagonal = matrix.diag()
        total = matrix.sum().clamp_min(1.0)
        class_total = matrix.sum(dim=1)
        union = class_total + matrix.sum(dim=0) - diagonal
        class_accuracy = torch.nan_to_num(diagonal / class_total, nan=0.0)
        class_iou = torch.nan_to_num(diagonal / union, nan=0.0)
        return SegmentationMetrics(
            pixel_accuracy=float((diagonal.sum() / total).item() * 100.0),
            mean_accuracy=float(class_accuracy.mean().item() * 100.0),
            mean_iou=float(class_iou.mean().item() * 100.0),
            per_class_accuracy=(class_accuracy * 100.0).tolist(),
            per_class_iou=(class_iou * 100.0).tolist(),
        )


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_poly_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int = 1,
    warmup_factor: float = 1e-3,
    power: float = 0.9,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    def multiplier(step: int) -> float:
        if warmup_steps and step <= warmup_steps:
            progress = step / warmup_steps
            return warmup_factor * (1.0 - progress) + progress
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        return (1.0 - progress) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_function: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc=f"Train {epoch + 1}", leave=False)
    for step, (images, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = loss_function(logits, targets)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        running_loss += float(loss.item())
        progress.set_postfix(
            loss=running_loss / step,
            lr=optimizer.param_groups[0]["lr"],
        )
    return running_loss / max(len(loader), 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> SegmentationMetrics:
    model.eval()
    confusion = ConfusionMatrix(num_classes)
    for images, targets in tqdm(loader, desc="Evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        confusion.update(targets, logits)
    return confusion.compute()
