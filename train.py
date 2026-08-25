"""Train HEFGNet on FoodSeg103 or UECFoodPixComplete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hefgnet import build_hefgnet
from hefgnet.data import build_dataset
from hefgnet.engine import (
    OHEMCrossEntropy,
    create_poly_scheduler,
    evaluate,
    seed_worker,
    set_seed,
    train_one_epoch,
)


DATASET_DEFAULTS = {
    "foodseg103": {"num_classes": 104, "batch_size": 7, "learning_rate": 1.2e-5},
    "uecfoodpixcomplete": {
        "num_classes": 103,
        "batch_size": 6,
        "learning_rate": 8e-6,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_DEFAULTS),
        default="foodseg103",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--backbone-weights", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--ohem-min-kept", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    best_miou: float,
    args: argparse.Namespace,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_miou": best_miou,
        "args": vars(args),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def main(args: argparse.Namespace) -> None:
    defaults = DATASET_DEFAULTS[args.dataset]
    num_classes = defaults["num_classes"]
    batch_size = args.batch_size or defaults["batch_size"]
    learning_rate = args.learning_rate or defaults["learning_rate"]
    output_dir = Path(args.output_dir or f"outputs/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, deterministic=args.deterministic)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"

    train_dataset = build_dataset(
        args.dataset, args.data_root, "train", args.image_size
    )
    test_dataset = build_dataset(
        args.dataset, args.data_root, "test", args.image_size
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_options,
    )

    model = build_hefgnet(
        num_classes=num_classes,
        backbone_weights=None if args.resume else args.backbone_weights,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    )
    scheduler = create_poly_scheduler(
        optimizer,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        power=args.poly_power,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    loss_function = OHEMCrossEntropy(
        ignore_index=255, threshold=0.7, min_kept=args.ohem_min_kept
    )

    start_epoch = 0
    best_miou = -1.0
    if args.resume:
        model.load_checkpoint(args.resume, strict=True)
        checkpoint = torch.load(args.resume, map_location="cpu")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint.get("best_miou", best_miou))
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    metrics_path = output_dir / "metrics.jsonl"
    print(
        f"dataset={args.dataset} classes={num_classes} batch={batch_size} "
        f"lr={learning_rate:g} device={device}"
    )
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_function,
            device,
            scaler,
            epoch,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        should_evaluate = (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        if should_evaluate:
            metrics = evaluate(model, test_loader, device, num_classes)
            record.update(
                {
                    "pixel_accuracy": metrics.pixel_accuracy,
                    "mean_accuracy": metrics.mean_accuracy,
                    "mean_iou": metrics.mean_iou,
                }
            )
            print(
                f"epoch={epoch + 1} loss={train_loss:.4f} "
                f"mIoU={metrics.mean_iou:.2f} mAcc={metrics.mean_accuracy:.2f}"
            )
            if metrics.mean_iou > best_miou:
                best_miou = metrics.mean_iou
                save_checkpoint(
                    output_dir / "best.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_miou,
                    args,
                )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save_checkpoint(
            output_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_miou,
            args,
        )


if __name__ == "__main__":
    main(parse_args())
