"""Evaluate an HEFGNet checkpoint using mIoU, mAcc, and pixel accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hefgnet import build_hefgnet
from hefgnet.data import build_dataset
from hefgnet.engine import evaluate, seed_worker, set_seed
from train import DATASET_DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_DEFAULTS),
        default="foodseg103",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--save-json", default=None)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    num_classes = DATASET_DEFAULTS[args.dataset]["num_classes"]
    dataset = build_dataset(args.dataset, args.data_root, "test", args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )

    model = build_hefgnet(
        num_classes=num_classes,
        checkpoint=args.checkpoint,
    ).to(device)
    metrics = evaluate(model, loader, device, num_classes)
    payload = metrics.as_dict()
    print(f"Pixel accuracy: {metrics.pixel_accuracy:.2f}%")
    print(f"mAcc: {metrics.mean_accuracy:.2f}%")
    print(f"mIoU: {metrics.mean_iou:.2f}%")
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(parse_args())
