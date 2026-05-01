from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.segformer_sunray import (
    SunraySegmentationDataset,
    discover_sunray_samples,
    load_sunray_model,
    save_sunray_model,
    split_samples,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sunray segmentation model.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root containing images/ and masks/")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--backbone", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--weights-path", type=Path, default=None, help="Existing model checkpoint or pretrained directory")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-tiny", action="store_true", help="Train the tiny fallback model instead of SegFormer")
    return parser.parse_args()


def _forward(bundle, pixel_values: torch.Tensor) -> torch.Tensor:
    if bundle.is_transformer:
        return bundle.model(pixel_values=pixel_values).logits
    return bundle.model(pixel_values)


def _compute_loss_and_iou(bundle, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, float]:
    pixel_values = batch["pixel_values"].to(bundle.device)
    labels = batch["labels"].to(bundle.device)
    logits = _forward(bundle, pixel_values)
    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    logits = logits[:, :1]
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    probs = torch.sigmoid(logits)
    preds = probs > 0.5
    targets = labels > 0.5
    intersection = (preds & targets).sum().float()
    union = (preds | targets).sum().float()
    iou = float((intersection / union).item()) if union.item() > 0 else 0.0
    return loss, iou


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_sunray_samples(args.dataset_root)
    if not samples:
        raise SystemExit(f"No training samples found under {args.dataset_root}")

    train_samples, val_samples = split_samples(samples, val_fraction=args.val_split)
    if not val_samples and len(train_samples) > 1:
        val_samples = train_samples[-1:]
        train_samples = train_samples[:-1]

    train_dataset = SunraySegmentationDataset(train_samples, image_size=args.image_size)
    val_dataset = SunraySegmentationDataset(val_samples, image_size=args.image_size) if val_samples else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers) if val_dataset else None

    bundle = load_sunray_model(
        weights_path=args.weights_path,
        backbone=args.backbone,
        input_size=args.image_size,
        prefer_transformer=not args.force_tiny,
        local_files_only=True,
    )
    bundle.model.train()
    optimizer = AdamW(bundle.model.parameters(), lr=args.lr)

    best_val_iou = -1.0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        bundle.model.train()
        train_loss_total = 0.0
        train_iou_total = 0.0
        train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, iou = _compute_loss_and_iou(bundle, batch)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item())
            train_iou_total += iou
            train_batches += 1

        train_loss = train_loss_total / max(train_batches, 1)
        train_iou = train_iou_total / max(train_batches, 1)
        val_loss = 0.0
        val_iou = 0.0
        val_batches = 0

        if val_loader is not None:
            bundle.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    loss, iou = _compute_loss_and_iou(bundle, batch)
                    val_loss += float(loss.item())
                    val_iou += iou
                    val_batches += 1
            val_loss /= max(val_batches, 1)
            val_iou /= max(val_batches, 1)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_iou": round(train_iou, 6),
                "val_loss": round(val_loss, 6),
                "val_iou": round(val_iou, 6),
            }
        )
        logging.info(
            "[SunrayTrain] epoch=%d train_loss=%.4f train_iou=%.4f val_loss=%.4f val_iou=%.4f",
            epoch,
            train_loss,
            train_iou,
            val_loss,
            val_iou,
        )

        should_save = val_iou >= best_val_iou
        if val_loader is None:
            should_save = epoch == args.epochs

        if should_save:
            best_val_iou = val_iou
            save_sunray_model(
                bundle,
                args.output_dir / "best_model",
                extra_metadata={
                    "epochs": args.epochs,
                    "architecture": bundle.architecture,
                    "backbone": args.backbone,
                    "best_val_iou": best_val_iou,
                    "train_samples": len(train_samples),
                    "val_samples": len(val_samples),
                },
            )

    summary = {
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "architecture": bundle.architecture,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "history": history,
        "best_val_iou": best_val_iou,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
