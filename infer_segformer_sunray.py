from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from models.segformer_sunray import load_sunray_model, predict_probability_map


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SegFormer sunray inference on one image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("output_mask_path", type=Path)
    parser.add_argument("--weights-path", type=Path, required=True, help="Checkpoint file or model directory")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--force-tiny", action="store_true")
    return parser.parse_args()


def _build_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.zeros_like(image)
    color[:, :, 1] = 255
    color[:, :, 2] = 255
    alpha = (mask.astype(np.float32) / 255.0) * 0.42
    blend = image.astype(np.float32) * (1.0 - alpha[..., None]) + color.astype(np.float32) * alpha[..., None]
    return np.clip(blend, 0, 255).astype(np.uint8)


def main() -> int:
    args = _parse_args()
    image = cv2.imread(str(args.image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {args.image_path}")

    bundle = load_sunray_model(
        weights_path=args.weights_path,
        input_size=args.image_size,
        prefer_transformer=not args.force_tiny,
        local_files_only=True,
    )
    probability = predict_probability_map(bundle, image)
    threshold = float(np.clip(max(0.42, np.percentile(probability, 92) * 0.72), 0.42, 0.68))
    binary_mask = (probability >= threshold).astype(np.uint8) * 255

    args.output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = args.output_mask_path.with_suffix("")
    cv2.imwrite(str(args.output_mask_path), binary_mask)
    cv2.imwrite(str(prefix.with_name(f"{prefix.stem}_prob.png")), np.clip(probability * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(prefix.with_name(f"{prefix.stem}_overlay.jpg")), _build_overlay(image, binary_mask))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
