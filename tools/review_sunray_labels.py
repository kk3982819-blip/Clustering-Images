from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create contact sheets for reviewing sunray pseudo-labels.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root containing images/ and masks/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=360)
    return parser.parse_args()


def _find_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    pairs: list[tuple[Path, Path]] = []
    if not images_dir.exists() or not masks_dir.exists():
        return pairs

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        mask_path = masks_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def _render_tile(image: np.ndarray, mask: np.ndarray, label: str, tile_width: int) -> np.ndarray:
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(image)
    color[:, :, 1] = 255
    color[:, :, 2] = 255
    alpha = (mask.astype(np.float32) / 255.0) * 0.42
    overlay = image.astype(np.float32) * (1.0 - alpha[..., None]) + color.astype(np.float32) * alpha[..., None]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    scale = tile_width / max(image.shape[1], 1)
    tile_height = max(60, int(round(image.shape[0] * scale)))
    resized = cv2.resize(overlay, (tile_width, tile_height), interpolation=cv2.INTER_AREA)

    label_band = np.full((42, tile_width, 3), 24, dtype=np.uint8)
    cv2.putText(label_band, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    return np.vstack([resized, label_band])


def main() -> int:
    args = _parse_args()
    pairs = _find_pairs(args.dataset_root)
    if not pairs:
        raise SystemExit(f"No image/mask pairs found under {args.dataset_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    tiles: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    sheets: list[np.ndarray] = []

    for image_path, mask_path in pairs:
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        coverage = float(np.count_nonzero(mask)) / float(max(mask.size, 1)) * 100.0
        label = f"{image_path.stem}  {coverage:.2f}%"
        tile = _render_tile(image, mask, label, args.tile_width)
        tiles.append(tile)
        manifest.append({"image": image_path.name, "mask": mask_path.name, "coverage_pct": round(coverage, 4)})

    if not tiles:
        raise SystemExit("No valid review tiles could be rendered.")

    tile_height = max(tile.shape[0] for tile in tiles)
    normalized_tiles = []
    for tile in tiles:
        if tile.shape[0] == tile_height:
            normalized_tiles.append(tile)
            continue
        pad = np.full((tile_height - tile.shape[0], tile.shape[1], 3), 18, dtype=np.uint8)
        normalized_tiles.append(np.vstack([tile, pad]))

    for idx, tile in enumerate(normalized_tiles, start=1):
        rows.append(tile)
        if len(rows) == args.columns or idx == len(normalized_tiles):
            while len(rows) < args.columns:
                rows.append(np.full((tile_height, args.tile_width, 3), 18, dtype=np.uint8))
            sheets.append(np.hstack(rows))
            rows = []

    for page_index, sheet in enumerate(sheets, start=1):
        cv2.imwrite(str(args.output_dir / f"review_page_{page_index:03d}.jpg"), sheet)

    (args.output_dir / "review_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(sheets)} review page(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
