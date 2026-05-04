from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sunray_pipeline import run_sunray_pipeline


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap pseudo-labels for sunray detection.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="pretrained_baseline")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def _iter_images(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        items = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES]
    else:
        items = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES]
    return sorted(items)


def _unique_name(path: Path, seen: dict[str, int]) -> str:
    stem = path.stem
    count = seen.get(stem, 0)
    seen[stem] = count + 1
    return f"{stem}_{count:03d}" if count else stem


def main() -> int:
    args = _parse_args()
    images = _iter_images(args.input_dir, recursive=args.recursive)
    if not images:
        raise SystemExit(f"No images found under {args.input_dir}")

    images_dir = args.output_dir / "images"
    masks_dir = args.output_dir / "masks"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    seen_stems: dict[str, int] = {}
    for image_path in images:
        unique_stem = _unique_name(image_path, seen_stems)
        copied_image_path = images_dir / f"{unique_stem}{image_path.suffix.lower()}"
        shutil.copy2(image_path, copied_image_path)

        output_mask_path = masks_dir / f"{unique_stem}.png"
        result = run_sunray_pipeline(copied_image_path, output_mask_path, mode=args.mode)
        manifest.append(
            {
                "image": copied_image_path.name,
                "mask": output_mask_path.name,
                "mode": result.mode_used,
                "success": result.success,
                "scene_type": result.scene_type,
                "segformer_used": result.segformer_used,
                "sam_refined": result.sam_refined,
                "mask_coverage_pct": result.metadata.get("mask_coverage_pct", 0.0),
            }
        )

    manifest_path = args.output_dir / "pseudo_label_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest)} pseudo-labels at {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
