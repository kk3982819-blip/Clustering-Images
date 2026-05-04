from __future__ import annotations

import sys
from pathlib import Path
import argparse

from mask_generator import generate_sunray_mask


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the sunray mask generator on one image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("output_mask_path", nargs="?", type=Path, default=None)
    parser.add_argument("--mode", default=None, help="legacy_floor_patch | pretrained_baseline | segformer_fused | segformer_fused_sam")
    args = parser.parse_args()

    image_path = args.image_path
    output_mask_path = args.output_mask_path if args.output_mask_path is not None else Path("output_web") / f"{image_path.stem}_sunraymask.png"
    success = generate_sunray_mask(image_path, output_mask_path, mode=args.mode)
    print(f"success={success} output={output_mask_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
