from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mask_generator import _component_masks, _find_window_boxes, _segment_openings, generate_sky_mask


@dataclass(slots=True)
class LightSourceHypothesis:
    x: float
    y: float
    score: float
    source_type: str
    bbox: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class LightSourceEstimation:
    sources: list[LightSourceHypothesis]
    opening_masks: list[np.ndarray]
    sky_mask: np.ndarray
    scene_type: str


def _rect_mask(shape: tuple[int, int], x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(
        mask,
        (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1))),
        (int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1))),
        255,
        -1,
    )
    return mask


def _load_or_generate_sky_mask(image_path: Path, shape: tuple[int, int], workdir: Path) -> np.ndarray:
    temp_mask_path = workdir / f"{image_path.stem}_sunray_skymask.png"
    if not temp_mask_path.exists():
        try:
            generate_sky_mask(image_path, temp_mask_path)
        except Exception as exc:
            logging.warning("[Sunray] Sky mask generation failed for %s: %s", image_path.name, exc)
            return np.zeros(shape, dtype=np.uint8)

    sky_mask = cv2.imread(str(temp_mask_path), cv2.IMREAD_GRAYSCALE)
    if sky_mask is None:
        return np.zeros(shape, dtype=np.uint8)

    h, w = shape
    if sky_mask.shape[:2] != (h, w):
        sky_mask = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (sky_mask > 0).astype(np.uint8) * 255


def _infer_openings_from_sky_mask(image_path: Path, shape: tuple[int, int], workdir: Path) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = shape
    sky_mask = _load_or_generate_sky_mask(image_path, shape, workdir)
    if np.count_nonzero(sky_mask) == 0:
        return [], sky_mask

    inferred: list[np.ndarray] = []
    min_area = max(120, int(h * w * 0.0004))
    for component in _component_masks(sky_mask, min_area=min_area):
        ys, xs = np.where(component > 0)
        if len(xs) == 0:
            continue

        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())
        opening_w = max(1, x2 - x1 + 1)
        opening_h = max(1, y2 - y1 + 1)

        pad_x = max(8, int(opening_w * 0.14))
        top = max(0, y1 - int(opening_h * 0.10))
        bottom = min(h - 1, y2 + max(int(opening_h * 0.24), int(h * 0.03)))
        inferred.append(_rect_mask((h, w), x1 - pad_x, top, x2 + pad_x, bottom))

    return inferred, sky_mask


def _infer_fast_openings_from_image(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    value = hsv[:, :, 2] / 255.0
    saturation = hsv[:, :, 1] / 255.0
    lightness = lab[:, :, 0] / 255.0
    coarse_lightness = cv2.GaussianBlur(lightness, (0, 0), max(5.0, min(h, w) * 0.012))

    upper_h = max(1, int(h * 0.78))
    upper_values = coarse_lightness[:upper_h, :]
    if upper_values.size == 0:
        return []

    bright_threshold = float(np.clip(np.percentile(upper_values, 84), 0.58, 0.88))
    value_threshold = float(np.clip(np.percentile(value[:upper_h, :], 82), 0.55, 0.92))
    candidate = (
        (coarse_lightness >= bright_threshold)
        & (value >= value_threshold)
        & (saturation <= 0.72)
    ).astype(np.uint8) * 255
    candidate[upper_h:, :] = 0
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )

    inferred: list[np.ndarray] = []
    min_area = max(180, int(h * w * 0.006))
    for component in _component_masks(candidate, min_area=min_area):
        ys, xs = np.where(component > 0)
        if xs.size == 0:
            continue

        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())
        opening_w = max(1, x2 - x1 + 1)
        opening_h = max(1, y2 - y1 + 1)
        area_ratio = xs.size / float(max(h * w, 1))
        aspect_ratio = opening_w / float(max(opening_h, 1))
        center_y = (y1 + y2) * 0.5
        touches_side = x1 <= int(w * 0.06) or x2 >= int(w * 0.94)
        near_top = y1 <= int(h * 0.16)
        mean_brightness = float(coarse_lightness[component > 0].mean())

        if area_ratio < 0.008 or area_ratio > 0.40:
            continue
        if center_y > h * 0.52:
            continue
        if not (touches_side or near_top or aspect_ratio >= 0.85):
            continue
        if mean_brightness < 0.60:
            continue

        pad_x = max(10, int(opening_w * 0.08))
        top = max(0, y1 - int(opening_h * 0.06))
        bottom = min(h - 1, y2 + max(int(opening_h * 0.18), int(h * 0.03)))
        inferred.append(_rect_mask((h, w), x1 - pad_x, top, x2 + pad_x, bottom))

    return inferred


def infer_opening_masks(
    image_path: Path,
    img: np.ndarray,
    workdir: Path,
    fast_mode: bool | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    if fast_mode is None:
        fast_mode = os.environ.get("SUNRAY_FAST_LIGHT_SOURCES", "").strip().lower() in {"1", "true", "yes"}
    heuristic_openings = _infer_fast_openings_from_image(img)
    if fast_mode:
        return heuristic_openings, np.zeros(img.shape[:2], dtype=np.uint8)

    disable_opening_models = os.environ.get("SUNRAY_DISABLE_OPENING_MODELS", "").strip().lower() in {"1", "true", "yes"}
    if disable_opening_models:
        boxes = np.empty((0, 4), dtype=np.float32)
        opening_masks = []
    else:
        boxes = _find_window_boxes(img)
        opening_masks = _segment_openings(img, boxes)

    components: list[np.ndarray] = []
    h, w = img.shape[:2]
    min_area = max(200, int(h * w * 0.001))
    components.extend(heuristic_openings)
    for mask in opening_masks:
        components.extend(_component_masks(mask, min_area=min_area))

    inferred, sky_mask = _infer_openings_from_sky_mask(image_path, (h, w), workdir)
    components.extend(inferred)
    return components, sky_mask


def _opening_sources(image: np.ndarray, opening_masks: list[np.ndarray]) -> list[LightSourceHypothesis]:
    if not opening_masks:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape[:2]
    image_area = float(max(h * w, 1))
    sources: list[LightSourceHypothesis] = []

    for opening in opening_masks:
        ys, xs = np.where(opening > 0)
        if xs.size == 0:
            continue

        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())
        area = float(xs.size)
        brightness = float(gray[opening > 0].mean()) / 255.0
        position_bonus = 1.0 - min(1.0, ((y1 + y2) * 0.5) / max(h, 1))
        area_score = min(1.0, area / (image_area * 0.10))
        score = float(np.clip(brightness * 0.45 + area_score * 0.35 + position_bonus * 0.20, 0.0, 1.0))
        sources.append(
            LightSourceHypothesis(
                x=float((x1 + x2) * 0.5),
                y=float(max(0, y1 + max(2, int((y2 - y1) * 0.10)))),
                score=score,
                source_type="window",
                bbox=(x1, y1, x2, y2),
            )
        )

    return sources


def _bright_top_sources(image: np.ndarray) -> list[LightSourceHypothesis]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    h, w = value.shape[:2]

    top_h = max(1, int(h * 0.62))
    top_values = value[:top_h, :]
    if top_values.size == 0:
        return []

    bright_threshold = max(205.0, float(np.percentile(top_values, 91)))
    candidate = ((value >= bright_threshold) & (sat <= 210)).astype(np.uint8) * 255
    candidate[top_h:, :] = 0
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )

    min_area = max(60, int(h * w * 0.00035))
    sources: list[LightSourceHypothesis] = []
    for component in _component_masks(candidate, min_area=min_area):
        ys, xs = np.where(component > 0)
        if xs.size == 0:
            continue
        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())
        component_value = float(value[component > 0].mean()) / 255.0
        top_bonus = 1.0 - min(1.0, y1 / max(h * 0.6, 1.0))
        spread_penalty = min(1.0, (x2 - x1 + 1) / max(w * 0.55, 1.0))
        score = float(np.clip(component_value * 0.65 + top_bonus * 0.25 - spread_penalty * 0.10, 0.0, 1.0))
        sources.append(
            LightSourceHypothesis(
                x=float((x1 + x2) * 0.5),
                y=float((y1 + y2) * 0.5),
                score=score,
                source_type="sun_visible",
                bbox=(x1, y1, x2, y2),
            )
        )

    return sources


def _sky_sources(sky_mask: np.ndarray) -> list[LightSourceHypothesis]:
    if np.count_nonzero(sky_mask) == 0:
        return []

    h, w = sky_mask.shape[:2]
    min_area = max(100, int(h * w * 0.001))
    sources: list[LightSourceHypothesis] = []
    for component in _component_masks(sky_mask, min_area=min_area):
        ys, xs = np.where(component > 0)
        if xs.size == 0:
            continue

        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())
        area_score = min(1.0, xs.size / float(max(h * w * 0.18, 1)))
        top_bonus = 1.0 - min(1.0, y1 / max(h * 0.75, 1.0))
        score = float(np.clip(area_score * 0.65 + top_bonus * 0.35, 0.0, 1.0))
        sources.append(
            LightSourceHypothesis(
                x=float((x1 + x2) * 0.5),
                y=float(max(0, y1 + int((y2 - y1 + 1) * 0.08))),
                score=score,
                source_type="sun_hidden",
                bbox=(x1, y1, x2, y2),
            )
        )

    return sources


def _dedupe_sources(sources: list[LightSourceHypothesis], image_shape: tuple[int, int]) -> list[LightSourceHypothesis]:
    if not sources:
        return []

    h, w = image_shape
    min_dist = max(14.0, min(h, w) * 0.08)
    deduped: list[LightSourceHypothesis] = []

    for source in sorted(sources, key=lambda item: item.score, reverse=True):
        keep = True
        for existing in deduped:
            if np.hypot(source.x - existing.x, source.y - existing.y) < min_dist:
                keep = False
                break
        if keep:
            deduped.append(source)

    return deduped


def _infer_scene_type(shape: tuple[int, int], sky_mask: np.ndarray, opening_masks: list[np.ndarray]) -> str:
    h, w = shape
    image_area = float(max(h * w, 1))
    sky_ratio = np.count_nonzero(sky_mask) / image_area
    opening_area = sum(np.count_nonzero(mask) for mask in opening_masks) / image_area

    if opening_masks and opening_area < 0.55:
        return "indoor_opening"
    if sky_ratio >= 0.12:
        return "outdoor_sky"
    return "ambiguous"


def estimate_light_sources(
    image_path: str | Path,
    image: np.ndarray | None = None,
    workdir: str | Path | None = None,
    max_sources: int = 4,
    fast_mode: bool | None = None,
) -> LightSourceEstimation:
    image_path = Path(image_path)
    if image is None:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

    workdir = Path(workdir) if workdir is not None else image_path.parent
    workdir.mkdir(parents=True, exist_ok=True)

    opening_masks, sky_mask = infer_opening_masks(image_path, image, workdir, fast_mode=fast_mode)
    sources = []
    sources.extend(_opening_sources(image, opening_masks))
    sources.extend(_sky_sources(sky_mask))
    sources.extend(_bright_top_sources(image))
    sources = _dedupe_sources(sources, image.shape[:2])[:max_sources]

    scene_type = _infer_scene_type(image.shape[:2], sky_mask, opening_masks)
    return LightSourceEstimation(
        sources=sources,
        opening_masks=opening_masks,
        sky_mask=sky_mask,
        scene_type=scene_type,
    )
