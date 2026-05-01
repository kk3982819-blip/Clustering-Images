from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from mask_generator import _component_masks, _find_window_boxes, _segment_openings


def _normalized_blur(mask: np.ndarray, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    max_value = float(blurred.max())
    if max_value <= 1e-6:
        return np.zeros_like(blurred, dtype=np.float32)
    return blurred / max_value


def _filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


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


def _infer_openings_from_sky_mask(image_path: Path, shape: tuple[int, int], workdir: Path) -> list[np.ndarray]:
    from mask_generator import generate_sky_mask as _generate_sky_mask

    h, w = shape
    temp_mask_path = workdir / f"{image_path.stem}_sunray_skymask.png"
    if not _generate_sky_mask(image_path, temp_mask_path):
        return []

    sky_mask = cv2.imread(str(temp_mask_path), cv2.IMREAD_GRAYSCALE)
    if sky_mask is None or np.count_nonzero(sky_mask) == 0:
        return []

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
        bottom = min(h - 1, y2 + max(int(opening_h * 3.2), int(h * 0.18)))
        inferred.append(
            _rect_mask(
                (h, w),
                x1 - pad_x,
                top,
                x2 + pad_x,
                bottom,
            )
        )

    return inferred


def _opening_components(image_path: Path, img: np.ndarray, workdir: Path) -> list[np.ndarray]:
    boxes = _find_window_boxes(img)
    masks = _segment_openings(img, boxes)

    components: list[np.ndarray] = []
    h, w = img.shape[:2]
    min_area = max(200, int(h * w * 0.001))
    for mask in masks:
        components.extend(_component_masks(mask, min_area=min_area))

    inferred = _infer_openings_from_sky_mask(image_path, (h, w), workdir)
    components.extend(inferred)
    return components


def _build_floor_prior(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    lower_gate = np.clip((yy - h * 0.52) / max(h * 0.48, 1.0), 0.0, 1.0)
    center_bias = 1.0 - np.clip(np.abs(xx - w * 0.5) / max(w * 0.75, 1.0), 0.0, 1.0)
    center_bias = 0.72 + center_bias * 0.28
    return np.clip(lower_gate * center_bias, 0.0, 1.0)


def _build_sunlight_corridor(shape: tuple[int, int], openings: list[np.ndarray]) -> np.ndarray:
    h, w = shape
    if not openings:
        return np.zeros((h, w), dtype=np.float32)

    corridor = np.zeros((h, w), dtype=np.float32)
    for opening in openings:
        ys, xs = np.where(opening > 0)
        if len(xs) == 0:
            continue

        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())
        opening_w = max(1, x2 - x1 + 1)
        opening_h = max(1, y2 - y1 + 1)

        top = int(np.clip(y2 - opening_h * 0.06, 0, h - 1))
        spread = max(24, int(opening_w * 0.62))
        polygon = np.array(
            [
                [int(np.clip(x1 + opening_w * 0.14, 0, w - 1)), top],
                [int(np.clip(x2 - opening_w * 0.14, 0, w - 1)), top],
                [int(np.clip(x2 + spread, 0, w - 1)), h - 1],
                [int(np.clip(x1 - spread * 0.38, 0, w - 1)), h - 1],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(corridor, polygon, 1.0)

    corridor = cv2.GaussianBlur(corridor, (0, 0), 22.0)
    max_value = float(corridor.max())
    if max_value <= 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    return corridor / max_value


def _component_overlap(component: np.ndarray, prior: np.ndarray) -> float:
    pixels = component > 0
    if not np.any(pixels):
        return 0.0
    return float(prior[pixels].mean())


def _filter_sunray_components(mask: np.ndarray, corridor_prior: np.ndarray, floor_prior: np.ndarray) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask)
    h, w = mask.shape[:2]
    min_area = max(280, int(h * w * 0.002))

    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if width < 12 or height < 8:
            continue

        component = np.zeros_like(mask)
        component[labels == label] = 255
        corridor_score = _component_overlap(component, corridor_prior)
        floor_score = _component_overlap(component, floor_prior)

        # Sun patches should live on the floor, below openings, and have a
        # reasonably elongated footprint rather than a tiny blob.
        aspect_ratio = width / max(height, 1)
        if corridor_score < 0.22 or floor_score < 0.18:
            continue
        if y < int(h * 0.48):
            continue
        if aspect_ratio < 1.1:
            continue

        kept[labels == label] = 255

    return kept


def _build_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    color = np.zeros_like(overlay)
    color[:, :, 1] = 255
    color[:, :, 2] = 255
    alpha = (mask.astype(np.float32) / 255.0) * 0.42
    blended = overlay.astype(np.float32) * (1.0 - alpha[..., None]) + color.astype(np.float32) * alpha[..., None]

    contour_input = mask.copy()
    contours, _ = cv2.findContours(contour_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.clip(blended, 0, 255).astype(np.uint8)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 2)
    return result


def generate_legacy_sunray_mask(image_path: str | Path, output_mask_path: str | Path) -> bool:
    image_path = Path(image_path)
    output_mask_path = Path(output_mask_path)

    img = cv2.imread(str(image_path))
    if img is None:
        return False

    h, w = img.shape[:2]
    openings = _opening_components(image_path, img, output_mask_path.parent)
    empty_mask = np.zeros((h, w), dtype=np.uint8)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    if not openings:
        cv2.imwrite(str(output_mask_path), empty_mask)
        logging.info("[Sunray] No openings detected for %s", image_path.name)
        return False

    floor_prior = _build_floor_prior((h, w))
    corridor_prior = _build_sunlight_corridor((h, w), openings)
    candidate_zone = ((corridor_prior > 0.18) & (floor_prior > 0.16)).astype(np.uint8) * 255
    if np.count_nonzero(candidate_zone) == 0:
        cv2.imwrite(str(output_mask_path), empty_mask)
        logging.info("[Sunray] Empty candidate zone for %s", image_path.name)
        return False

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    l_channel = lab[:, :, 0]
    b_channel = lab[:, :, 2]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    sigma = max(10.0, min(h, w) * 0.028)
    l_coarse = cv2.GaussianBlur(l_channel, (0, 0), sigma)
    b_coarse = cv2.GaussianBlur(b_channel, (0, 0), sigma)
    s_coarse = cv2.GaussianBlur(saturation, (0, 0), sigma)
    v_coarse = cv2.GaussianBlur(value, (0, 0), sigma)

    outer_ring = cv2.dilate(
        candidate_zone,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
        iterations=1,
    )
    background_ring = cv2.subtract(outer_ring, candidate_zone)
    background_ring = cv2.bitwise_and(
        background_ring,
        ((floor_prior > 0.14) & (corridor_prior < 0.12)).astype(np.uint8) * 255,
    )
    if np.count_nonzero(background_ring) < max(220, int(h * w * 0.0015)):
        background_ring = ((floor_prior > 0.18) & (corridor_prior < 0.10)).astype(np.uint8) * 255

    if np.count_nonzero(background_ring) == 0:
        cv2.imwrite(str(output_mask_path), empty_mask)
        logging.info("[Sunray] No floor baseline available for %s", image_path.name)
        return False

    baseline_l = float(np.median(l_coarse[background_ring > 0]))
    baseline_b = float(np.median(b_coarse[background_ring > 0]))
    baseline_s = float(np.median(s_coarse[background_ring > 0]))
    baseline_v = float(np.median(v_coarse[background_ring > 0]))

    brightness_delta = l_coarse - baseline_l
    warmth_delta = b_coarse - baseline_b
    saturation_delta = s_coarse - baseline_s
    value_delta = v_coarse - baseline_v

    bright_candidate = brightness_delta > 11.0
    warm_candidate = warmth_delta > 1.0
    vivid_candidate = (value_delta > 6.0) | (saturation_delta > -8.0)

    candidate_mask = (bright_candidate & warm_candidate & vivid_candidate & (candidate_zone > 0)).astype(np.uint8) * 255

    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1,
    )
    candidate_mask = cv2.dilate(
        candidate_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    candidate_mask = _filter_small_components(
        candidate_mask,
        min_area=max(260, int(h * w * 0.0016)),
    )
    candidate_mask = _filter_sunray_components(candidate_mask, corridor_prior, floor_prior)

    if np.count_nonzero(candidate_mask) == 0:
        cv2.imwrite(str(output_mask_path), empty_mask)
        logging.info("[Sunray] No confident sunray patch for %s", image_path.name)
        return False

    candidate_mask = cv2.GaussianBlur(candidate_mask, (0, 0), 4.0)
    candidate_mask = (candidate_mask > 48).astype(np.uint8) * 255
    cv2.imwrite(str(output_mask_path), candidate_mask)

    debug_prefix = output_mask_path.with_suffix("")
    cv2.imwrite(str(debug_prefix.with_name(f"{debug_prefix.stem}_corridor.png")), np.clip(corridor_prior * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(debug_prefix.with_name(f"{debug_prefix.stem}_candidate_raw.png")), candidate_zone)
    cv2.imwrite(str(debug_prefix.with_name(f"{debug_prefix.stem}_overlay.jpg")), _build_overlay(img, candidate_mask))

    pct = np.count_nonzero(candidate_mask) / float(h * w) * 100.0
    logging.info("[Sunray] Sunray mask saved - %.2f%% of image", pct)
    return True


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python sunray_detector.py <input_path> <output_mask_path>")
    generate_legacy_sunray_mask(sys.argv[1], sys.argv[2])


generate_sunray_mask = generate_legacy_sunray_mask
