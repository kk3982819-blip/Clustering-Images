from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from light_source_estimator import infer_opening_masks
from mask_generator import generate_sky_mask


REFERENCE_WEATHER_DIRS = {
    "sunrise": "sunrise",
    "sunset": "sunset",
    "clear": "clear blue sky",
    "cloudy": "cloudy",
}


SKY_ASSET_MAP = {
    "sunny": "clear_day",
    "clear": "clear_day",
    "sunrise": "golden_hour",
    "partly_cloudy": "partly_cloudy",
    "high_wisps": "partly_cloudy",
    "cloudy": "overcast",
    "foggy": "overcast",
    "rainy": "overcast",
    "drizzling": "overcast",
    "windy": "partly_cloudy",
    "dramatic": "sunset",
    "sunset": "sunset",          # FIX: use real photographic sky, not fake gradient
    "golden_hour": "golden_hour",
    "dusk": "golden_hour",
    "night": "golden_hour",
    "snowy": "overcast",
}

LIGHTING_BY_WEATHER = {
    "sunny": {
        "warmth": 0.25,
        "beam": 0.20,
        "exposure": 1.06,
        "tint": (1.00, 1.00, 1.00),
        "gamma": 0.96,
        "saturation": 1.05,
    },
    "clear": {
        "warmth": 0.18,
        "beam": 0.34,
        "exposure": 1.01,
        "tint": (0.96, 0.98, 1.00),
        "gamma": 0.99,
        "contrast": 1.01,
        "saturation": 1.04,
    },
    "partly_cloudy": {
        "warmth": 0.82,
        "beam": 0.62,
        "exposure": 1.00,
        "tint": (0.98, 0.98, 0.96),
    },
    "high_wisps": {
        "warmth": 0.76,
        "beam": 0.52,
        "exposure": 1.02,
        "tint": (1.00, 0.98, 0.95),
    },
    "cloudy": {
        "warmth": 0.42,
        "beam": 0.18,
        "exposure": 0.98,
        "tint": (0.91, 0.93, 0.95),
        "saturation": 0.92,
    },
    "foggy": {
        "warmth": 0.28,
        "beam": 0.08,
        "exposure": 0.98,
        "tint": (0.93, 0.95, 0.97),
        "contrast": 0.90,
        "saturation": 0.72,
    },
    "windy": {
        "warmth": 0.70,
        "beam": 0.45,
        "exposure": 1.00,
        "tint": (0.94, 0.96, 0.98),
    },
    "rainy": {
        "warmth": 0.22,
        "beam": 0.06,
        "exposure": 0.84,
        "tint": (0.82, 0.86, 0.92),
        "contrast": 0.94,
        "saturation": 0.70,
    },
    "drizzling": {
        "warmth": 0.30,
        "beam": 0.10,
        "exposure": 0.90,
        "tint": (0.86, 0.90, 0.94),
        "contrast": 0.94,
        "saturation": 0.78,
    },
    "snowy": {
        "warmth": 0.40,
        "beam": 0.20,
        "exposure": 1.06,
        "tint": (0.96, 0.98, 1.00),
        "contrast": 0.92,
        "saturation": 0.76,
    },
    "sunset": {
        # Sunset should feel warm and directional, without orange floodlighting.
        "warmth": 0.44,
        "beam": 0.36,
        "exposure": 0.95,
        "tint": (1.00, 0.90, 0.74),
        "gamma": 0.96,
        "saturation": 0.98,
    },
    "sunrise": {
        "warmth": 0.40,
        "beam": 0.30,
        "exposure": 0.96,
        "tint": (1.00, 0.92, 0.78),
        "gamma": 0.98,
        "saturation": 0.96,
    },
    "golden_hour": {
        "warmth": 0.52,
        "beam": 0.40,
        "exposure": 0.97,
        "tint": (1.00, 0.92, 0.76),
        "gamma": 0.97,
        "saturation": 1.02,
    },
    "dusk": {
        "warmth": 0.80,
        "beam": 0.35,
        "exposure": 0.78,
        "tint": (0.70, 0.74, 0.98),
        "gamma": 0.96,
        "saturation": 0.88,
    },
    "night": {
        "warmth": 0.18,
        "beam": 0.04,
        "exposure": 0.50,
        "tint": (0.34, 0.42, 0.78),
        "contrast": 1.08,
        "gamma": 1.08,
        "saturation": 0.68,
    },
    "dramatic": {
        "warmth": 0.55,
        "beam": 0.22,
        "exposure": 0.82,
        "tint": (0.72, 0.76, 0.88),
        "contrast": 1.08,
        "saturation": 0.82,
    },
    "rainbow": {
        "warmth": 0.88,
        "beam": 0.58,
        "exposure": 1.02,
        "tint": (0.95, 0.98, 1.00),
        "saturation": 1.06,
    },
}

GROUND_LIGHT_WEATHERS = {"sunny", "clear", "sunrise", "sunset", "golden_hour", "dusk", "night", "rainbow"}
PROJECTED_GROUND_LIGHT_WEATHERS = {"sunrise", "sunset"}

SKY_REGENERATION_PROMPTS = {
    "sunrise": "clean photorealistic sunrise sky outside the window, soft warm early morning light, natural cloud detail",
    "sunset": "clean photorealistic sunset sky outside the window, warm golden light, natural cloud detail",
    "clear": "clean photorealistic clear blue daytime sky outside the window, crisp natural daylight, no clouds, high detail",
    "cloudy": "clean photorealistic cloudy overcast sky outside the window, soft diffused daylight, natural cloud detail",
    "sunny": "clean photorealistic sunny blue daytime sky outside the window, crisp natural daylight, minimal clouds",
    "partly_cloudy": "clean photorealistic partly cloudy daytime sky outside the window, natural daylight, high detail",
    "dramatic": "clean photorealistic dramatic cloudy sky outside the window, natural contrast, high detail",
    "night": "clean photorealistic night sky outside the window, subtle distant city glow, realistic exposure",
}

SKY_REGENERATION_NEGATIVE_PROMPT = (
    "cartoon, painting, illustration, CGI, fake, blurry, low resolution, noisy, distorted architecture, "
    "warped window frames, changed furniture, changed walls, changed floor, duplicate objects, extra windows, "
    "curtains, people, cars, buildings, trees inside the room, text, watermark"
)


def _lighting_profile(weather: str) -> dict[str, float | tuple[float, float, float]]:
    return LIGHTING_BY_WEATHER.get(weather, LIGHTING_BY_WEATHER["partly_cloudy"])


def _sky_regeneration_prompt(weather: str) -> str:
    weather_prompt = SKY_REGENERATION_PROMPTS.get(weather, SKY_REGENERATION_PROMPTS["sunny"])
    return (
        f"{weather_prompt}, high-end real estate photography, natural exposure, realistic lens perspective, "
        "preserve the room, preserve straight architectural lines, preserve window frames, seamless edge blending"
    )


def _fit_cover(image: np.ndarray, width: int, height: int, focus_x: float = 0.72, focus_y: float = 0.78) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = max(width / max(src_w, 1), height / max(src_h, 1))
    new_w = max(width, int(round(src_w * scale)))
    new_h = max(height, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    max_x = max(0, new_w - width)
    max_y = max(0, new_h - height)
    crop_x = int(np.clip(focus_x * max_x, 0, max_x))
    crop_y = int(np.clip(focus_y * max_y, 0, max_y))
    return resized[crop_y:crop_y + height, crop_x:crop_x + width].copy()


def _find_reference_image(weather: str, reference_root: Path) -> Path | None:
    folder_name = REFERENCE_WEATHER_DIRS.get(weather)
    if not folder_name:
        return None

    reference_dir = reference_root / folder_name
    if not reference_dir.exists() or not reference_dir.is_dir():
        return None

    candidates = sorted(
        path for path in reference_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    return candidates[0] if candidates else None


def _reference_cache_path(reference_image_path: Path, reference_root: Path) -> Path:
    cache_dir = reference_root / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{reference_image_path.parent.name}_{reference_image_path.stem}_skymask.png"
    return cache_dir / cache_name


def _fallback_reference_sky_mask(reference_image: np.ndarray, weather: str) -> np.ndarray:
    opening_mask = _build_reference_opening_mask(reference_image)
    if np.count_nonzero(opening_mask) == 0:
        return np.zeros(reference_image.shape[:2], dtype=np.uint8)

    h, w = reference_image.shape[:2]
    hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    lightness = lab[:, :, 0] / 255.0
    hue = hsv[:, :, 0]

    pale_sky = (lightness > 0.54) & (saturation < 0.34) & (value > 0.34)
    blue_sky = (hue >= 76.0) & (hue <= 132.0) & (saturation > 0.08) & (value > 0.32)
    cloudy_sky = (lightness > 0.48) & (saturation < 0.22) & (value > 0.30)

    if weather == "cloudy":
        sky_like = cloudy_sky | pale_sky | blue_sky
        top_ratio = 0.62
    elif weather == "clear":
        sky_like = blue_sky | pale_sky
        top_ratio = 0.58
    else:
        sky_like = pale_sky | blue_sky
        top_ratio = 0.58

    upper_band = np.zeros((h, w), dtype=np.uint8)
    upper_band[: max(1, int(h * 0.82)), :] = 255
    candidate = (sky_like & (opening_mask > 0) & (upper_band > 0)).astype(np.uint8) * 255
    candidate = _connected_to_top_mask(candidate)

    if np.count_nonzero(candidate) == 0:
        components = _extract_mask_components(opening_mask, min_area_ratio=0.0035)
        fallback = np.zeros((h, w), dtype=np.uint8)
        for component in components:
            x = int(component["x"])
            y = int(component["y"])
            cw = int(component["w"])
            ch = int(component["h"])
            cutoff = min(h, y + max(1, int(ch * top_ratio)))
            component_mask = np.asarray(component["mask"], dtype=np.uint8)
            region = np.zeros((h, w), dtype=np.uint8)
            region[y:cutoff, x:x + cw] = 255
            fallback = cv2.bitwise_or(fallback, cv2.bitwise_and(component_mask, region))
        candidate = fallback

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    if weather == "cloudy" and np.count_nonzero(candidate) > 0:
        row_coverage = candidate.mean(axis=1) / 255.0
        active_rows = np.where(row_coverage > 0.08)[0]
        if active_rows.size > 0:
            max_bottom = min(int(active_rows.max()), max(1, int(h * 0.58)))
            candidate[max_bottom + 1 :, :] = 0
            candidate = cv2.GaussianBlur(candidate.astype(np.float32), (0, 0), 1.2)
            candidate = ((candidate > 18.0).astype(np.uint8) * 255)
    return candidate


def _load_weather_reference(
    weather: str,
    width: int,
    height: int,
    reference_root: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    reference_image_path = _find_reference_image(weather, reference_root)
    if reference_image_path is None:
        return None, None

    reference_image = cv2.imread(str(reference_image_path))
    if reference_image is None:
        return None, None

    reference_mask_path = _reference_cache_path(reference_image_path, reference_root)
    if not reference_mask_path.exists():
        try:
            generate_sky_mask(reference_image_path, reference_mask_path)
        except Exception as exc:
            logging.warning("[FullSceneGenerator] Could not generate reference sky mask for %s: %s", reference_image_path.name, exc)
            reference_mask = _fallback_reference_sky_mask(reference_image, weather)
            if np.count_nonzero(reference_mask) == 0:
                return reference_image, None
            cv2.imwrite(str(reference_mask_path), reference_mask)

    reference_mask = cv2.imread(str(reference_mask_path), cv2.IMREAD_GRAYSCALE)
    min_reference_sky_pixels = max(64, int(reference_image.shape[0] * reference_image.shape[1] * 0.001))
    if reference_mask is None or int(np.count_nonzero(reference_mask)) < min_reference_sky_pixels:
        reference_mask = _fallback_reference_sky_mask(reference_image, weather)
        if np.count_nonzero(reference_mask) > 0:
            cv2.imwrite(str(reference_mask_path), reference_mask)
        else:
            reference_mask = None

    if reference_image.shape[:2] != (height, width):
        reference_image = cv2.resize(reference_image, (width, height), interpolation=cv2.INTER_CUBIC)
    if reference_mask is not None and reference_mask.shape[:2] != (height, width):
        reference_mask = cv2.resize(reference_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    if reference_mask is None:
        return reference_image, None
    return reference_image, (reference_mask > 0).astype(np.uint8) * 255


def _load_sky_asset(weather: str, width: int, height: int, sky_assets_dir: Path) -> np.ndarray:
    asset_name = SKY_ASSET_MAP.get(weather, "golden_hour")
    asset_path = sky_assets_dir / f"{asset_name}.jpg"
    if not asset_path.exists():
        asset_path = sky_assets_dir / "golden_hour.jpg"

    asset = cv2.imread(str(asset_path))
    if asset is None:
        raise FileNotFoundError(f"Could not load sky asset: {asset_path}")

    return _fit_cover(asset, width, height)


def _resolve_processing_max_dim(processing_max_dim: int | None) -> int:
    if processing_max_dim is not None:
        return max(1200, int(processing_max_dim))
    raw_value = os.environ.get("PIXELDWELL_SCENE_MAX_DIM", "2200").strip()
    try:
        return max(1200, int(raw_value))
    except ValueError:
        return 2200


def _resize_for_processing(image: np.ndarray, processing_max_dim: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest_side = max(h, w)
    if longest_side <= processing_max_dim:
        return image.copy(), 1.0

    scale = processing_max_dim / float(longest_side)
    resized = cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _build_opening_stats(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    binary = (mask > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    openings: list[tuple[int, int, int, int]] = []
    image_h, image_w = mask.shape[:2]
    min_area = max(200, int(image_h * image_w * 0.001))

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        openings.append((x, y, w, h))

    openings.sort(key=lambda item: item[2] * item[3], reverse=True)
    return openings


def _build_reference_opening_mask(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    try:
        masks, _ = infer_opening_masks(Path("opening_probe.jpg"), image, Path("output_web"), fast_mode=True)
    except Exception:
        masks = []
    if not masks:
        return np.zeros((h, w), dtype=np.uint8)

    opening_mask = np.zeros((h, w), dtype=np.uint8)
    image_area = float(max(h * w, 1))
    min_area = max(200, int(h * w * 0.0012))
    for raw_mask in masks:
        mask_u8 = (raw_mask > 0).astype(np.uint8) * 255
        ys, xs = np.where(mask_u8 > 0)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        area_ratio = xs.size / image_area
        width = max(1, x2 - x1 + 1)
        height = max(1, y2 - y1 + 1)
        aspect_ratio = width / float(height)
        center_y_ratio = ((y1 + y2) * 0.5) / float(max(h, 1))
        bottom_ratio = y2 / float(max(h, 1))
        if xs.size < min_area:
            continue
        if not (0.02 <= area_ratio <= 0.40):
            continue
        if center_y_ratio < 0.10 or center_y_ratio > 0.46:
            continue
        if bottom_ratio < 0.22:
            continue
        if aspect_ratio < 0.90:
            continue
        opening_mask = cv2.bitwise_or(opening_mask, mask_u8)

    if np.count_nonzero(opening_mask) == 0:
        return opening_mask

    opening_mask = cv2.morphologyEx(
        opening_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    return opening_mask


def _build_mask_from_opening_polygons(
    image_shape: tuple[int, int],
    opening_polygons: Sequence[Sequence[Sequence[float | int]]] | None,
    opening_source_size: tuple[int, int] | None = None,
) -> np.ndarray:
    h, w = image_shape
    if not opening_polygons:
        return np.zeros((h, w), dtype=np.uint8)

    source_w, source_h = opening_source_size if opening_source_size is not None else (w, h)
    source_w = max(int(source_w), 1)
    source_h = max(int(source_h), 1)
    scale_x = w / float(source_w)
    scale_y = h / float(source_h)

    mask = np.zeros((h, w), dtype=np.uint8)
    for polygon in opening_polygons:
        if not polygon or len(polygon) < 3:
            continue
        scaled_points: list[list[int]] = []
        for point in polygon:
            if len(point) < 2:
                continue
            px = int(round(float(point[0]) * scale_x))
            py = int(round(float(point[1]) * scale_y))
            scaled_points.append([
                int(np.clip(px, 0, max(w - 1, 0))),
                int(np.clip(py, 0, max(h - 1, 0))),
            ])
        if len(scaled_points) >= 3:
            cv2.fillPoly(mask, [np.asarray(scaled_points, dtype=np.int32)], 255)

    if np.count_nonzero(mask) == 0:
        return mask

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return mask


def _extract_mask_components(mask: np.ndarray, min_area_ratio: float = 0.0025) -> list[dict[str, np.ndarray | int | float]]:
    binary = (mask > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return []

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    image_h, image_w = mask.shape[:2]
    image_area = float(max(image_h * image_w, 1))
    min_area = max(120, int(image_area * min_area_ratio))
    components: list[dict[str, np.ndarray | int | float]] = []

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_mask = np.zeros_like(mask, dtype=np.uint8)
        component_mask[labels == label] = 255
        components.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": float(area),
                "mask": component_mask,
            }
        )

    components.sort(key=lambda item: float(item["area"]), reverse=True)
    return components


def _estimate_opening_visibility(target_roi_u8: np.ndarray, target_mask_u8: np.ndarray) -> np.ndarray:
    target_mask_u8 = (target_mask_u8 > 0).astype(np.uint8) * 255
    if np.count_nonzero(target_mask_u8) == 0:
        return np.zeros(target_mask_u8.shape[:2], dtype=np.float32)

    hsv = cv2.cvtColor(target_roi_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(target_roi_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(target_roi_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    lightness = lab[:, :, 0] / 255.0

    blind_gate = np.clip((0.22 - saturation) / 0.22, 0.0, 1.0) * np.clip((lightness - 0.56) / 0.26, 0.0, 1.0)
    frame_gate = np.clip((0.14 - saturation) / 0.14, 0.0, 1.0) * np.clip((value - 0.72) / 0.18, 0.0, 1.0)

    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    horizontal_texture = np.abs(sobel_y)
    max_texture = float(horizontal_texture.max())
    if max_texture > 1e-6:
        horizontal_texture /= max_texture
    blind_texture = horizontal_texture * blind_gate

    dist = cv2.distanceTransform(target_mask_u8, cv2.DIST_L2, 5)
    border_gate = np.clip(dist / 7.0, 0.0, 1.0)

    visibility = 1.0 - np.clip(blind_gate * 0.78 + blind_texture * 0.62 + frame_gate * 0.58, 0.0, 0.97)
    visibility = cv2.GaussianBlur(visibility.astype(np.float32), (0, 0), 1.2)
    visibility *= border_gate
    visibility[target_mask_u8 == 0] = 0.0
    return np.clip(visibility, 0.0, 1.0).astype(np.float32)


def _masked_lab_mean_std(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    binary = mask > 0
    if int(np.count_nonzero(binary)) < 80:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[binary]
    if pixels.size == 0:
        return None
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    std = np.maximum(std, 1.0)
    return mean, std


def _connected_to_top_mask(candidate_mask: np.ndarray) -> np.ndarray:
    candidate_mask = (candidate_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(candidate_mask) == 0:
        return candidate_mask

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    h = candidate_mask.shape[0]
    keep = np.zeros_like(candidate_mask)
    top_limit = max(1, int(h * 0.18))
    min_area = max(24, int(candidate_mask.size * 0.002))
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        top = int(stats[label, cv2.CC_STAT_TOP])
        if area < min_area:
            continue
        if top > top_limit:
            continue
        keep[labels == label] = 255
    return keep


def _clear_sky_pixel_candidate(
    image: np.ndarray,
    allowed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    allowed = (allowed_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(allowed) == 0:
        empty = np.zeros(allowed.shape[:2], dtype=np.uint8)
        return empty, empty

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blue_channel, green_channel, red_channel = cv2.split(image.astype(np.int16))

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    lightness = lab[:, :, 0] / 255.0

    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    texture = cv2.GaussianBlur(laplacian, (5, 5), 0)
    smooth = texture < (38.0 + value * 74.0)

    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edge_mag = cv2.GaussianBlur(sobel_x + sobel_y, (5, 5), 0)
    low_edge = edge_mag < (72.0 + value * 58.0)

    blue_color = (
        (hue >= 78.0)
        & (hue <= 134.0)
        & (saturation > 0.075)
        & (value > 0.30)
        & (blue_channel > green_channel + 4)
        & (blue_channel > red_channel + 8)
    )
    pale_cloud = (lightness > 0.60) & (saturation < 0.30) & (value > 0.48)

    green_content = cv2.inRange(hsv.astype(np.uint8), np.array([25, 14, 18]), np.array([112, 255, 245])) > 0
    brown_content = cv2.inRange(hsv.astype(np.uint8), np.array([8, 28, 18]), np.array([38, 255, 230])) > 0
    dark_structure = (value < 0.34) & (saturation > 0.06)
    reject = green_content | brown_content | dark_structure

    allowed_bool = allowed > 0
    blue = (blue_color & smooth & low_edge & ~reject & allowed_bool).astype(np.uint8) * 255

    support_size = max(5, int(round(min(image.shape[:2]) * 0.022))) | 1
    blue_support = cv2.dilate(
        blue,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (support_size, support_size)),
        iterations=1,
    )
    neutral = (
        pale_cloud
        & smooth
        & low_edge
        & ~reject
        & allowed_bool
        & (blue_support > 0)
    ).astype(np.uint8) * 255

    candidate = cv2.bitwise_or(blue, neutral)
    return candidate, blue


def _trim_clear_sky_to_opening(candidate: np.ndarray, blue: np.ndarray, allowed_mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    candidate = (candidate > 0).astype(np.uint8) * 255
    allowed = (allowed_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(candidate) == 0 or np.count_nonzero(allowed) == 0:
        return np.zeros(candidate.shape[:2], dtype=np.uint8)

    candidate = cv2.bitwise_and(candidate, allowed)
    candidate = _connected_to_top_mask(candidate)
    if np.count_nonzero(candidate) == 0:
        return candidate

    h, w = candidate.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    green_content = cv2.inRange(hsv, np.array([25, 16, 18]), np.array([112, 255, 245]))
    brown_content = cv2.inRange(hsv, np.array([8, 30, 18]), np.array([38, 255, 230]))
    dark_content = (((hsv[:, :, 2] < 88) & (hsv[:, :, 1] > 16)).astype(np.uint8) * 255)
    exterior_content = cv2.bitwise_or(cv2.bitwise_or(green_content, brown_content), dark_content)

    edge_mag = cv2.GaussianBlur(
        np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)),
        (5, 5),
        0,
    )
    high_edge = (edge_mag > 84.0).astype(np.uint8) * 255

    valid = allowed > 0
    blue_rows: list[float] = []
    candidate_rows: list[float] = []
    content_rows: list[float] = []
    edge_rows: list[float] = []
    min_valid = max(8, int(w * 0.04))
    for y in range(h):
        row_valid = valid[y]
        valid_count = int(np.count_nonzero(row_valid))
        if valid_count < min_valid:
            blue_rows.append(0.0)
            candidate_rows.append(0.0)
            content_rows.append(0.0)
            edge_rows.append(0.0)
            continue
        blue_rows.append(np.count_nonzero(blue[y][row_valid] > 0) / float(valid_count))
        candidate_rows.append(np.count_nonzero(candidate[y][row_valid] > 0) / float(valid_count))
        content_rows.append(np.count_nonzero(exterior_content[y][row_valid] > 0) / float(valid_count))
        edge_rows.append(np.count_nonzero(high_edge[y][row_valid] > 0) / float(valid_count))

    blue_arr = np.array(blue_rows, dtype=np.float32)
    candidate_arr = np.array(candidate_rows, dtype=np.float32)
    content_arr = np.array(content_rows, dtype=np.float32)
    edge_arr = np.array(edge_rows, dtype=np.float32)
    cutoff = h

    scan_start = max(1, int(h * 0.25))
    band_h = max(6, int(round(h * 0.025)))
    for y in range(scan_start, max(scan_start, h - band_h + 1)):
        band_content = float(np.mean(content_arr[y:y + band_h]))
        band_edge = float(np.mean(edge_arr[y:y + band_h]))
        band_blue = float(np.mean(blue_arr[y:y + band_h]))
        if band_content > 0.13 and band_blue < 0.30:
            cutoff = min(cutoff, max(0, y - int(round(h * 0.018))))
            break
        if y > h * 0.42 and band_edge > 0.18 and band_blue < 0.10:
            cutoff = min(cutoff, max(0, y - int(round(h * 0.012))))
            break

    strong_blue_rows = np.where(blue_arr > 0.035)[0]
    if strong_blue_rows.size > max(3, int(h * 0.025)):
        bottom_blue = int(strong_blue_rows.max())
        lower_start = min(h, bottom_blue + max(8, int(h * 0.045)))
        if lower_start < h:
            lower_candidate = float(np.mean(candidate_arr[lower_start:]))
            lower_blue = float(np.mean(blue_arr[lower_start:]))
            if lower_candidate > 0.18 and lower_blue < 0.018:
                cutoff = min(cutoff, lower_start)

    if cutoff < h:
        candidate[cutoff:, :] = 0

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    candidate = cv2.bitwise_and(candidate, allowed)
    return candidate


def _build_opening_sky_candidate(target_roi_u8: np.ndarray, visibility: np.ndarray, target_mask_u8: np.ndarray) -> np.ndarray:
    allowed = ((visibility > 0.12) & (target_mask_u8 > 0)).astype(np.uint8) * 255
    candidate, blue = _clear_sky_pixel_candidate(target_roi_u8, allowed)
    candidate = _trim_clear_sky_to_opening(candidate, blue, allowed, target_roi_u8)
    return candidate


def _estimate_secondary_opening_transmission(
    target_roi_u8: np.ndarray,
    target_mask_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_mask = (target_mask_u8 > 0).astype(np.uint8)
    h, w = target_mask.shape[:2]
    if np.count_nonzero(target_mask) == 0:
        empty = np.zeros((h, w), dtype=np.float32)
        return empty, empty

    hsv = cv2.cvtColor(target_roi_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(target_roi_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(target_roi_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    lightness = lab[:, :, 0] / 255.0
    hue = hsv[:, :, 0]

    pale_sky = (lightness > 0.54) & (saturation < 0.36)
    blue_sky = (hue >= 76.0) & (hue <= 132.0) & (saturation > 0.08) & (value > 0.28)

    upper_bias = np.zeros((h, w), dtype=np.float32)
    upper_bias[: max(1, int(h * 0.72)), :] = 1.0
    vertical_falloff = np.linspace(1.0, 0.42, h, dtype=np.float32)[:, np.newaxis]

    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    horizontal_texture = np.abs(sobel_y)
    max_texture = float(horizontal_texture.max())
    if max_texture > 1e-6:
        horizontal_texture /= max_texture
    blind_lines = horizontal_texture * np.clip((0.22 - saturation) / 0.22, 0.0, 1.0) * np.clip((lightness - 0.54) / 0.28, 0.0, 1.0)
    frame_gate = np.clip((0.16 - saturation) / 0.16, 0.0, 1.0) * np.clip((value - 0.70) / 0.20, 0.0, 1.0)

    transmission = np.clip(
        ((pale_sky | blue_sky).astype(np.float32) * 0.70)
        + upper_bias * vertical_falloff * 0.34
        + np.clip((lightness - 0.44) / 0.36, 0.0, 1.0) * 0.14,
        0.0,
        1.0,
    )
    preserve_mask = np.clip(blind_lines * 0.78 + frame_gate * 0.88, 0.0, 1.0)
    transmission = np.clip(transmission * (1.0 - preserve_mask * 0.32), 0.0, 1.0)

    transmission = cv2.GaussianBlur(transmission.astype(np.float32), (0, 0), 1.0)
    preserve_mask = cv2.GaussianBlur(preserve_mask.astype(np.float32), (0, 0), 0.8)
    transmission[target_mask == 0] = 0.0
    preserve_mask[target_mask == 0] = 0.0
    return transmission.astype(np.float32), preserve_mask.astype(np.float32)


def _infer_reference_crop_sky_mask(
    reference_crop: np.ndarray,
    reference_component_mask_u8: np.ndarray,
    weather: str,
) -> np.ndarray:
    component_mask = (reference_component_mask_u8 > 0).astype(np.uint8) * 255
    if np.count_nonzero(component_mask) == 0:
        return np.zeros(reference_crop.shape[:2], dtype=np.uint8)

    h, w = component_mask.shape[:2]
    hsv = cv2.cvtColor(reference_crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(reference_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    lightness = lab[:, :, 0] / 255.0
    hue = hsv[:, :, 0]

    pale_sky = (lightness > 0.54) & (saturation < 0.34) & (value > 0.34)
    blue_sky = (hue >= 76.0) & (hue <= 132.0) & (saturation > 0.08) & (value > 0.28)
    gray_cloud = (lightness > 0.48) & (saturation < 0.18) & (value > 0.30)

    if weather == "cloudy":
        sky_like = gray_cloud | pale_sky | blue_sky
        top_ratio = 0.56
    elif weather == "clear":
        sky_like = blue_sky | pale_sky
        top_ratio = 0.58
    else:
        sky_like = pale_sky | blue_sky
        top_ratio = 0.58

    upper_band = np.zeros((h, w), dtype=np.uint8)
    upper_band[: max(1, int(h * (0.74 if weather == "cloudy" else 0.82))), :] = 255
    candidate = (sky_like & (component_mask > 0) & (upper_band > 0)).astype(np.uint8) * 255
    candidate = _connected_to_top_mask(candidate)

    if np.count_nonzero(candidate) == 0:
        component_dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        inner_gate = np.clip(component_dist / max(4.0, min(h, w) * 0.05), 0.0, 1.0)
        fallback = np.zeros((h, w), dtype=np.uint8)
        fallback[: max(1, int(h * top_ratio)), :] = 255
        candidate = ((fallback > 0) & (inner_gate > 0.15) & (component_mask > 0)).astype(np.uint8) * 255

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return candidate


def _fallback_secondary_sky_candidate(visibility: np.ndarray, target_mask_u8: np.ndarray) -> np.ndarray:
    h, w = target_mask_u8.shape[:2]
    upper_band = np.zeros((h, w), dtype=np.uint8)
    upper_band[: max(1, int(h * 0.58)), :] = 255
    candidate = ((visibility > 0.08) & (upper_band > 0) & (target_mask_u8 > 0)).astype(np.uint8) * 255
    candidate = _connected_to_top_mask(candidate)
    return candidate


def _transfer_view_appearance(
    target_roi_u8: np.ndarray,
    visibility: np.ndarray,
    reference_roi_u8: np.ndarray,
    reference_visibility: np.ndarray,
    weather: str,
) -> np.ndarray:
    target_mask_u8 = (visibility > 0.10).astype(np.uint8) * 255
    reference_mask_u8 = (reference_visibility > 0.10).astype(np.uint8) * 255
    target_stats = _masked_lab_mean_std(target_roi_u8, target_mask_u8)
    reference_stats = _masked_lab_mean_std(reference_roi_u8, reference_mask_u8)
    if target_stats is None or reference_stats is None:
        return target_roi_u8.copy()

    target_mean, target_std = target_stats
    reference_mean, reference_std = reference_stats
    lab = cv2.cvtColor(target_roi_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    transformed = lab.copy()
    valid = target_mask_u8 > 0
    transformed_pixels = transformed[valid]
    transformed_pixels = ((transformed_pixels - target_mean) / target_std) * reference_std + reference_mean
    transformed[valid] = transformed_pixels

    if weather == "sunrise":
        transformed[:, :, 1] += visibility * 1.8
        transformed[:, :, 2] += visibility * 7.0
    elif weather == "sunset":
        transformed[:, :, 1] += visibility * 1.2
        transformed[:, :, 2] += visibility * 5.4
    elif weather == "cloudy":
        transformed[:, :, 1] -= visibility * 0.8
        transformed[:, :, 2] -= visibility * 1.1
        transformed[:, :, 0] -= visibility * 3.2
    elif weather == "clear":
        transformed[:, :, 0] += visibility * 0.5
        transformed[:, :, 1] -= visibility * 0.2
        transformed[:, :, 2] -= visibility * 2.4

    transformed = np.clip(transformed, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(transformed, cv2.COLOR_LAB2BGR)


def _build_reference_replacement_layer(
    base_image: np.ndarray,
    reference_image: np.ndarray | None,
    opening_mask: np.ndarray,
    reference_sky_mask: np.ndarray | None = None,
    weather: str = "clear",
) -> np.ndarray | None:
    if reference_image is None or np.count_nonzero(opening_mask) == 0:
        return None

    reference_opening_mask = _build_reference_opening_mask(reference_image)
    target_components = _extract_mask_components(opening_mask, min_area_ratio=0.0035)
    reference_components = _extract_mask_components(reference_opening_mask, min_area_ratio=0.0035)
    if not target_components or not reference_components:
        return None

    layer = base_image.copy()
    image_h, image_w = base_image.shape[:2]
    image_area = float(max(image_h * image_w, 1))
    unmatched_reference_components = [dict(component) for component in reference_components]

    for index, target_component in enumerate(target_components):
        candidate_pool = unmatched_reference_components if unmatched_reference_components else reference_components
        target_cx = (float(target_component["x"]) + float(target_component["w"]) * 0.5) / float(max(image_w, 1))
        target_cy = (float(target_component["y"]) + float(target_component["h"]) * 0.5) / float(max(image_h, 1))
        target_area_ratio = float(target_component["area"]) / image_area
        target_aspect = float(target_component["w"]) / float(max(float(target_component["h"]), 1.0))

        best_match_index = 0
        best_match_score = float("inf")
        for candidate_index, candidate_component in enumerate(candidate_pool):
            candidate_cx = (float(candidate_component["x"]) + float(candidate_component["w"]) * 0.5) / float(max(image_w, 1))
            candidate_cy = (float(candidate_component["y"]) + float(candidate_component["h"]) * 0.5) / float(max(image_h, 1))
            candidate_area_ratio = float(candidate_component["area"]) / image_area
            candidate_aspect = float(candidate_component["w"]) / float(max(float(candidate_component["h"]), 1.0))

            center_score = abs(target_cx - candidate_cx) * 2.2 + abs(target_cy - candidate_cy) * 1.1
            area_score = abs(np.log(max(target_area_ratio, 1e-6) / max(candidate_area_ratio, 1e-6))) * 0.35
            aspect_score = abs(np.log(max(target_aspect, 1e-6) / max(candidate_aspect, 1e-6))) * 0.55
            match_score = center_score + area_score + aspect_score
            if match_score < best_match_score:
                best_match_score = match_score
                best_match_index = candidate_index

        reference_component = candidate_pool[best_match_index]
        if unmatched_reference_components:
            unmatched_reference_components.pop(best_match_index)
        tx = int(target_component["x"])
        ty = int(target_component["y"])
        tw = int(target_component["w"])
        th = int(target_component["h"])
        rx = int(reference_component["x"])
        ry = int(reference_component["y"])
        rw = int(reference_component["w"])
        rh = int(reference_component["h"])
        if tw <= 1 or th <= 1 or rw <= 1 or rh <= 1:
            continue

        reference_crop = reference_image[ry:ry + rh, rx:rx + rw]
        if reference_crop.size == 0:
            continue
        fitted_crop = _fit_cover(reference_crop, tw, th, focus_x=0.50, focus_y=0.56)

        target_mask_u8 = np.asarray(target_component["mask"], dtype=np.uint8)[ty:ty + th, tx:tx + tw]
        target_roi_u8 = base_image[ty:ty + th, tx:tx + tw]
        visibility = _estimate_opening_visibility(target_roi_u8, target_mask_u8)
        target_dist = cv2.distanceTransform((target_mask_u8 > 0).astype(np.uint8) * 255, cv2.DIST_L2, 5)
        edge_trim = max(2.0, min(tw, th) * 0.028)
        inner_gate = np.clip((target_dist - 1.5) / edge_trim, 0.0, 1.0)

        reference_mask_u8 = np.asarray(reference_component["mask"], dtype=np.uint8)[ry:ry + rh, rx:rx + rw]
        reference_visibility = _estimate_opening_visibility(reference_crop, reference_mask_u8)
        inferred_reference_sky_crop_mask = _infer_reference_crop_sky_mask(
            reference_crop,
            reference_mask_u8,
            weather,
        )
        transformed_outdoor = _transfer_view_appearance(
            target_roi_u8=target_roi_u8,
            visibility=visibility,
            reference_roi_u8=reference_crop,
            reference_visibility=reference_visibility,
            weather=weather,
        )

        sky_candidate = _build_opening_sky_candidate(target_roi_u8, visibility, target_mask_u8)
        opening_area_ratio = (tw * th) / image_area
        is_secondary_opening = index > 0 or opening_area_ratio < 0.045
        secondary_transmission = None
        secondary_preserve = None
        if is_secondary_opening:
            secondary_transmission, secondary_preserve = _estimate_secondary_opening_transmission(
                target_roi_u8,
                target_mask_u8,
            )
        if is_secondary_opening and np.count_nonzero(sky_candidate) < max(36, int(tw * th * 0.02)):
            sky_candidate = _fallback_secondary_sky_candidate(visibility, target_mask_u8)
        sky_alpha = cv2.GaussianBlur((sky_candidate.astype(np.float32) / 255.0) * visibility, (0, 0), 1.8)
        if is_secondary_opening and secondary_transmission is not None:
            sky_alpha = np.maximum(
                sky_alpha,
                cv2.GaussianBlur(secondary_transmission.astype(np.float32), (0, 0), 1.2) * 0.92,
            )
        clear_transfer_mask = None
        fitted_sky_mask = None

        styled_roi = transformed_outdoor.astype(np.float32)
        direct_reference_strength = 0.0
        if is_secondary_opening:
            if weather == "sunrise":
                direct_reference_strength = 0.58
            elif weather == "sunset":
                direct_reference_strength = 0.52
            elif weather == "cloudy":
                direct_reference_strength = 0.22
            elif weather == "clear":
                direct_reference_strength = 0.0
        elif weather in {"sunrise", "sunset"}:
            direct_reference_strength = 0.10
        elif weather == "cloudy":
            direct_reference_strength = 0.0
        elif weather == "clear":
            direct_reference_strength = 0.0

        if direct_reference_strength > 0.0:
            direct_ref_mask = visibility.astype(np.float32)
            if is_secondary_opening and secondary_transmission is not None:
                direct_ref_mask = np.maximum(direct_ref_mask, secondary_transmission * 1.00)
                if secondary_preserve is not None:
                    direct_ref_mask *= 1.0 - secondary_preserve * 0.10
            direct_ref_alpha = cv2.GaussianBlur((direct_ref_mask * direct_reference_strength).astype(np.float32), (0, 0), 0.8)
            styled_roi = styled_roi * (1.0 - direct_ref_alpha[..., None]) + fitted_crop.astype(np.float32) * direct_ref_alpha[..., None]
            if is_secondary_opening and secondary_transmission is not None and weather in {"sunrise", "sunset"}:
                styled_roi_lab = cv2.cvtColor(np.clip(styled_roi, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
                warm_scale = 7.5 if weather == "sunrise" else 10.0
                styled_roi_lab[:, :, 2] = np.clip(styled_roi_lab[:, :, 2] + secondary_transmission * warm_scale, 0.0, 255.0)
                styled_roi_lab[:, :, 1] = np.clip(styled_roi_lab[:, :, 1] + secondary_transmission * (warm_scale * 0.18), 0.0, 255.0)
                styled_roi = cv2.cvtColor(styled_roi_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

        if reference_sky_mask is not None and reference_sky_mask.shape[:2] == reference_image.shape[:2]:
            reference_sky_crop_mask = reference_sky_mask[ry:ry + rh, rx:rx + rw]
            if reference_sky_crop_mask.size > 0 and np.count_nonzero(reference_sky_crop_mask) > 0:
                reference_sky_crop_mask = cv2.bitwise_or(reference_sky_crop_mask, inferred_reference_sky_crop_mask)
            else:
                reference_sky_crop_mask = inferred_reference_sky_crop_mask
            if reference_sky_crop_mask.size > 0 and np.count_nonzero(reference_sky_crop_mask) > 0:
                fitted_sky_mask = cv2.resize(reference_sky_crop_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                fitted_sky_mask = cv2.GaussianBlur((fitted_sky_mask > 0).astype(np.float32), (0, 0), 1.6)
                ref_sky_alpha = sky_alpha * fitted_sky_mask
                if weather == "cloudy":
                    ref_sky_alpha = np.maximum(
                        ref_sky_alpha,
                        fitted_sky_mask * np.clip(inner_gate * 0.72 + visibility * 0.34, 0.0, 1.0),
                    )
                if is_secondary_opening and secondary_preserve is not None:
                    ref_sky_alpha = np.maximum(ref_sky_alpha, secondary_transmission * 0.88 if secondary_transmission is not None else ref_sky_alpha)
                    ref_sky_alpha *= 1.0 - secondary_preserve * 0.08
                styled_roi = styled_roi * (1.0 - ref_sky_alpha[..., None]) + fitted_crop.astype(np.float32) * ref_sky_alpha[..., None]
        else:
            if np.count_nonzero(inferred_reference_sky_crop_mask) > 0:
                fitted_sky_mask = cv2.resize(inferred_reference_sky_crop_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                fitted_sky_mask = cv2.GaussianBlur((fitted_sky_mask > 0).astype(np.float32), (0, 0), 1.6)
            ref_sky_alpha = sky_alpha
            if weather == "cloudy":
                ref_sky_alpha = np.maximum(
                    ref_sky_alpha,
                    np.clip(inner_gate * 0.32 + visibility * 0.40, 0.0, 0.88),
                )
                if fitted_sky_mask is not None:
                    ref_sky_alpha = np.maximum(
                        ref_sky_alpha,
                        fitted_sky_mask * np.clip(inner_gate * 0.72 + visibility * 0.34, 0.0, 1.0),
                    )
            if is_secondary_opening and secondary_preserve is not None:
                ref_sky_alpha = np.maximum(ref_sky_alpha, secondary_transmission * 0.88 if secondary_transmission is not None else ref_sky_alpha)
                ref_sky_alpha *= 1.0 - secondary_preserve * 0.08
            styled_roi = styled_roi * (1.0 - ref_sky_alpha[..., None]) + fitted_crop.astype(np.float32) * ref_sky_alpha[..., None]

        cloudy_transfer_mask = None
        if weather == "cloudy":
            upper_focus = np.linspace(1.0, 0.46, th, dtype=np.float32)[:, np.newaxis]
            cloudy_transfer_mask = np.clip(inner_gate * (0.40 + visibility * 0.60), 0.0, 1.0)
            if fitted_sky_mask is not None:
                cloudy_transfer_mask = np.maximum(
                    cloudy_transfer_mask,
                    fitted_sky_mask * np.clip(0.84 + upper_focus * 0.16, 0.0, 1.0),
                )
            if is_secondary_opening and secondary_transmission is not None:
                cloudy_transfer_mask = np.maximum(
                    cloudy_transfer_mask,
                    secondary_transmission * np.clip(0.86 + upper_focus * 0.10, 0.0, 1.0),
                )
                if secondary_preserve is not None:
                    cloudy_transfer_mask *= 1.0 - secondary_preserve * 0.18

            styled_lab = cv2.cvtColor(np.clip(styled_roi, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            cloudy_sky_focus = np.clip(cloudy_transfer_mask * (0.52 + upper_focus * 0.48), 0.0, 1.0)
            styled_lab[:, :, 0] = np.clip(styled_lab[:, :, 0] + cloudy_sky_focus * 3.4, 0.0, 255.0)
            styled_lab[:, :, 1] = np.clip(styled_lab[:, :, 1] - cloudy_sky_focus * 1.0, 0.0, 255.0)
            styled_lab[:, :, 2] = np.clip(styled_lab[:, :, 2] - cloudy_sky_focus * 3.8, 0.0, 255.0)
            styled_roi = cv2.cvtColor(styled_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

        if weather == "clear":
            clear_transfer_mask = cv2.GaussianBlur((sky_candidate.astype(np.float32) / 255.0), (0, 0), 1.0) * 0.24
            target_sky_gate = cv2.GaussianBlur((sky_candidate > 0).astype(np.float32), (0, 0), 2.2)
            target_sky_gate = np.clip(target_sky_gate * 1.85, 0.0, 1.0)
            if fitted_sky_mask is not None:
                clear_transfer_mask = np.maximum(clear_transfer_mask, fitted_sky_mask * target_sky_gate * 0.92)
            clear_transfer_mask *= np.clip(visibility * 1.12, 0.0, 1.0)
            clear_transfer_mask *= inner_gate
            if is_secondary_opening and secondary_transmission is not None:
                upper_focus = np.linspace(1.0, 0.08, th, dtype=np.float32)[:, np.newaxis]
                clear_transfer_mask = np.maximum(
                    clear_transfer_mask,
                    secondary_transmission * upper_focus * 0.12,
                )
            if is_secondary_opening and secondary_preserve is not None:
                clear_transfer_mask *= 1.0 - secondary_preserve * 0.22
            clear_transfer_mask = cv2.GaussianBlur(np.clip(clear_transfer_mask, 0.0, 1.0), (0, 0), 0.9)
            clear_support = cv2.dilate(
                (sky_candidate > 0).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            ).astype(np.float32) / 255.0
            clear_support = cv2.GaussianBlur(clear_support, (0, 0), 0.7)
            clear_transfer_mask *= np.clip(clear_support * 1.08, 0.0, 1.0)
            clear_boost = clear_transfer_mask
            if float(clear_boost.max()) > 1e-4:
                vertical_focus = np.linspace(1.0, 0.42, th, dtype=np.float32)[:, np.newaxis]
                clear_sky_focus = np.clip(clear_boost * (0.52 + vertical_focus * 0.68), 0.0, 1.0)
                styled_hsv = cv2.cvtColor(np.clip(styled_roi, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
                sky_like = clear_sky_focus > 0.06
                styled_hsv[:, :, 0] = np.where(sky_like, np.clip(styled_hsv[:, :, 0], 96.0, 112.0), styled_hsv[:, :, 0])
                styled_hsv[:, :, 1] = np.clip(
                    styled_hsv[:, :, 1] * (1.0 + clear_sky_focus * 0.10) + clear_sky_focus * 4.2,
                    0.0,
                    255.0,
                )
                styled_hsv[:, :, 2] = np.clip(styled_hsv[:, :, 2] - clear_sky_focus * 2.0, 0.0, 255.0)
                styled_roi = cv2.cvtColor(styled_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
                styled_lab = cv2.cvtColor(np.clip(styled_roi, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
                styled_lab[:, :, 2] = np.clip(styled_lab[:, :, 2] - clear_sky_focus * 2.4, 0.0, 255.0)
                styled_roi = cv2.cvtColor(styled_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

        target_mask = visibility.astype(np.float32)
        if weather == "clear" and clear_transfer_mask is not None:
            target_mask = clear_transfer_mask
        elif weather == "cloudy" and cloudy_transfer_mask is not None:
            target_mask = np.maximum(target_mask, cloudy_transfer_mask * 0.98)
        elif is_secondary_opening and secondary_transmission is not None:
            target_mask = np.maximum(target_mask, secondary_transmission * 0.98)
            if secondary_preserve is not None:
                target_mask *= 1.0 - secondary_preserve * 0.08
        target_mask = cv2.GaussianBlur(target_mask.astype(np.float32), (0, 0), 0.9)
        target_roi = layer[ty:ty + th, tx:tx + tw].astype(np.float32)
        blended_roi = target_roi * (1.0 - target_mask[..., None]) + styled_roi * target_mask[..., None]
        layer[ty:ty + th, tx:tx + tw] = np.clip(blended_roi, 0, 255).astype(np.uint8)

    return layer


def _build_sky_alpha(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(binary) == 0:
        return np.zeros(binary.shape, dtype=np.float32)

    # Smaller close kernel to preserve the more accurate mask details
    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    # Crisper feather: 5px ramp + 1.5σ blur
    alpha = np.clip(dist / 5.0, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)
    alpha[cleaned == 0] = 0.0
    return alpha.astype(np.float32)


def _build_clear_replacement_mask(image: np.ndarray, sky_mask: np.ndarray, opening_mask: np.ndarray) -> np.ndarray:
    sky_binary = (sky_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(sky_binary) == 0:
        return sky_binary

    if np.count_nonzero(opening_mask) > 0:
        original_sky_binary = sky_binary.copy()
        opening_binary = (opening_mask > 0).astype(np.uint8) * 255
        sky_binary = cv2.bitwise_and(sky_binary, opening_binary)
        if np.count_nonzero(sky_binary) == 0:
            sky_binary = original_sky_binary

    refined = np.zeros_like(sky_binary)
    components = _extract_mask_components(sky_binary, min_area_ratio=0.0005)
    for component in components:
        x = int(component["x"])
        y = int(component["y"])
        w = int(component["w"])
        h = int(component["h"])
        if w < 4 or h < 4:
            continue

        component_mask = np.asarray(component["mask"], dtype=np.uint8)
        roi_mask = component_mask[y:y + h, x:x + w]
        roi_image = image[y:y + h, x:x + w]
        candidate, blue = _clear_sky_pixel_candidate(roi_image, roi_mask)
        candidate = _trim_clear_sky_to_opening(candidate, blue, roi_mask, roi_image)
        if np.count_nonzero(candidate) == 0:
            continue
        refined[y:y + h, x:x + w] = cv2.bitwise_or(refined[y:y + h, x:x + w], candidate)

    if np.count_nonzero(refined) > 0:
        sky_binary = refined

    sky_binary = cv2.morphologyEx(
        sky_binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return sky_binary


def _restore_clear_non_sky_opening(
    original: np.ndarray,
    result: np.ndarray,
    opening_mask: np.ndarray,
    replacement_mask: np.ndarray,
) -> np.ndarray:
    opening_binary = (opening_mask > 0).astype(np.uint8) * 255
    sky_binary = (replacement_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(opening_binary) == 0 or np.count_nonzero(sky_binary) == 0:
        return result

    sky_guard = cv2.dilate(
        sky_binary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    restore_mask = cv2.bitwise_and(opening_binary, cv2.bitwise_not(sky_guard))
    if np.count_nonzero(restore_mask) == 0:
        return result

    restore_mask = cv2.morphologyEx(
        restore_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    alpha = cv2.GaussianBlur(restore_mask.astype(np.float32) / 255.0, (0, 0), 0.8)
    alpha = np.clip(alpha * 1.18, 0.0, 1.0)
    restored = result.astype(np.float32) * (1.0 - alpha[..., None]) + original.astype(np.float32) * alpha[..., None]
    return np.clip(restored, 0, 255).astype(np.uint8)


def _filter_opening_mask_by_sky(opening_mask: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    opening_binary = (opening_mask > 0).astype(np.uint8) * 255
    sky_binary = (sky_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(opening_binary) == 0 or np.count_nonzero(sky_binary) == 0:
        return opening_binary

    h, w = opening_binary.shape[:2]
    support_kernel_size = max(15, int(round(min(h, w) * 0.018))) | 1
    sky_support = cv2.dilate(
        sky_binary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (support_kernel_size, support_kernel_size)),
        iterations=1,
    )
    components = _extract_mask_components(opening_binary, min_area_ratio=0.0012)
    if not components:
        return opening_binary

    filtered = np.zeros_like(opening_binary)
    for component in components:
        component_mask = np.asarray(component["mask"], dtype=np.uint8)
        area = max(1.0, float(component["area"]))
        support_overlap = int(np.count_nonzero(cv2.bitwise_and(component_mask, sky_support)))
        direct_overlap = int(np.count_nonzero(cv2.bitwise_and(component_mask, sky_binary)))
        if direct_overlap >= max(24, int(area * 0.002)) or support_overlap >= max(48, int(area * 0.006)):
            filtered = cv2.bitwise_or(filtered, component_mask)

    if np.count_nonzero(filtered) == 0:
        return opening_binary
    return filtered


def _normalized_blur(mask: np.ndarray, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    max_value = float(blurred.max())
    if max_value <= 1e-6:
        return blurred
    return blurred / max_value


def _build_directional_sun_sweep(
    image_shape: tuple[int, int],
    origin_x: float,
    origin_y: float,
    target_x: float,
    target_y: float,
    thickness: float,
) -> np.ndarray:
    h, w = image_shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vx = float(target_x - origin_x)
    vy = float(target_y - origin_y)
    denom = max(vx * vx + vy * vy, 1e-6)
    t = ((xx - origin_x) * vx + (yy - origin_y) * vy) / denom
    t = np.clip(t, 0.0, 1.0)
    proj_x = origin_x + t * vx
    proj_y = origin_y + t * vy
    dist_sq = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
    width_sq = max(thickness * thickness, 1.0)
    sweep = np.exp(-dist_sq / (2.0 * width_sq))
    longitudinal = 0.52 + 0.48 * (1.0 - t)
    return np.clip(sweep * longitudinal, 0.0, 1.0).astype(np.float32)


def _build_opening_local_glow(
    image_shape: tuple[int, int],
    openings: list[tuple[int, int, int, int]],
) -> np.ndarray:
    h, w = image_shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    glow = np.zeros((h, w), dtype=np.float32)
    if not openings:
        return glow

    for x, y, opening_w, opening_h in openings:
        center_x = x + opening_w * 0.50
        center_y = y + opening_h * 0.60
        sigma_x = max(opening_w * 0.70, w * 0.05)
        sigma_y = max(opening_h * 1.10, h * 0.07)
        local = np.exp(-(((xx - center_x) ** 2) / (2.0 * sigma_x ** 2) + ((yy - center_y) ** 2) / (2.0 * sigma_y ** 2)))
        glow = np.maximum(glow, local.astype(np.float32))

    return np.clip(glow, 0.0, 1.0)


def _build_floor_beam_mask(image_shape: tuple[int, int], openings: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = image_shape
    beam_mask = np.zeros((h, w), dtype=np.float32)
    if not openings:
        return beam_mask

    for x, y, opening_w, opening_h in openings:
        top = int(np.clip(y + opening_h * 0.58, 0, h - 1))
        bottom = h - 1
        spread = max(30, int(opening_w * 0.60))
        polygon = np.array(
            [
                [x + opening_w * 0.12, top],
                [x + opening_w * 0.88, top],
                [min(w - 1, x + opening_w + spread), bottom],
                [max(0, x - spread * 0.35), bottom],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(beam_mask, polygon, 1.0)

    beam_mask = cv2.GaussianBlur(beam_mask, (0, 0), 26.0)
    beam_max = float(beam_mask.max())
    if beam_max <= 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    return beam_mask / beam_max


def _build_opening_protection_mask(image_shape: tuple[int, int], openings: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = image_shape
    protection = np.zeros((h, w), dtype=np.float32)
    if not openings:
        return protection

    for x, y, opening_w, opening_h in openings:
        extension = 2.70 if opening_h < h * 0.24 else 1.12
        pad_x = max(3, int(opening_w * 0.04))
        pad_y = max(2, int(opening_h * 0.03))
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(w, int(x + opening_w + pad_x))
        y2 = min(h, int(y + opening_h * extension))
        if x2 > x1 and y2 > y1:
            protection[y1:y2, x1:x2] = 1.0

    protection = cv2.GaussianBlur(protection, (0, 0), max(2.0, min(h, w) * 0.0025))
    return np.clip(protection, 0.0, 1.0)


def _build_floor_cleanup_zone(image_shape: tuple[int, int], openings: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = image_shape
    zone = np.zeros((h, w), dtype=np.uint8)
    if not openings:
        return zone

    for x, y, opening_w, opening_h in openings:
        threshold_y = y + opening_h * (2.62 if opening_h < h * 0.24 else 0.98)
        top_y = int(np.clip(threshold_y - h * 0.055, h * 0.46, h - 2))
        top_spread = max(opening_w * 0.18, w * 0.025)
        bottom_spread = max(opening_w * 1.10, h * 0.24)
        polygon = np.array(
            [
                [max(0, int(x - top_spread)), top_y],
                [min(w - 1, int(x + opening_w + top_spread)), top_y],
                [min(w - 1, int(x + opening_w + bottom_spread)), h - 1],
                [max(0, int(x - bottom_spread)), h - 1],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(zone, polygon, 255)

    return zone


def _estimate_opening_light_profile(
    image: np.ndarray,
    opening: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    image_h, image_w = image.shape[:2]
    x, y, opening_w, opening_h = opening
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(image_w, int(x + opening_w))
    y2 = min(image_h, int(y + opening_h))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), max(1.0, min(gray.shape[:2]) * 0.012))

    crop_x1 = int(gray.shape[1] * 0.04)
    crop_x2 = max(crop_x1 + 1, int(gray.shape[1] * 0.96))
    crop_y1 = int(gray.shape[0] * 0.08)
    crop_y2 = max(crop_y1 + 1, int(gray.shape[0] * 0.98))
    gray = gray[crop_y1:crop_y2, crop_x1:crop_x2]
    if gray.size == 0:
        return None

    low = float(np.quantile(gray, 0.12))
    high = float(np.quantile(gray, 0.90))
    if high - low < 1e-4:
        return None

    normalized = np.clip((gray - low) / max(high - low, 1e-4), 0.0, 1.0)
    normalized = cv2.GaussianBlur(normalized, (0, 0), max(0.8, min(normalized.shape[:2]) * 0.010))

    upper = normalized[:max(1, int(normalized.shape[0] * 0.58)), :]
    lower = normalized[int(normalized.shape[0] * 0.42):, :]

    coarse = cv2.GaussianBlur(normalized, (0, 0), max(1.4, min(normalized.shape[:2]) * 0.040))
    detail = np.clip(normalized - coarse * 0.52, 0.0, 1.0)
    transmittance = np.clip(coarse * 0.72 + detail * 1.18, 0.0, 1.0)
    transmittance *= np.linspace(1.0, 0.90, transmittance.shape[0], dtype=np.float32)[:, np.newaxis]

    upper_profile = upper.mean(axis=0)
    lower_profile = lower.mean(axis=0)
    column_profile = np.clip(upper_profile * 0.40 + lower_profile * 0.60, 0.0, 1.0)

    profile_low = float(np.quantile(column_profile, 0.25))
    profile_high = float(np.quantile(column_profile, 0.85))
    column_profile = np.clip(
        (column_profile - profile_low) / max(profile_high - profile_low, 1e-4),
        0.0,
        1.0,
    ) ** 1.15
    column_profile = cv2.GaussianBlur(
        column_profile[np.newaxis, :],
        (0, 0),
        max(1.0, column_profile.shape[0] * 0.018),
    ).reshape(-1)

    weights = np.maximum(upper ** 2.0, 1e-6)
    x_coords = np.linspace(0.0, 1.0, weights.shape[1], dtype=np.float32)
    column_weights = weights.mean(axis=0)
    sun_bias = float((x_coords * column_weights).sum() / max(column_weights.sum(), 1e-6) - 0.5)
    openness = float(np.clip(np.quantile(normalized, 0.80), 0.25, 1.0))

    if float(column_profile.max()) <= 1e-4 or float(transmittance.max()) <= 1e-4:
        return None

    return (
        transmittance.astype(np.float32),
        column_profile.astype(np.float32),
        float(np.clip(sun_bias, -0.45, 0.45)),
        openness,
    )


def _build_projected_ground_light(
    image: np.ndarray,
    openings: list[tuple[int, int, int, int]],
    weather: str,
    reference_image: np.ndarray | None = None,
) -> np.ndarray:
    h, w = image.shape[:2]
    projection = np.zeros((h, w), dtype=np.float32)
    if weather not in PROJECTED_GROUND_LIGHT_WEATHERS or not openings:
        return projection

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    floor_gate = np.clip((yy - h * 0.50) / max(h * 0.34, 1.0), 0.0, 1.0)

    for index, opening in enumerate(sorted(openings, key=lambda item: item[2] * item[3], reverse=True)):
        if index >= 2:
            break

        profile_result = _estimate_opening_light_profile(image, opening)
        reference_profile_result = None
        if reference_image is not None and reference_image.shape[:2] == image.shape[:2]:
            reference_profile_result = _estimate_opening_light_profile(reference_image, opening)

        if profile_result is None and reference_profile_result is None:
            x, y, opening_w, opening_h = opening
            column_profile = np.ones(max(16, int(opening_w)), dtype=np.float32)
            sun_bias = 0.16 if weather == "sunrise" else 0.22
            openness = 0.70
        elif profile_result is None:
            _, column_profile, sun_bias, openness = reference_profile_result
        elif reference_profile_result is None:
            _, column_profile, sun_bias, openness = profile_result
        else:
            _, target_profile, target_bias, target_openness = profile_result
            _, reference_profile, reference_bias, reference_openness = reference_profile_result
            reference_profile = cv2.resize(
                reference_profile.reshape(1, -1),
                (target_profile.shape[0], 1),
                interpolation=cv2.INTER_CUBIC,
            ).reshape(-1)
            column_profile = np.clip(target_profile * 0.58 + reference_profile * 0.42, 0.0, 1.0)
            sun_bias = float(np.clip(target_bias * 0.38 + reference_bias * 0.62, -0.48, 0.48))
            openness = float(np.clip(max(target_openness, reference_openness), 0.25, 1.0))

        x, y, opening_w, opening_h = opening
        if column_profile.size < 8 or opening_w < 8 or opening_h < 8:
            continue

        threshold_y = y + opening_h * (2.62 if opening_h < h * 0.24 else 0.98)
        threshold_y = float(np.clip(threshold_y, h * 0.48, h * 0.84))
        source_x = x + opening_w * np.clip(0.50 + sun_bias * 0.18, 0.34, 0.66)
        distance_y = np.maximum(yy - threshold_y, 0.0)
        starts_on_floor = np.clip((yy - threshold_y + h * 0.018) / max(h * 0.08, 1.0), 0.0, 1.0)

        lateral_center = source_x - sun_bias * distance_y * 0.46
        spread_x = opening_w * (0.95 + openness * 0.30) + distance_y * (0.88 if weather == "sunrise" else 1.02)
        spread_x = np.maximum(spread_x, opening_w * 0.55)
        length_y = max(h * (0.40 if weather == "sunrise" else 0.48), opening_h * 1.60)

        x_term = ((xx - lateral_center) / spread_x) ** 2
        y_term = (distance_y / length_y) ** 2
        fan = np.exp(-(x_term * 1.08 + y_term * 1.62)) * starts_on_floor

        near_spread_x = max(opening_w * 0.92, w * 0.045)
        near_length = max(h * 0.12, opening_h * 0.55)
        contact = np.exp(-(((xx - source_x) / near_spread_x) ** 2 + (distance_y / near_length) ** 2) * 1.25) * starts_on_floor

        side_spread = opening_w * 1.70 + distance_y * 1.28
        ambient_spill = np.exp(-((xx - source_x) / np.maximum(side_spread, 1.0)) ** 2) * np.exp(-distance_y / max(h * 0.55, 1.0)) * starts_on_floor

        profile_low = float(np.quantile(column_profile, 0.12))
        profile_high = float(np.quantile(column_profile, 0.88))
        if profile_high - profile_low > 1e-4:
            column_profile = np.clip((column_profile - profile_low) / (profile_high - profile_low), 0.0, 1.0)
        column_profile = cv2.GaussianBlur(
            column_profile.reshape(1, -1).astype(np.float32),
            (0, 0),
            max(1.0, column_profile.shape[0] * 0.018),
        ).reshape(-1)
        shadow_line = cv2.resize(
            column_profile.reshape(1, -1),
            (w, 1),
            interpolation=cv2.INTER_CUBIC,
        ).reshape(-1)
        shadow_line = cv2.GaussianBlur(shadow_line.reshape(1, -1), (0, 0), max(3.0, w * 0.006)).reshape(-1)
        shadow_modulation = 0.88 + 0.12 * shadow_line[np.newaxis, :]

        weighted_projection = (
            fan * (0.62 + openness * 0.22)
            + contact * (0.24 + openness * 0.06)
            + ambient_spill * 0.12
        ) * shadow_modulation
        projection = np.maximum(projection, np.clip(weighted_projection, 0.0, 1.0))

    projection *= floor_gate
    floor_falloff = np.clip(1.0 - (yy / max(h, 1)) * 0.14, 0.0, 1.0)
    projection *= floor_falloff
    projection = cv2.GaussianBlur(
        projection,
        (0, 0),
        max(14.0, min(h, w) * 0.016),
    )
    return np.clip(projection * (0.72 if weather == "sunset" else 0.68), 0.0, 1.0)


def _filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def _expand_floor_patch_components(mask: np.ndarray, image_shape: tuple[int, int], padding: int) -> np.ndarray:
    h, w = image_shape
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    expanded = np.zeros_like(mask)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_w = int(stats[label, cv2.CC_STAT_WIDTH])
        component_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if component_w < 8 or component_h < 5:
            continue

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + component_w + padding)
        y2 = min(h, y + component_h + padding)
        expanded[y1:y2, x1:x2] = 255

    return cv2.bitwise_or(mask, expanded)


def _suppress_floor_light_patches(
    image: np.ndarray,
    openings: list[tuple[int, int, int, int]],
    weather: str | None = None,
    protected_mask: np.ndarray | None = None,
) -> np.ndarray:
    if not openings:
        return image

    h, w = image.shape[:2]
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    if weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        floor_gate = np.clip((yy - h * 0.48) / max(h * 0.42, 1.0), 0.0, 1.0)
    else:
        floor_gate = np.clip((yy - h * 0.56) / max(h * 0.44, 1.0), 0.0, 1.0)
    beam_gate = _build_floor_beam_mask((h, w), openings)
    if weather == "cloudy":
        candidate_zone = ((beam_gate > 0.14) & (floor_gate > 0.14)).astype(np.uint8) * 255
    elif weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        candidate_zone = _build_floor_cleanup_zone((h, w), openings)
    else:
        candidate_zone = ((beam_gate > 0.22) & (floor_gate > 0.18)).astype(np.uint8) * 255
    if protected_mask is not None:
        protected = protected_mask
        if protected.ndim == 3:
            protected = protected[:, :, 0]
        if protected.shape[:2] != (h, w):
            protected = cv2.resize(protected, (w, h), interpolation=cv2.INTER_NEAREST)
        protected = cv2.dilate(
            (protected > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
        candidate_zone = cv2.bitwise_and(candidate_zone, cv2.bitwise_not(protected))
    if np.count_nonzero(candidate_zone) == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = cv2.split(lab)
    sigma = max(10.0, min(h, w) * 0.03)
    l_coarse = cv2.GaussianBlur(l_channel, (0, 0), sigma)
    a_coarse = cv2.GaussianBlur(a_channel, (0, 0), sigma)
    b_coarse = cv2.GaussianBlur(b_channel, (0, 0), sigma)

    outer_ring = cv2.dilate(
        candidate_zone,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)),
        iterations=1,
    )
    background_ring = cv2.subtract(outer_ring, candidate_zone)
    if weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        background_selector = (floor_gate > 0.08).astype(np.uint8) * 255
    else:
        background_selector = ((floor_gate > 0.18) & (beam_gate < (0.10 if weather == "cloudy" else 0.18))).astype(np.uint8) * 255
    background_ring = cv2.bitwise_and(
        background_ring,
        background_selector,
    )
    if np.count_nonzero(background_ring) < max(200, int(h * w * 0.002)):
        return image

    if weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        background_l = float(np.quantile(l_coarse[background_ring > 0], 0.35))
    else:
        background_l = float(np.median(l_coarse[background_ring > 0]))
    background_a = float(np.median(a_coarse[background_ring > 0]))
    background_b = float(np.median(b_coarse[background_ring > 0]))
    brightness_delta = l_coarse - background_l
    warmth_delta = b_coarse - background_b
    if weather == "cloudy":
        bright_candidate = (brightness_delta > 6.0) | ((brightness_delta > 3.5) & (warmth_delta > 0.6))
        min_component_area = max(320, int(h * w * 0.0026))
        soft_gain = 1.42
    elif weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        raw_brightness_delta = l_channel - background_l
        bright_candidate = (
            (brightness_delta > 9.0)
            | ((brightness_delta > 5.5) & (warmth_delta > 1.0))
            | (raw_brightness_delta > 16.0)
            | (l_channel > 222.0)
        )
        min_component_area = max(380, int(h * w * 0.0022))
        soft_gain = 1.20
    else:
        bright_candidate = (brightness_delta > 9.0) | ((brightness_delta > 5.0) & (warmth_delta > 1.5))
        min_component_area = max(450, int(h * w * 0.0035))
        soft_gain = 1.18
    patch_mask = (bright_candidate & (candidate_zone > 0)).astype(np.uint8) * 255
    if np.count_nonzero(patch_mask) == 0:
        return image

    patch_mask = cv2.morphologyEx(
        patch_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1,
    )
    patch_mask = cv2.dilate(
        patch_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    )
    if weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        patch_mask = cv2.dilate(
            patch_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            iterations=1,
        )
    patch_mask = _filter_small_components(
        patch_mask,
        min_area=min_component_area,
    )
    if np.count_nonzero(patch_mask) == 0:
        return image

    soft_mask = cv2.GaussianBlur(patch_mask.astype(np.float32) / 255.0, (0, 0), 10.0)
    soft_mask = np.clip(soft_mask * soft_gain, 0.0, 1.0)

    target_l = cv2.inpaint(np.clip(l_coarse, 0, 255).astype(np.uint8), patch_mask, 13, cv2.INPAINT_NS).astype(np.float32)
    target_a = cv2.inpaint(np.clip(a_coarse, 0, 255).astype(np.uint8), patch_mask, 13, cv2.INPAINT_NS).astype(np.float32)
    target_b = cv2.inpaint(np.clip(b_coarse, 0, 255).astype(np.uint8), patch_mask, 13, cv2.INPAINT_NS).astype(np.float32)
    if weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        patch_binary = patch_mask > 0
        target_l[patch_binary] = np.minimum(target_l[patch_binary], background_l + 2.0)
        target_a[patch_binary] = target_a[patch_binary] * 0.55 + background_a * 0.45
        target_b[patch_binary] = np.minimum(target_b[patch_binary], background_b + 1.0)

    l_detail = l_channel - l_coarse
    a_detail = a_channel - a_coarse
    b_detail = b_channel - b_coarse

    if weather == "cloudy":
        new_l = target_l + 0.20 * l_detail
        new_a = target_a + 0.08 * a_detail
        new_b = target_b + 0.04 * b_detail
    elif weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        new_l = target_l + 0.34 * l_detail
        new_a = target_a + 0.22 * a_detail
        new_b = target_b + 0.18 * b_detail
    else:
        new_l = target_l + 0.68 * l_detail
        new_a = target_a + 0.20 * a_detail
        new_b = target_b + 0.20 * b_detail

    corrected_lab = lab.copy()
    corrected_lab[:, :, 0] = l_channel * (1.0 - soft_mask) + new_l * soft_mask
    corrected_lab[:, :, 1] = a_channel * (1.0 - soft_mask) + new_a * soft_mask
    corrected_lab[:, :, 2] = b_channel * (1.0 - soft_mask) + new_b * soft_mask
    corrected_lab = np.clip(corrected_lab, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)


def _suppress_warm_wall_artifacts(
    original_image: np.ndarray,
    relit_image: np.ndarray,
    openings: list[tuple[int, int, int, int]],
    weather: str,
) -> np.ndarray:
    if weather not in PROJECTED_GROUND_LIGHT_WEATHERS or not openings:
        return relit_image

    h, w = relit_image.shape[:2]
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    floor_gate = np.clip((yy - h * 0.60) / max(h * 0.40, 1.0), 0.0, 1.0)
    non_floor = 1.0 - floor_gate

    orig_lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    relit_lab = cv2.cvtColor(relit_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    warm_delta = relit_lab[:, :, 2] - orig_lab[:, :, 2]
    light_delta = relit_lab[:, :, 0] - orig_lab[:, :, 0]
    local_sigma = max(24.0, w * 0.035)
    warm_baseline = cv2.GaussianBlur(warm_delta, (0, 0), local_sigma)
    light_baseline = cv2.GaussianBlur(light_delta, (0, 0), local_sigma)
    local_warm_excess = warm_delta - warm_baseline
    local_light_excess = light_delta - light_baseline

    orig_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(orig_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(orig_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    gradient_ref = float(np.quantile(gradient, 0.95))
    if gradient_ref > 1e-6:
        gradient /= gradient_ref
    smooth_wall = np.clip((0.20 - gradient) / 0.20, 0.0, 1.0)

    preserve = np.zeros((h, w), dtype=np.uint8)
    for x, y, opening_w, opening_h in openings:
        rim = np.array(
            [
                [max(0, int(x)), max(0, int(y))],
                [min(w - 1, int(x + opening_w)), max(0, int(y))],
                [min(w - 1, int(x + opening_w)), min(h - 1, int(y + opening_h))],
                [max(0, int(x)), min(h - 1, int(y + opening_h))],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(preserve, rim, 255)
    preserve = cv2.GaussianBlur(preserve.astype(np.float32) / 255.0, (0, 0), max(4.0, w * 0.004))

    artifact_strength = (
        np.clip((local_warm_excess - 1.8) / 8.5, 0.0, 1.0)
        * np.clip((local_light_excess - 1.2) / 7.5, 0.0, 1.0)
        * smooth_wall
        * non_floor
        * (1.0 - preserve)
    )
    artifact_strength = cv2.GaussianBlur(artifact_strength.astype(np.float32), (0, 0), max(8.0, w * 0.008))
    artifact_strength = np.clip(artifact_strength * 0.92, 0.0, 0.88)

    if float(artifact_strength.max()) <= 1e-4:
        return relit_image

    relit_f = relit_image.astype(np.float32)
    orig_f = original_image.astype(np.float32)
    corrected = relit_f - (relit_f - orig_f) * artifact_strength[..., None] * 0.86
    return np.clip(corrected, 0.0, 255.0).astype(np.uint8)


def _apply_sky(result: np.ndarray, sky_layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3ch = alpha[..., None]
    blended = result.astype(np.float32) * (1.0 - alpha_3ch) + sky_layer.astype(np.float32) * alpha_3ch
    return np.clip(blended, 0, 255).astype(np.uint8)


def _masked_lab_mean(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if image is None or mask is None:
        return None
    binary = mask > 0
    if int(np.count_nonzero(binary)) < 200:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[binary].mean(axis=0)


def _apply_reference_interior_tone(
    image: np.ndarray,
    reference_image: np.ndarray,
    interior_gate: np.ndarray,
    room_spill: np.ndarray,
    sun_glow: np.ndarray,
    floor_gate: np.ndarray,
    beam_mask: np.ndarray,
    weather: str,
) -> np.ndarray:
    reference_opening_mask = _build_reference_opening_mask(reference_image)
    reference_interior_mask = cv2.bitwise_not(reference_opening_mask)
    target_interior_mask = ((interior_gate > 0.32).astype(np.uint8) * 255)

    reference_mean = _masked_lab_mean(reference_image, reference_interior_mask)
    target_mean = _masked_lab_mean(image, target_interior_mask)
    if reference_mean is None or target_mean is None:
        return image

    mean_delta = reference_mean - target_mean
    mean_delta[0] = float(np.clip(mean_delta[0], -24.0, 24.0))
    mean_delta[1] = float(np.clip(mean_delta[1], -12.0, 12.0))
    mean_delta[2] = float(np.clip(mean_delta[2], -18.0, 26.0))

    if weather == "sunrise":
        tone_mask = np.clip(room_spill * 0.16 + sun_glow * 0.10 + 0.08, 0.0, 0.30)
        channel_strength = np.array([0.22, 0.30, 0.48], dtype=np.float32)
    elif weather == "sunset":
        tone_mask = np.clip(room_spill * 0.14 + sun_glow * 0.08 + 0.06, 0.0, 0.24)
        channel_strength = np.array([0.18, 0.24, 0.40], dtype=np.float32)
    elif weather == "cloudy":
        tone_mask = np.clip(room_spill * 0.12 + sun_glow * 0.06 + 0.06, 0.0, 0.20)
        channel_strength = np.array([0.10, 0.10, 0.14], dtype=np.float32)
    elif weather == "clear":
        tone_mask = np.clip(room_spill * 0.06 + beam_mask * 0.04 + floor_gate * 0.03 + 0.02, 0.0, 0.10)
        channel_strength = np.array([0.08, 0.05, 0.04], dtype=np.float32)
    else:
        tone_mask = np.clip(room_spill * 0.12 + sun_glow * 0.06 + 0.06, 0.0, 0.20)
        channel_strength = np.array([0.10, 0.10, 0.14], dtype=np.float32)

    tone_mask *= interior_gate
    relit_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    for channel_index in range(3):
        relit_lab[:, :, channel_index] = np.clip(
            relit_lab[:, :, channel_index] + tone_mask * mean_delta[channel_index] * channel_strength[channel_index],
            0.0,
            255.0,
        )
    return cv2.cvtColor(relit_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _apply_reference_guided_lighting(
    base_image: np.ndarray,
    reference_image: np.ndarray | None,
    sky_mask: np.ndarray,
    openings: list[tuple[int, int, int, int]],
    weather: str,
) -> np.ndarray:
    if reference_image is None or reference_image.shape[:2] != base_image.shape[:2]:
        return base_image

    h, w = base_image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    image_area = float(max(h * w, 1))
    sky_binary = (sky_mask > 0).astype(np.float32)
    interior_gate = 1.0 - cv2.GaussianBlur(sky_binary, (0, 0), 2.2)
    room_spill = _normalized_blur(sky_binary, sigma=max(h, w) * 0.13)
    floor_gate = np.clip((yy - h * 0.50) / max(h * 0.50, 1.0), 0.0, 1.0)
    beam_mask = _build_floor_beam_mask((h, w), openings)
    single_large_opening = len(openings) == 1 and ((openings[0][2] * openings[0][3]) / image_area) > 0.06

    if openings:
        anchor_x, anchor_y, anchor_w, anchor_h = max(openings, key=lambda item: item[2] * item[3])
        profile_image = reference_image if reference_image is not None and reference_image.shape[:2] == base_image.shape[:2] else base_image
        profile_result = _estimate_opening_light_profile(profile_image, (anchor_x, anchor_y, anchor_w, anchor_h))
        sun_bias = float(profile_result[2]) if profile_result is not None else 0.22
        sun_x = anchor_x + anchor_w * np.clip(0.60 + sun_bias * 0.62, 0.16, 0.92)
        sun_y = anchor_y + anchor_h * 0.46
    else:
        sun_x = w * 0.76
        sun_y = h * 0.48
        sun_bias = 0.18

    sigma_x = max(w * 0.24, 1.0)
    sigma_y = max(h * 0.18, 1.0)
    sun_glow = np.exp(-(((xx - sun_x) ** 2) / (2.0 * sigma_x ** 2) + ((yy - sun_y) ** 2) / (2.0 * sigma_y ** 2)))
    sun_glow = cv2.GaussianBlur((sun_glow * interior_gate).astype(np.float32), (0, 0), 16.0)
    sweep_target_x = sun_x - np.sign(sun_bias if abs(sun_bias) > 0.05 else 1.0) * max(w * 0.24, anchor_w * 0.95 if openings else w * 0.24)
    sweep_target_y = h * (0.86 if weather == "sunrise" else 0.92)
    sun_sweep = _build_directional_sun_sweep(
        (h, w),
        sun_x,
        anchor_y + anchor_h * 0.92 if openings else sun_y,
        sweep_target_x,
        sweep_target_y,
        thickness=max(anchor_w * 0.42 if openings else w * 0.12, w * 0.08),
    )
    sun_sweep *= interior_gate
    sun_sweep_floor = sun_sweep * np.clip(0.14 + floor_gate * 0.92, 0.0, 1.0)
    if weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        sun_sweep_floor = np.zeros_like(sun_sweep_floor)

    sigma = max(15.0, min(h, w) * 0.028)
    base_blur = cv2.GaussianBlur(base_image.astype(np.float32), (0, 0), sigma)
    reference_blur = cv2.GaussianBlur(reference_image.astype(np.float32), (0, 0), sigma)
    reference_tone = cv2.GaussianBlur(reference_image.astype(np.float32), (0, 0), sigma * 1.8)
    illumination_delta = reference_blur - base_blur

    if weather == "sunrise":
        guide = np.clip(room_spill * 0.24 + sun_glow * 0.18 + 0.04, 0.0, 0.30)
    elif weather == "sunset":
        guide = np.clip(room_spill * 0.20 + sun_glow * 0.14 + 0.03, 0.0, 0.24)
    elif weather == "cloudy":
        guide = np.clip(room_spill * 0.18 + sun_glow * 0.10 + 0.06, 0.0, 0.24)
    elif weather == "clear":
        guide = np.clip(room_spill * 0.08 + beam_mask * 0.10 + floor_gate * 0.04 + 0.02, 0.0, 0.14)
    else:
        guide = np.clip(room_spill * 0.18 + sun_glow * 0.10 + 0.06, 0.0, 0.24)

    if single_large_opening and weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        guide *= (1.0 - floor_gate * 0.70)

    guide *= interior_gate
    relit = np.clip(base_image.astype(np.float32) + illumination_delta * guide[..., None], 0, 255).astype(np.uint8)

    if weather in {"sunrise", "sunset"}:
        if weather == "sunrise":
            warm_blend = np.clip(sun_glow * 0.08 + room_spill * 0.08, 0.0, 0.14) * interior_gate
        else:
            warm_blend = np.clip(sun_glow * 0.06 + room_spill * 0.05, 0.0, 0.10) * interior_gate
        if single_large_opening:
            warm_blend *= (1.0 - floor_gate * 0.75)
        relit_blur = cv2.GaussianBlur(relit.astype(np.float32), (0, 0), sigma * 1.4)
        warm_delta = reference_tone - relit_blur
        relit = np.clip(
            relit.astype(np.float32) + warm_delta * warm_blend[..., None],
            0,
            255,
        ).astype(np.uint8)

    if weather == "sunrise":
        ambient_reference = np.clip(room_spill * 0.08 + sun_glow * 0.05 + 0.03, 0.0, 0.12) * interior_gate
        if single_large_opening:
            ambient_reference *= (1.0 - floor_gate * 0.72)
        relit_blur = cv2.GaussianBlur(relit.astype(np.float32), (0, 0), sigma * 1.8)
        ambient_delta = reference_tone - relit_blur
        relit = np.clip(
            relit.astype(np.float32) + ambient_delta * ambient_reference[..., None],
            0,
            255,
        ).astype(np.uint8)

    relit = _apply_reference_interior_tone(
        image=relit,
        reference_image=reference_image,
        interior_gate=interior_gate,
        room_spill=room_spill,
        sun_glow=sun_glow,
        floor_gate=floor_gate,
        beam_mask=beam_mask,
        weather=weather,
    )

    return relit


def _apply_room_relighting(
    image: np.ndarray,
    sky_mask: np.ndarray,
    openings: list[tuple[int, int, int, int]],
    weather: str,
    reference_image: np.ndarray | None = None,
) -> np.ndarray:
    profile = _lighting_profile(weather)
    ground_light_enabled = weather in GROUND_LIGHT_WEATHERS
    original_floor_cleanup_enabled = weather in PROJECTED_GROUND_LIGHT_WEATHERS
    projected_ground_enabled = weather in PROJECTED_GROUND_LIGHT_WEATHERS

    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    image_area = float(max(h * w, 1))

    image_f = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sky_binary = (sky_mask > 0).astype(np.float32)
    sky_halo = _normalized_blur(sky_binary, sigma=max(h, w) * 0.05)
    room_spill = _normalized_blur(sky_binary, sigma=max(h, w) * 0.13)
    opening_protection = _build_opening_protection_mask((h, w), openings)
    protected_relight_mask = np.maximum(sky_binary, opening_protection)
    sky_interior_gate = 1.0 - cv2.GaussianBlur(sky_binary, (0, 0), 2.2)
    interior_gate = np.clip(sky_interior_gate * (1.0 - opening_protection * 0.96), 0.0, 1.0)
    if original_floor_cleanup_enabled:
        image = _suppress_floor_light_patches(
            image,
            openings,
            weather=weather,
            protected_mask=None,
        )
        image_f = image.astype(np.float32) / 255.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if openings:
        anchor_x, anchor_y, anchor_w, anchor_h = max(
            openings, key=lambda item: item[2] * item[3]
        )
        profile_image = reference_image if reference_image is not None and reference_image.shape[:2] == image.shape[:2] else image
        profile_result = _estimate_opening_light_profile(profile_image, (anchor_x, anchor_y, anchor_w, anchor_h))
        sun_bias = float(profile_result[2]) if profile_result is not None else 0.22
        sun_x = anchor_x + anchor_w * np.clip(0.64 + sun_bias * 0.60, 0.18, 0.94)
        sun_y = anchor_y + anchor_h * 0.54
    else:
        sun_x = w * 0.84
        sun_y = h * 0.60
        sun_bias = 0.18
    single_large_opening = len(openings) == 1 and ((openings[0][2] * openings[0][3]) / image_area) > 0.06

    sigma_x = max(w * 0.28, 1.0)
    sigma_y = max(h * 0.24, 1.0)
    sun_glow = np.exp(-(((xx - sun_x) ** 2) / (2.0 * sigma_x ** 2) + ((yy - sun_y) ** 2) / (2.0 * sigma_y ** 2)))
    sun_glow *= interior_gate
    sun_glow = cv2.GaussianBlur(sun_glow, (0, 0), 18.0)
    sweep_target_x = sun_x - np.sign(sun_bias if abs(sun_bias) > 0.05 else 1.0) * max(w * 0.24, anchor_w * 1.05 if openings else w * 0.24)
    sweep_target_y = h * (0.84 if weather == "sunrise" else 0.90)
    sun_sweep = _build_directional_sun_sweep(
        (h, w),
        sun_x,
        anchor_y + anchor_h * 0.92 if openings else sun_y,
        sweep_target_x,
        sweep_target_y,
        thickness=max(anchor_w * 0.44 if openings else w * 0.12, w * 0.08),
    )
    sun_sweep *= interior_gate
    floor_gate = np.clip((yy - h * 0.50) / max(h * 0.50, 1.0), 0.0, 1.0)
    sun_sweep_floor = sun_sweep * np.clip(0.14 + floor_gate * 0.92, 0.0, 1.0)
    if original_floor_cleanup_enabled:
        sun_sweep_floor = np.zeros_like(sun_sweep_floor)

    warmth_strength = float(profile.get("warmth", 1.0)) if ground_light_enabled else 0.0
    beam_strength = float(profile.get("beam", 1.0)) if ground_light_enabled else 0.0
    ambient_strength = 0.18 if ground_light_enabled else 0.0
    room_warmth = (ambient_strength + 0.24 * room_spill + 0.16 * sky_halo + 0.36 * sun_glow + 0.16 * sun_sweep_floor) * warmth_strength
    if original_floor_cleanup_enabled:
        room_warmth *= (1.0 - floor_gate * 0.82)
    if weather == "clear":
        room_warmth *= 0.22
    global_falloff = np.clip((0.90 - yy / max(h, 1) * 0.12) * float(profile.get("exposure", 1.0)), 0.68, 1.04)

    result = image_f.copy()
    contrast = float(profile.get("contrast", 1.0))
    if abs(contrast - 1.0) > 1e-6:
        result = np.clip((result - 0.5) * contrast + 0.5, 0.0, 1.0)
    result[:, :, 2] *= 1.0 + room_warmth * 0.28
    result[:, :, 1] *= 1.0 + room_warmth * 0.12
    result[:, :, 0] *= 1.0 - room_warmth * 0.10
    result *= global_falloff[..., None]
    tint = np.array(profile.get("tint", (1.0, 0.90, 0.78)), dtype=np.float32)
    tint_amount = np.clip(room_warmth * 0.12, 0.0, 0.22)
    if weather == "clear":
        tint_amount = np.clip((sky_halo * 0.03 + room_spill * 0.02) * beam_strength, 0.0, 0.05)
    result = result * (1.0 - tint_amount[..., None]) + tint * tint_amount[..., None]

    wall_gate = np.clip(1.0 - floor_gate * 0.78, 0.0, 1.0) * interior_gate
    ceiling_bias = np.clip(1.0 - yy / max(h * 0.78, 1.0), 0.0, 1.0)
    if weather == "clear":
        wall_wash = np.clip(
            room_spill * 0.16
            + sky_halo * 0.10
            + sun_glow * 0.08,
            0.0,
            0.22,
        ) * wall_gate
    else:
        wall_wash = np.clip(
            room_spill * 0.55 * beam_strength
            + sun_glow * 0.70 * beam_strength
            + 0.08 * beam_strength,
            0.0,
            1.0,
        ) * wall_gate
    if single_large_opening and weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        wall_wash *= 0.36
    wall_wash *= 0.55 + ceiling_bias * 0.45

    cream = np.zeros_like(result)
    if weather == "clear":
        cream[:, :, 2] = 0.96
        cream[:, :, 1] = 0.98
        cream[:, :, 0] = 1.00
    else:
        cream[:, :, 2] = 1.0
        cream[:, :, 1] = 0.90
        cream[:, :, 0] = 0.78

    highlights = np.clip((gray - 0.34) / 0.42, 0.0, 1.0)
    cream_blend = np.clip(highlights * wall_wash * (0.08 if weather == "clear" else 0.26), 0.0, 0.26 if weather != "clear" else 0.08)
    result = result * (1.0 - cream_blend[..., None]) + cream * cream_blend[..., None]

    warm_overlay = np.zeros_like(result)
    if weather == "clear":
        warm_overlay[:, :, 2] = 1.00
        warm_overlay[:, :, 1] = 0.97
        warm_overlay[:, :, 0] = 0.90
        direct_sun_mix = np.clip(sun_glow * 0.08 + sun_sweep_floor * 0.02, 0.0, 0.10)
    elif weather == "sunrise":
        warm_overlay[:, :, 2] = 1.00
        warm_overlay[:, :, 1] = 0.86
        warm_overlay[:, :, 0] = 0.64
        direct_sun_mix = np.clip(sun_glow * 0.24 + sun_sweep_floor * 0.06, 0.0, 0.26)
    elif weather == "sunset":
        warm_overlay[:, :, 2] = 1.00
        warm_overlay[:, :, 1] = 0.84
        warm_overlay[:, :, 0] = 0.58
        direct_sun_mix = np.clip(sun_glow * 0.28 + sun_sweep_floor * 0.07, 0.0, 0.29)
    elif weather == "cloudy":
        warm_overlay[:, :, 2] = 1.0
        warm_overlay[:, :, 1] = 0.72
        warm_overlay[:, :, 0] = 0.24
        direct_sun_mix = np.clip(sun_glow * 0.34 + sun_sweep_floor * 0.08, 0.0, 0.42)
    else:
        warm_overlay[:, :, 2] = 1.0
        warm_overlay[:, :, 1] = 0.72
        warm_overlay[:, :, 0] = 0.24
        direct_sun_mix = np.clip(sun_glow * 0.24 + sun_sweep_floor * 0.06, 0.0, 0.28)
    if single_large_opening and weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        direct_sun_mix *= (0.24 + floor_gate * 0.76)
    if original_floor_cleanup_enabled:
        direct_sun_mix *= (1.0 - floor_gate * 0.60)
    result = np.clip(result + (1.0 - result) * warm_overlay * direct_sun_mix[..., None], 0.0, 1.0)

    if original_floor_cleanup_enabled:
        floor_warmth = np.zeros((h, w), dtype=np.float32)
    elif weather == "clear":
        floor_warmth = np.clip(
            room_spill * floor_gate * 0.03
            + sun_glow * floor_gate * 0.02,
            0.0,
            0.08,
        )
    else:
        floor_warmth = np.clip(
            room_spill * floor_gate * 0.12 * beam_strength
            + sun_glow * floor_gate * 0.04 * beam_strength,
            0.0,
            0.30,
        )
    if weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        floor_warmth = np.clip(floor_warmth + sun_sweep_floor * floor_gate * 0.12 * beam_strength, 0.0, 0.28)
    result[:, :, 2] = np.clip(result[:, :, 2] + floor_warmth * 0.20, 0.0, 1.0)
    result[:, :, 1] = np.clip(result[:, :, 1] + floor_warmth * 0.10, 0.0, 1.0)
    result[:, :, 0] = np.clip(result[:, :, 0] - floor_warmth * 0.04, 0.0, 1.0)

    if projected_ground_enabled:
        projected_ground = _build_projected_ground_light(
            image,
            openings,
            weather,
            reference_image=reference_image,
        )
        projected_ground *= sky_interior_gate
        projected_strength = 0.55 if weather == "sunrise" else 0.65
        projected_warmth = 0.38 if weather == "sunrise" else 0.45
        projected_lift = np.clip(projected_ground * projected_strength, 0.0, 0.38)
        result = np.clip(result * (1.0 + projected_lift[..., None] * 0.48), 0.0, 1.0)
        result[:, :, 2] = np.clip(result[:, :, 2] + projected_ground * projected_warmth, 0.0, 1.0)
        result[:, :, 1] = np.clip(result[:, :, 1] + projected_ground * projected_warmth * 0.62, 0.0, 1.0)
        result[:, :, 0] = np.clip(result[:, :, 0] - projected_ground * projected_warmth * 0.14, 0.0, 1.0)

    result = np.power(np.clip(result, 0.0, 1.0), float(profile.get("gamma", 1.0)))

    graded = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    saturation = float(profile.get("saturation", 1.0))
    if abs(saturation - 1.0) > 1e-6:
        hsv = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(graded, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip((lab[:, :, 0] - 128.0) * 1.02 + 128.0 - 2.0, 0.0, 255.0)
    # LAB colour push: only for sunset/warm weathers
    if weather in {"sunrise", "sunset", "golden_hour", "dusk"}:
        lab[:, :, 1] = np.clip(lab[:, :, 1] + room_warmth * 2.0, 0.0, 255.0)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + room_warmth * 5.0, 0.0, 255.0)
    elif weather == "clear":
        lab[:, :, 1] = np.clip(lab[:, :, 1] - sky_halo * 0.4, 0.0, 255.0)
        lab[:, :, 2] = np.clip(lab[:, :, 2] - sky_halo * 1.8, 0.0, 255.0)
    else:
        # For sunny/overcast, keep it very subtle to maintain white walls
        lab[:, :, 1] = np.clip(lab[:, :, 1] + room_warmth * 0.5, 0.0, 255.0)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + room_warmth * 0.8, 0.0, 255.0)
    relit = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    relit = _apply_reference_guided_lighting(relit, reference_image, sky_mask, openings, weather)
    relit = _suppress_warm_wall_artifacts(image, relit, openings, weather)
    if original_floor_cleanup_enabled:
        return relit
    return _suppress_floor_light_patches(relit, openings, weather=weather)


def _build_regenerative_sky_mask(
    input_path: Path,
    mask_path: Path,
    image_shape: tuple[int, int],
    opening_polygons: Sequence[Sequence[Sequence[float | int]]] | None = None,
    opening_source_size: tuple[int, int] | None = None,
    force_regenerate: bool = False,
) -> np.ndarray:
    h, w = image_shape
    if force_regenerate or not mask_path.exists():
        generate_sky_mask(input_path, mask_path)

    sky_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if sky_mask is None:
        sky_mask = np.zeros((h, w), dtype=np.uint8)
    elif sky_mask.shape[:2] != (h, w):
        sky_mask = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    sky_mask = (sky_mask > 0).astype(np.uint8) * 255
    opening_mask = _build_mask_from_opening_polygons(
        image_shape=(h, w),
        opening_polygons=opening_polygons,
        opening_source_size=opening_source_size,
    )

    if np.count_nonzero(opening_mask) > 0 and np.count_nonzero(sky_mask) > 0:
        opening_mask = _filter_opening_mask_by_sky(opening_mask, sky_mask)

    regeneration_mask = cv2.bitwise_or(sky_mask, opening_mask)
    if np.count_nonzero(regeneration_mask) == 0:
        return regeneration_mask

    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    regeneration_mask = cv2.morphologyEx(regeneration_mask, cv2.MORPH_CLOSE, cleanup_kernel, iterations=1)
    regeneration_mask = cv2.dilate(regeneration_mask, cleanup_kernel, iterations=1)
    return (regeneration_mask > 0).astype(np.uint8) * 255


def _mask_bounding_box(mask: np.ndarray, padding: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    h, w = mask.shape[:2]
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(w, int(xs.max()) + 1 + padding)
    y2 = min(h, int(ys.max()) + 1 + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _build_seeded_sky_input(
    image: np.ndarray,
    mask: np.ndarray,
    weather: str,
    sky_assets_dir: str | Path = "static/sky_assets",
) -> np.ndarray:
    """Prefill the masked sky with a clean photographic sky before diffusion refinement."""
    h, w = image.shape[:2]
    try:
        sky_layer = _load_sky_asset(weather, w, h, Path(sky_assets_dir))
    except Exception as exc:
        logging.warning("[SkyRegenerate] Could not load seed sky asset; using original pixels: %s", exc)
        return image.copy()

    alpha = _build_sky_alpha(mask)
    if np.count_nonzero(alpha) == 0:
        return image.copy()
    return _apply_sky(image.copy(), sky_layer, alpha)


def _write_regeneration_crop(
    seed_image: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
) -> tuple[Path, Path, tuple[int, int, int, int]]:
    h, w = mask.shape[:2]
    padding = max(96, int(round(max(h, w) * 0.055)))
    bbox = _mask_bounding_box(mask, padding=padding)
    if bbox is None:
        raise RuntimeError("Cannot create regeneration crop from empty mask")

    x1, y1, x2, y2 = bbox
    crop_image = seed_image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    crop_image_path = output_path.with_name(f"{output_path.stem}_regen_crop_input.jpg")
    crop_mask_path = output_path.with_name(f"{output_path.stem}_regen_crop_mask.png")
    if not cv2.imwrite(str(crop_image_path), crop_image):
        raise RuntimeError(f"Could not write regeneration crop: {crop_image_path}")
    if not cv2.imwrite(str(crop_mask_path), crop_mask):
        raise RuntimeError(f"Could not write regeneration crop mask: {crop_mask_path}")
    return crop_image_path, crop_mask_path, bbox


def _composite_regenerated_crop(
    original_image: np.ndarray,
    generated_crop_rgb,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop_mask = mask[y1:y2, x1:x2]
    alpha = _build_sky_alpha(crop_mask)
    if np.count_nonzero(alpha) == 0:
        return original_image.copy()

    generated_crop = cv2.cvtColor(np.asarray(generated_crop_rgb.convert("RGB")), cv2.COLOR_RGB2BGR)
    target_h, target_w = crop_mask.shape[:2]
    if generated_crop.shape[:2] != (target_h, target_w):
        generated_crop = cv2.resize(generated_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    result = original_image.copy()
    roi = result[y1:y2, x1:x2].astype(np.float32)
    alpha_3ch = alpha[..., None]
    blended = roi * (1.0 - alpha_3ch) + generated_crop.astype(np.float32) * alpha_3ch
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def generate_regenerative_sky_variant(
    input_path: str | Path,
    output_path: str | Path,
    weather: str = "sunny",
    opening_polygons: Sequence[Sequence[Sequence[float | int]]] | None = None,
    opening_source_size: tuple[int, int] | None = None,
    mask_path: str | Path | None = None,
    fallback_to_composite: bool = True,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not load input image: {input_path}")

    h, w = image.shape[:2]
    regeneration_mask_path = Path(mask_path) if mask_path is not None else output_path.with_name(f"{output_path.stem}_regen_skymask.png")
    regeneration_mask_path.parent.mkdir(parents=True, exist_ok=True)
    regeneration_mask = _build_regenerative_sky_mask(
        input_path=input_path,
        mask_path=regeneration_mask_path,
        image_shape=(h, w),
        opening_polygons=opening_polygons,
        opening_source_size=opening_source_size,
        force_regenerate=mask_path is None,
    )
    if np.count_nonzero(regeneration_mask) == 0:
        logging.warning("[SkyRegenerate] Empty sky mask for %s; saving original image.", input_path.name)
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not write output image: {output_path}")
        return output_path

    if not cv2.imwrite(str(regeneration_mask_path), regeneration_mask):
        raise RuntimeError(f"Could not write regeneration mask: {regeneration_mask_path}")

    seed_image = _build_seeded_sky_input(image, regeneration_mask, weather)
    crop_image_path, crop_mask_path, crop_bbox = _write_regeneration_crop(
        seed_image=seed_image,
        mask=regeneration_mask,
        output_path=output_path,
    )

    prompt = _sky_regeneration_prompt(weather)
    strength = float(os.environ.get("PIXELDWELL_SKY_REGEN_STRENGTH", "0.52"))
    guidance_scale = float(os.environ.get("PIXELDWELL_SKY_REGEN_GUIDANCE", "8.0"))
    steps = int(os.environ.get("PIXELDWELL_SKY_REGEN_STEPS", "36"))
    crop_max_side = int(os.environ.get("PIXELDWELL_SKY_REGEN_CROP_MAX_SIDE", "1024"))

    try:
        from image_regeneration_model import get_regeneration_model

        logging.info(
            "[SkyRegenerate] Inpainting sky weather=%s crop=%s mask=%s strength=%.2f",
            weather,
            crop_image_path.name,
            crop_mask_path.name,
            strength,
        )
        model = get_regeneration_model()
        regenerated = model.regenerate_image(
            crop_image_path,
            mask_path=crop_mask_path,
            prompt=prompt,
            strength=strength,
            negative_prompt=SKY_REGENERATION_NEGATIVE_PROMPT,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            max_side=crop_max_side,
        )
        result = _composite_regenerated_crop(
            original_image=image,
            generated_crop_rgb=regenerated,
            mask=regeneration_mask,
            bbox=crop_bbox,
        )
        if not cv2.imwrite(str(output_path), result):
            raise RuntimeError(f"Could not write output image: {output_path}")
        logging.info("[SkyRegenerate] Saved generative sky replacement to %s", output_path)
        return output_path
    except Exception as exc:
        logging.error("[SkyRegenerate] Generative sky replacement failed: %s", exc)
        if not fallback_to_composite:
            raise
        logging.info("[SkyRegenerate] Falling back to composited full-scene sky replacement.")
        return generate_full_scene_variant(
            input_path=input_path,
            output_path=output_path,
            weather=weather,
            opening_polygons=opening_polygons,
            opening_source_size=opening_source_size,
        )
    finally:
        if os.environ.get("PIXELDWELL_KEEP_REGEN_DEBUG", "0").strip() != "1":
            for temp_path in (crop_image_path, crop_mask_path):
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass


def generate_full_scene_variant(
    input_path: str | Path,
    output_path: str | Path,
    weather: str = "sunny",
    sky_assets_dir: str | Path = "static/sky_assets",
    reference_root: str | Path = "reference",
    opening_polygons: Sequence[Sequence[Sequence[float | int]]] | None = None,
    opening_source_size: tuple[int, int] | None = None,
    processing_max_dim: int | None = None,
) -> Path:
    started_at = time.perf_counter()
    input_path = Path(input_path)
    output_path = Path(output_path)
    sky_assets_dir = Path(sky_assets_dir)
    reference_root = Path(reference_root)

    original_image = cv2.imread(str(input_path))
    if original_image is None:
        raise FileNotFoundError(f"Could not load input image: {input_path}")

    processing_limit = _resolve_processing_max_dim(processing_max_dim)
    image, resize_scale = _resize_for_processing(original_image, processing_limit)
    original_h, original_w = original_image.shape[:2]
    h, w = image.shape[:2]
    logging.info(
        "[FullSceneGenerator] Start %s weather=%s input=%dx%d working=%dx%d",
        input_path.name,
        weather,
        original_w,
        original_h,
        w,
        h,
    )

    processing_input_path = input_path
    temp_input_path: Path | None = None
    if resize_scale < 0.999:
        temp_input_path = output_path.with_name(f"{output_path.stem}_working_input.jpg")
        temp_input_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(temp_input_path), image):
            raise RuntimeError(f"Could not write temporary processing image: {temp_input_path}")
        processing_input_path = temp_input_path
        logging.info(
            "[FullSceneGenerator] Downscaled working image by %.3f for faster processing",
            resize_scale,
        )

    mask_path = output_path.with_name(f"{output_path.stem}_skymask.png")
    if not mask_path.exists():
        generate_sky_mask(processing_input_path, mask_path)
    sky_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if sky_mask is None:
        raise RuntimeError(f"Could not load generated mask: {mask_path}")
    if sky_mask.shape[:2] != (h, w):
        sky_mask = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    logging.info("[FullSceneGenerator] Sky mask ready coverage=%.2f%%", (np.count_nonzero(sky_mask) / float(max(h * w, 1))) * 100.0)

    reference_image, reference_sky_mask = _load_weather_reference(weather, w, h, reference_root)
    opening_mask = _build_mask_from_opening_polygons(
        image_shape=(h, w),
        opening_polygons=opening_polygons,
        opening_source_size=opening_source_size,
    )
    if np.count_nonzero(opening_mask) == 0 and reference_image is not None:
        opening_mask = _build_reference_opening_mask(image)
    if np.count_nonzero(opening_mask) > 0 and np.count_nonzero(sky_mask) > 0:
        opening_mask = _filter_opening_mask_by_sky(opening_mask, sky_mask)

    composite_mask = cv2.bitwise_or(opening_mask, sky_mask)
    if np.count_nonzero(composite_mask) == 0:
        composite_mask = sky_mask
    logging.info(
        "[FullSceneGenerator] Composite opening/sky mask coverage=%.2f%%",
        (np.count_nonzero(composite_mask) / float(max(h * w, 1))) * 100.0,
    )

    reference_layer = _build_reference_replacement_layer(
        image,
        reference_image,
        opening_mask,
        reference_sky_mask=reference_sky_mask,
        weather=weather,
    )
    sky_asset = reference_layer if reference_layer is not None else _load_sky_asset(weather, w, h, sky_assets_dir)
    replacement_mask = composite_mask
    if weather == "clear":
        clear_replacement_mask = _build_clear_replacement_mask(image, sky_mask, opening_mask)
        if np.count_nonzero(clear_replacement_mask) > 0:
            replacement_mask = clear_replacement_mask
    sky_alpha = _build_sky_alpha(replacement_mask)
    openings = _build_opening_stats(opening_mask if np.count_nonzero(opening_mask) > 0 else composite_mask)

    result = _apply_room_relighting(image, composite_mask, openings, weather, reference_image=reference_image)
    result = _apply_sky(result, sky_asset, sky_alpha)
    if weather == "clear":
        result = _restore_clear_non_sky_opening(image, result, opening_mask, replacement_mask)
    if resize_scale < 0.999 and result.shape[:2] != (original_h, original_w):
        result = cv2.resize(result, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        logging.info("[FullSceneGenerator] Upscaled final output back to %dx%d", original_w, original_h)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Could not write output image: {output_path}")

    if temp_input_path is not None:
        try:
            temp_input_path.unlink(missing_ok=True)
        except Exception:
            pass

    logging.info("[FullSceneGenerator] Saved %s in %.2fs", output_path, time.perf_counter() - started_at)
    return output_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if len(sys.argv) < 3:
        raise SystemExit("Usage: python full_scene_generator.py <input_path> <output_path> [weather]")

    weather_arg = sys.argv[3] if len(sys.argv) > 3 else "sunny"
    generate_full_scene_variant(sys.argv[1], sys.argv[2], weather=weather_arg)
