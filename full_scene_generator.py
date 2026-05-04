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
    "night": "night moon",
    "dramatic": "Dramatic Clouds",
    "dusk": "Dusk",
    "foggy": "Foggy",
    "golden_hour": "Golden Hour",
    "rainy": "Rainy",
    "partly_cloudy": "Soft Cloudy",
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
    "night": "night",
    "snowy": "overcast",
    "rainbow": "clear_day",
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
        "warmth": 0.30,
        "beam": 0.10,
        "exposure": 0.99,
        "tint": (0.92, 0.94, 0.96),
        "contrast": 0.98,
        "saturation": 0.90,
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
        "warmth": 0.62,
        "beam": 0.45,
        "exposure": 0.98,
        "tint": (1.00, 0.90, 0.70),
        "gamma": 0.96,
        "saturation": 1.05,
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
        "warmth": 0.0,
        "beam": 0.0,
        "exposure": 0.85,
        "tint": (1.00, 0.82, 0.76),
        "contrast": 1.0,
        "gamma": 1.0,
        "saturation": 0.65,
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

def _is_mask_rectangular(mask: np.ndarray) -> bool:
    """Detect if a mask is a near-perfect rectangle (indicative of a YOLO fallback box)."""
    binary = (mask > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return False
    x_b, y_b, w_b, h_b = cv2.boundingRect(binary)
    if w_b == 0 or h_b == 0:
        return False
    rect_area = w_b * h_b
    mask_pixels = np.count_nonzero(binary)
    return (mask_pixels / rect_area) > 0.98


GROUND_LIGHT_WEATHERS = {"sunny", "clear", "sunrise", "sunset", "golden_hour", "dusk", "rainbow"}
PROJECTED_GROUND_LIGHT_WEATHERS = {"sunrise", "sunset"}
DIFFUSE_CLOUD_WEATHERS = {"cloudy", "partly_cloudy"}
WARM_REFERENCE_WEATHERS = {"sunrise", "sunset", "golden_hour", "dusk"}

SKY_REGENERATION_PROMPTS = {
    "sunrise": "photorealistic sunrise sky, soft warm morning light, DSLR raw photo, 8k, natural detail",
    "sunset": "photorealistic sunset sky, warm golden light, cinematic layered clouds, DSLR raw photo, 8k",
    "clear": "photorealistic clear blue sky, sharp natural daylight, DSLR raw photo, 8k, deep blue",
    "cloudy": "photorealistic overcast sky, soft diffused light, natural cloud texture, DSLR raw photo, 8k",
    "sunny": "photorealistic sunny sky, crisp daylight, high-altitude perspective, DSLR raw photo, 8k",
    "partly_cloudy": "photorealistic partly cloudy sky, natural daylight, sharp cloud edges, DSLR raw photo, 8k",
    "dramatic": "photorealistic dramatic clouds, high contrast sky, cinematic atmosphere, DSLR raw photo, 8k",
    "night": "photorealistic night sky, deep navy horizon, realistic glowing moon with natural crater detail, subtle film grain, DSLR raw photo, 8k",
}

SKY_REGENERATION_NEGATIVE_PROMPT = (
    "painted, digital painting, CGI, cartoon, illustration, oversmoothed, plastic, glossy, "
    "blurry, over-smoothed appearance, fake, 3d render, low resolution, noisy, "
    "distorted architecture, warped window frames, changed furniture, changed walls, "
    "changed floor, text, watermark"
)


def _lighting_profile(weather: str) -> dict[str, float | tuple[float, float, float]]:
    return LIGHTING_BY_WEATHER.get(weather, LIGHTING_BY_WEATHER["partly_cloudy"])


def _sky_regeneration_prompt(weather: str) -> str:
    weather_prompt = SKY_REGENERATION_PROMPTS.get(weather, SKY_REGENERATION_PROMPTS["sunny"])
    return (
        f"{weather_prompt}, Make this image section look fully realistic and natural. Regenerate only that area while keeping the original shape, perspective, lighting, shadows, reflection direction, and color tone exactly consistent with the rest of the image. Add natural real-world texture, fine surface detail, subtle imperfections, and realistic material behavior. Output should look like a real DSLR photo."
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
        if _is_standalone_sky_reference(reference_image, weather):
            return _standalone_reference_sky_mask(reference_image, weather)
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
    night_sky = (
        (hue >= 82.0)
        & (hue <= 136.0)
        & (saturation > 0.055)
        & (value > 0.055)
        & (value < 0.62)
    )
    moon_haze = (lightness > 0.30) & (saturation < 0.30) & (value > 0.18)

    # Warm sky detection for golden hour / sunset / sunrise type references
    warm_sky = (
        (hue >= 0.0) & (hue <= 42.0)  # orange-red-yellow hues
        & (saturation > 0.06)
        & (value > 0.28)
    )
    golden_sky = (
        (lightness > 0.32)
        & (value > 0.26)
        & (
            ((hue >= 0.0) & (hue <= 48.0))   # warm hues
            | ((hue >= 150.0) & (hue <= 180.0))  # wrap-around red
        )
    )

    if weather == "night":
        sky_like = night_sky | moon_haze | blue_sky
        top_ratio = 0.66
    elif weather in DIFFUSE_CLOUD_WEATHERS:
        sky_like = cloudy_sky | pale_sky | blue_sky
        top_ratio = 0.62
    elif weather == "clear":
        sky_like = blue_sky | pale_sky
        top_ratio = 0.58
    elif weather in {"golden_hour", "sunset", "sunrise", "dusk"}:
        sky_like = warm_sky | golden_sky | pale_sky | blue_sky
        top_ratio = 0.70
    elif weather == "dramatic":
        sky_like = blue_sky | pale_sky | cloudy_sky | warm_sky
        top_ratio = 0.68
    else:
        sky_like = pale_sky | blue_sky | warm_sky
        top_ratio = 0.58

    upper_band = np.zeros((h, w), dtype=np.uint8)
    upper_band[: max(1, int(h * (0.88 if weather == "night" else 0.82))), :] = 255
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
    if weather in DIFFUSE_CLOUD_WEATHERS and np.count_nonzero(candidate) > 0:
        row_coverage = candidate.mean(axis=1) / 255.0
        active_rows = np.where(row_coverage > 0.08)[0]
        if active_rows.size > 0:
            max_bottom = min(int(active_rows.max()), max(1, int(h * 0.58)))
            candidate[max_bottom + 1 :, :] = 0
            candidate = cv2.GaussianBlur(candidate.astype(np.float32), (0, 0), 1.2)
            candidate = ((candidate > 18.0).astype(np.uint8) * 255)
    return candidate


def _standalone_sky_color_candidate(reference_image: np.ndarray, weather: str) -> np.ndarray:
    hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    lightness = lab[:, :, 0] / 255.0

    pale_sky = (lightness > 0.34) & (saturation < 0.42) & (value > 0.24)
    blue_sky = (hue >= 76.0) & (hue <= 134.0) & (saturation > 0.045) & (value > 0.22)
    warm_sky = (
        (((hue >= 0.0) & (hue <= 48.0)) | ((hue >= 150.0) & (hue <= 180.0)))
        & (saturation > 0.045)
        & (value > 0.22)
    )
    if weather in WARM_REFERENCE_WEATHERS:
        candidate = pale_sky | blue_sky | warm_sky
    else:
        candidate = pale_sky | blue_sky

    mask = candidate.astype(np.uint8) * 255
    if np.count_nonzero(mask) == 0:
        return mask
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return mask


def _standalone_reference_sky_mask(reference_image: np.ndarray, weather: str) -> np.ndarray:
    mask = _standalone_sky_color_candidate(reference_image, weather)
    h, w = reference_image.shape[:2]
    area = float(max(h * w, 1))
    if np.count_nonzero(mask) < max(64, int(area * 0.25)):
        return np.full((h, w), 255, dtype=np.uint8)
    return mask


def _is_standalone_sky_reference(reference_image: np.ndarray, weather: str) -> bool:
    if weather not in WARM_REFERENCE_WEATHERS:
        return False

    h, w = reference_image.shape[:2]
    if h <= 0 or w <= 0 or (w / float(h)) < 1.18:
        return False

    opening_mask = _build_reference_opening_mask(reference_image)
    opening_coverage = np.count_nonzero(opening_mask) / float(max(h * w, 1))
    if opening_coverage > 0.018:
        return False

    sky_candidate = _standalone_sky_color_candidate(reference_image, weather)
    sky_coverage = np.count_nonzero(sky_candidate) / float(max(h * w, 1))
    return sky_coverage > 0.35


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

    if weather == "night":
        if reference_image.shape[:2] != (height, width):
            reference_image = cv2.resize(reference_image, (width, height), interpolation=cv2.INTER_CUBIC)
        return reference_image, None

    if _is_standalone_sky_reference(reference_image, weather):
        return reference_image, _standalone_reference_sky_mask(reference_image, weather)

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
    reference_area = float(max(reference_image.shape[0] * reference_image.shape[1], 1))
    min_reference_sky_pixels = max(64, int(reference_area * 0.008))
    reference_pixel_count = int(np.count_nonzero(reference_mask)) if reference_mask is not None else 0
    fallback_reference_mask = _fallback_reference_sky_mask(reference_image, weather)
    fallback_pixel_count = int(np.count_nonzero(fallback_reference_mask))
    if (
        reference_mask is None
        or reference_pixel_count < min_reference_sky_pixels
        or (reference_pixel_count < reference_area * 0.012 and fallback_pixel_count > reference_pixel_count * 2)
    ):
        reference_mask = fallback_reference_mask
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
    if weather == "night":
        return _build_night_sky_asset(width, height)

    asset_name = SKY_ASSET_MAP.get(weather, "golden_hour")
    asset_path = sky_assets_dir / f"{asset_name}.jpg"
    if not asset_path.exists():
        asset_path = sky_assets_dir / "golden_hour.jpg"

    asset = cv2.imread(str(asset_path))
    if asset is None:
        raise FileNotFoundError(f"Could not load sky asset: {asset_path}")

    return _fit_cover(asset, width, height)


def _build_night_sky_asset(width: int, height: int) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    top = np.array([28.0, 22.0, 16.0], dtype=np.float32)
    bottom = np.array([54.0, 42.0, 28.0], dtype=np.float32)
    layer = top.reshape(1, 1, 3) * (1.0 - yy[..., None]) + bottom.reshape(1, 1, 3) * yy[..., None]
    layer = layer + np.zeros((height, width, 1), dtype=np.float32)

    horizon = np.exp(-((yy - 0.82) ** 2) / (2.0 * 0.16 * 0.16))
    layer += horizon[..., None] * np.array([12.0, 9.0, 5.0], dtype=np.float32).reshape(1, 1, 3)

    # Deterministic, very faint stars so the result is not noisy or random.
    star_count = max(12, min(90, int(width * height * 0.000012)))
    seed = (width * 73856093) ^ (height * 19349663)
    rng = np.random.default_rng(seed)
    for _ in range(star_count):
        x = int(rng.integers(0, max(width, 1)))
        y = int(rng.integers(0, max(int(height * 0.58), 1)))
        strength = float(rng.uniform(8.0, 24.0))
        radius = float(rng.uniform(0.45, 0.95))
        x1 = max(0, x - 2)
        x2 = min(width, x + 3)
        y1 = max(0, y - 2)
        y2 = min(height, y + 3)
        if x2 <= x1 or y2 <= y1:
            continue
        local_y, local_x = np.mgrid[y1:y2, x1:x2].astype(np.float32)
        alpha = np.exp(-((local_x - x) ** 2 + (local_y - y) ** 2) / (2.0 * radius * radius)) * strength
        layer[y1:y2, x1:x2] += alpha[..., None]

    return np.clip(layer, 0, 255).astype(np.uint8)


def _add_night_moon(
    sky_layer: np.ndarray,
    replacement_mask: np.ndarray,
) -> np.ndarray:
    replacement = (replacement_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(replacement) == 0:
        return sky_layer

    components = _extract_mask_components(replacement, min_area_ratio=0.00025)
    usable = [
        component
        for component in components
        if int(component["w"]) >= 12 and int(component["h"]) >= 12
    ]
    if not usable:
        return sky_layer

    max_area = max(float(component["area"]) for component in usable)
    primary_components = [
        component
        for component in usable
        if float(component["area"]) >= max_area * 0.60
    ]
    component = max(primary_components, key=lambda item: int(item["x"]) + int(item["w"]) * 0.5)
    x = int(component["x"])
    y = int(component["y"])
    box_w = int(component["w"])
    box_h = int(component["h"])
    if box_w <= 8 or box_h <= 8:
        return sky_layer

    cx = int(round(x + box_w * 0.64))
    cy = int(round(y + box_h * 0.31))
    component_mask = np.asarray(component["mask"], dtype=np.uint8)
    if not (0 <= cx < replacement.shape[1] and 0 <= cy < replacement.shape[0]) or component_mask[cy, cx] == 0:
        nearest = _nearest_mask_point(component_mask, cx, cy)
        if nearest is None:
            return sky_layer
        cx, cy = nearest

    h, w = replacement.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    distance_sq = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2
    moon_radius = max(4.0, min(box_w, box_h) * 0.034)
    core = np.exp(-distance_sq / (2.0 * moon_radius * moon_radius))
    disk = np.clip((core - 0.36) / 0.64, 0.0, 1.0)
    disk = cv2.GaussianBlur(disk.astype(np.float32), (0, 0), 1.1)

    halo_sigma = max(16.0, min(box_w, box_h) * 0.13)
    halo = np.exp(-distance_sq / (2.0 * halo_sigma * halo_sigma)) * 0.08
    mask_alpha = cv2.GaussianBlur(component_mask.astype(np.float32) / 255.0, (0, 0), 1.3)
    disk_alpha = np.clip(disk * mask_alpha * 0.64, 0.0, 0.64)
    halo_alpha = np.clip(halo * mask_alpha, 0.0, 0.08)

    moon_color = np.array([186.0, 198.0, 210.0], dtype=np.float32)
    halo_color = np.array([58.0, 65.0, 82.0], dtype=np.float32)
    layer_f = sky_layer.astype(np.float32)
    layer_f = layer_f * (1.0 - halo_alpha[..., None]) + halo_color.reshape(1, 1, 3) * halo_alpha[..., None]
    layer_f = layer_f * (1.0 - disk_alpha[..., None]) + moon_color.reshape(1, 1, 3) * disk_alpha[..., None]
    return np.clip(layer_f, 0, 255).astype(np.uint8)


def _build_smooth_clear_sky_texture(sky_crop: np.ndarray, blue_mask: np.ndarray) -> np.ndarray:
    valid_blue = blue_mask > 0
    if sky_crop.size == 0 or int(np.count_nonzero(valid_blue)) < 50:
        sigma = max(2.0, min(sky_crop.shape[:2]) * 0.045)
        return cv2.GaussianBlur(sky_crop, (0, 0), sigma)

    hsv = cv2.cvtColor(sky_crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    samples = hsv[valid_blue]
    hue = float(np.median(samples[:, 0]))
    saturation = float(np.quantile(samples[:, 1], 0.60))
    value = float(np.quantile(samples[:, 2], 0.58))

    h, w = sky_crop.shape[:2]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]

    sat_top = float(np.clip(saturation * 1.05 + 4.0, 50.0, 142.0))
    sat_bottom = float(np.clip(saturation * 0.78 + 2.0, 34.0, 112.0))
    val_top = float(np.clip(value * 0.98, 138.0, 238.0))
    val_bottom = float(np.clip(value * 1.06 + 4.0, 158.0, 250.0))

    smooth_hsv = np.zeros((h, w, 3), dtype=np.float32)
    smooth_hsv[:, :, 0] = hue
    smooth_hsv[:, :, 1] = sat_top * (1.0 - yy) + sat_bottom * yy
    smooth_hsv[:, :, 2] = val_top * (1.0 - yy) + val_bottom * yy + (1.0 - np.abs(xx)) * 1.8
    smooth_hsv[:, :, 2] = np.clip(smooth_hsv[:, :, 2], 0.0, 255.0)
    return cv2.cvtColor(smooth_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _detect_clear_reference_sun(
    reference_image: np.ndarray | None,
    reference_sky_mask: np.ndarray | None,
) -> tuple[float, float, np.ndarray] | None:
    if reference_image is None or reference_sky_mask is None or np.count_nonzero(reference_sky_mask) == 0:
        return None

    mask = (reference_sky_mask > 0).astype(np.uint8) * 255
    h, w = reference_image.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if x2 <= x1 + 12 or y2 <= y1 + 12:
        return None

    support = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)),
        iterations=1,
    )
    hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    lightness = lab[:, :, 0] / 255.0
    yellow = lab[:, :, 2]

    candidate = (
        (support > 0)
        & (lightness > 0.78)
        & (value > 0.72)
        & (saturation < 0.38)
        & (yellow > 130.0)
    ).astype(np.uint8) * 255
    candidate[:y1, :] = 0
    candidate[y2:, :] = 0
    candidate[:, :x1] = 0
    candidate[:, x2:] = 0

    if np.count_nonzero(candidate) == 0:
        return None

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    best_label = 0
    best_score = -1.0
    sky_area = float(max(np.count_nonzero(mask), 1))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(8, int(sky_area * 0.00015)):
            continue
        cx, cy = centroids[label]
        rel_x = (float(cx) - x1) / float(max(x2 - x1, 1))
        rel_y = (float(cy) - y1) / float(max(y2 - y1, 1))
        if rel_y < 0.16 or rel_y > 0.92:
            continue
        component = labels == label
        score = area * (1.0 + float(lightness[component].mean())) * (1.0 + max(0.0, rel_x - 0.42))
        if score > best_score:
            best_score = score
            best_label = label

    if best_label == 0:
        return None

    component = labels == best_label
    weights = np.maximum(lightness[component] - 0.72, 0.02)
    yy, xx = np.where(component)
    cx = float(np.average(xx, weights=weights))
    cy = float(np.average(yy, weights=weights))
    rel_x = float(np.clip((cx - x1) / float(max(x2 - x1, 1)), 0.12, 0.90))
    rel_y = float(np.clip((cy - y1) / float(max(y2 - y1, 1)), 0.16, 0.84))
    sun_color = np.quantile(reference_image[component].reshape(-1, 3), 0.86, axis=0).astype(np.float32)
    sun_color = np.clip(sun_color * 0.72 + np.array([238.0, 244.0, 255.0], dtype=np.float32) * 0.28, 0.0, 255.0)
    return rel_x, rel_y, sun_color


def _nearest_mask_point(mask: np.ndarray, x: int, y: int) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    distances = (xs.astype(np.float32) - float(x)) ** 2 + (ys.astype(np.float32) - float(y)) ** 2
    index = int(np.argmin(distances))
    return int(xs[index]), int(ys[index])


def _add_clear_reference_sun_glow(
    sky_layer: np.ndarray,
    reference_image: np.ndarray | None,
    reference_sky_mask: np.ndarray | None,
    replacement_mask: np.ndarray,
) -> np.ndarray:
    profile = _detect_clear_reference_sun(reference_image, reference_sky_mask)
    replacement = (replacement_mask > 0).astype(np.uint8) * 255
    if profile is None or np.count_nonzero(replacement) == 0:
        return sky_layer

    ys, xs = np.where(replacement > 0)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 8 or box_h <= 8:
        return sky_layer

    rel_x, rel_y, sun_color = profile
    cx = int(round(x1 + rel_x * box_w))
    cy = int(round(y1 + rel_y * box_h))
    if not (0 <= cx < replacement.shape[1] and 0 <= cy < replacement.shape[0]) or replacement[cy, cx] == 0:
        nearest = _nearest_mask_point(replacement, cx, cy)
        if nearest is None:
            return sky_layer
        cx, cy = nearest

    h, w = replacement.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    distance_sq = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2
    small_side = max(1.0, float(min(box_w, box_h)))
    core_sigma = max(5.0, small_side * 0.024)
    halo_sigma = max(18.0, small_side * 0.115)
    core = np.exp(-distance_sq / (2.0 * core_sigma * core_sigma)) * 0.18
    halo = np.exp(-distance_sq / (2.0 * halo_sigma * halo_sigma)) * 0.08
    mask_alpha = cv2.GaussianBlur(replacement.astype(np.float32) / 255.0, (0, 0), 1.2)
    alpha = np.clip((core + halo) * mask_alpha, 0.0, 0.22)
    if float(alpha.max()) <= 1e-4:
        return sky_layer

    layer_f = sky_layer.astype(np.float32)
    sun = np.full_like(layer_f, sun_color.reshape(1, 1, 3), dtype=np.float32)
    blended = layer_f * (1.0 - alpha[..., None]) + sun * alpha[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def _enrich_clear_blue_sky_layer(sky_layer: np.ndarray, replacement_mask: np.ndarray) -> np.ndarray:
    replacement = (replacement_mask > 0).astype(np.float32) / 255.0
    if np.count_nonzero(replacement) == 0:
        return sky_layer

    hsv = cv2.cvtColor(sky_layer, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    blue_focus = (
        (replacement > 0)
        & (hue >= 76.0)
        & (hue <= 132.0)
        & ~((value > 232.0) & (saturation < 70.0))
    ).astype(np.float32)
    blue_focus = cv2.GaussianBlur(blue_focus, (0, 0), 0.8)
    if float(blue_focus.max()) <= 1e-4:
        return sky_layer

    hsv[:, :, 1] = np.clip(saturation + blue_focus * (5.0 + saturation * 0.035), 0.0, 255.0)
    hsv[:, :, 2] = np.clip(value + blue_focus * 1.5, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _extract_clear_blue_reference_texture(reference_image: np.ndarray) -> np.ndarray | None:
    hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0

    raw_blue_mask = (
        (hue >= 76.0)
        & (hue <= 132.0)
        & (saturation > 0.10)
        & (value > 0.35)
    ).astype(np.uint8) * 255
    blue_mask = raw_blue_mask.copy()
    if np.count_nonzero(blue_mask) < max(300, int(reference_image.shape[0] * reference_image.shape[1] * 0.002)):
        return None

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    components = _extract_mask_components(blue_mask, min_area_ratio=0.0008)
    if not components:
        return None

    h, w = reference_image.shape[:2]
    component = components[0]
    x = int(component["x"])
    y = int(component["y"])
    component_w = int(component["w"])
    component_h = int(component["h"])
    pad_x = int(round(component_w * 0.06))
    pad_y = int(round(component_h * 0.06))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + component_w + pad_x)
    y2 = min(h, y + component_h + pad_y)
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return None

    sky_crop = reference_image[y1:y2, x1:x2].copy()
    component_mask = np.asarray(component["mask"], dtype=np.uint8)[y1:y2, x1:x2]
    crop_mask = cv2.bitwise_and(raw_blue_mask[y1:y2, x1:x2], component_mask)
    if sky_crop.size == 0 or np.count_nonzero(crop_mask) < 80:
        return None

    crop_mask = cv2.morphologyEx(
        crop_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    non_sky_mask = ((crop_mask == 0).astype(np.uint8) * 255)
    if np.count_nonzero(non_sky_mask) > 0:
        radius = max(5, int(round(min(sky_crop.shape[:2]) * 0.018)))
        sky_crop = cv2.inpaint(sky_crop, non_sky_mask, radius, cv2.INPAINT_TELEA)
    return _build_smooth_clear_sky_texture(sky_crop, crop_mask)


def _enhance_cloudy_reference_texture(texture: np.ndarray) -> np.ndarray:
    if texture.size == 0:
        return texture

    lab = cv2.cvtColor(texture, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[:, :, 0]
    sigma = max(2.0, min(texture.shape[:2]) * 0.035)
    base = cv2.GaussianBlur(lightness, (0, 0), sigma)
    cloud_detail = lightness - base
    lab[:, :, 0] = np.clip(lightness + cloud_detail * 0.45 - 1.8, 0.0, 255.0)
    lab[:, :, 1] = np.clip(lab[:, :, 1] * 0.98 + 128.0 * 0.02, 0.0, 255.0)
    lab[:, :, 2] = np.clip(lab[:, :, 2] * 0.96 + 128.0 * 0.04, 0.0, 255.0)

    enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(enhanced, 5, 12, 6)


def _extract_cloudy_reference_texture(reference_image: np.ndarray) -> np.ndarray | None:
    h, w = reference_image.shape[:2]
    if h < 80 or w < 120:
        return None

    hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    lightness = lab[:, :, 0] / 255.0

    blue_sky = (hue >= 76.0) & (hue <= 132.0) & (saturation > 0.035) & (value > 0.34)
    pale_cloud = (lightness > 0.55) & (saturation < 0.30) & (value > 0.42)
    gray_cloud = (lightness > 0.43) & (saturation < 0.34) & (value > 0.30)
    sky_like = blue_sky | pale_cloud | gray_cloud

    hsv_u8 = hsv.astype(np.uint8)
    green_content = cv2.inRange(hsv_u8, np.array([25, 22, 18]), np.array([88, 255, 245])) > 0
    brown_content = cv2.inRange(hsv_u8, np.array([8, 26, 18]), np.array([42, 255, 230])) > 0
    dark_structure = (value < 0.28) & (saturation > 0.035)
    reject_content = green_content | brown_content | dark_structure

    soft_base = cv2.GaussianBlur(gray, (0, 0), 7.0)
    local_detail = np.abs(gray - soft_base)
    edge_detail = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)
    textured_cloud = (local_detail > 2.2) | (edge_detail > 2.8) | blue_sky

    candidate = (sky_like & textured_cloud & ~reject_content).astype(np.uint8) * 255
    candidate[: int(h * 0.20), :] = 0
    candidate[int(h * 0.60) :, :] = 0
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    candidate = cv2.dilate(
        candidate,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 9)),
        iterations=1,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 15)),
        iterations=2,
    )
    candidate = cv2.bitwise_and(candidate, (sky_like & ~reject_content).astype(np.uint8) * 255)

    components = _extract_mask_components(candidate, min_area_ratio=0.0002)
    best_component: dict[str, np.ndarray | int | float] | None = None
    best_score = -1e9
    image_area = float(max(h * w, 1))
    for component in components:
        x = int(component["x"])
        y = int(component["y"])
        component_w = int(component["w"])
        component_h = int(component["h"])
        area = float(component["area"])
        if component_w < max(44, int(w * 0.02)) or component_h < max(26, int(h * 0.018)):
            continue

        x2 = x + component_w
        y2 = y + component_h
        center_y = ((y + y2) * 0.5) / float(max(h, 1))
        if center_y < 0.20 or center_y > 0.58:
            continue

        below_x1 = max(0, x - int(component_w * 0.04))
        below_x2 = min(w, x2 + int(component_w * 0.04))
        below_y1 = min(h, y2 + 2)
        below_y2 = min(h, y2 + int(h * 0.20))
        below_green = below_brown = below_dark = 0.0
        if below_y2 > below_y1 and below_x2 > below_x1:
            below_slice = (slice(below_y1, below_y2), slice(below_x1, below_x2))
            below_green = float(green_content[below_slice].mean())
            below_brown = float(brown_content[below_slice].mean())
            below_dark = float(dark_structure[below_slice].mean())

        component_mask = np.asarray(component["mask"], dtype=np.uint8) > 0
        gray_std = float(gray[component_mask].std()) if np.count_nonzero(component_mask) else 0.0
        edge_mean = float(edge_detail[component_mask].mean()) if np.count_nonzero(component_mask) else 0.0
        score = (
            min(area / image_area * 18.0, 3.0)
            + below_green * 6.0
            + below_brown * 1.6
            + below_dark * 1.4
            + min(gray_std / 18.0, 1.8)
            + min(edge_mean / 7.0, 1.5)
            - abs(center_y - 0.39) * 2.0
        )
        if component_w > w * 0.72 and below_green < 0.08:
            score -= 3.5
        if y < h * 0.24 and below_green < 0.06:
            score -= 1.2

        if score > best_score:
            best_score = score
            best_component = component

    if best_component is None:
        return None

    selected = np.asarray(best_component["mask"], dtype=np.uint8)
    selected = cv2.morphologyEx(
        selected,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 9)),
        iterations=1,
    )
    ys, xs = np.where(selected > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    raw_x1 = int(xs.min())
    raw_x2 = int(xs.max()) + 1
    raw_y1 = int(ys.min())
    raw_y2 = int(ys.max()) + 1
    crop_w = raw_x2 - raw_x1
    crop_h = raw_y2 - raw_y1
    if crop_w < 60 or crop_h < 45:
        return None

    x1 = max(0, raw_x1 + int(crop_w * 0.045))
    x2 = min(w, raw_x2 - int(crop_w * 0.035))
    y1 = max(0, raw_y1 + int(crop_h * 0.095))
    y2 = min(h, raw_y2)

    exterior_content = green_content | brown_content | dark_structure
    if x2 > x1:
        row_content = exterior_content[:, x1:x2].mean(axis=1)
        scan_start = max(y1 + 10, raw_y1 + int(crop_h * 0.34))
        scan_end = min(y2, int(h * 0.54))
        band_h = max(4, int(h * 0.006))
        for row in range(scan_start, max(scan_start, scan_end - band_h + 1)):
            if float(row_content[row:row + band_h].mean()) > 0.055:
                y2 = min(y2, max(y1 + 20, row - int(crop_h * 0.035)))
                break
    y2 = min(y2, int(h * 0.52))
    if x2 <= x1 + 20 or y2 <= y1 + 20:
        return None

    sky_crop = reference_image[y1:y2, x1:x2].copy()
    crop_mask = (sky_like[y1:y2, x1:x2] & ~reject_content[y1:y2, x1:x2]).astype(np.uint8) * 255
    if sky_crop.size == 0 or np.count_nonzero(crop_mask) < max(80, int(sky_crop.shape[0] * sky_crop.shape[1] * 0.20)):
        return None

    crop_mask = cv2.morphologyEx(
        crop_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    sky_crop, crop_mask = _select_clean_reference_sky_span(sky_crop, crop_mask)
    if sky_crop.size == 0 or np.count_nonzero(crop_mask) < 40:
        return None

    if sky_crop.shape[0] >= 70:
        top_trim = int(round(sky_crop.shape[0] * 0.27))
        if sky_crop.shape[0] - top_trim >= max(36, int(sky_crop.shape[0] * 0.48)):
            sky_crop = sky_crop[top_trim:, :]
            crop_mask = crop_mask[top_trim:, :]

    non_sky_mask = ((crop_mask == 0).astype(np.uint8) * 255)
    if np.count_nonzero(non_sky_mask) > 0:
        radius = max(3, int(round(min(sky_crop.shape[:2]) * 0.012)))
        sky_crop = cv2.inpaint(sky_crop, non_sky_mask, radius, cv2.INPAINT_TELEA)

    return _enhance_cloudy_reference_texture(sky_crop)


def _extract_reference_sky_texture(
    reference_image: np.ndarray | None,
    reference_sky_mask: np.ndarray | None,
    weather: str,
) -> np.ndarray | None:
    if reference_image is None:
        return None

    if weather == "clear":
        clear_texture = _extract_clear_blue_reference_texture(reference_image)
        if clear_texture is not None:
            return clear_texture

    if weather == "cloudy":
        cloudy_texture = _extract_cloudy_reference_texture(reference_image)
        if cloudy_texture is not None:
            return cloudy_texture

    if reference_sky_mask is None or np.count_nonzero(reference_sky_mask) == 0:
        reference_sky_mask = _fallback_reference_sky_mask(reference_image, weather)

    if reference_sky_mask is None or np.count_nonzero(reference_sky_mask) == 0:
        return None

    mask = (reference_sky_mask > 0).astype(np.uint8) * 255
    if mask.shape[:2] != reference_image.shape[:2]:
        mask = cv2.resize(mask, (reference_image.shape[1], reference_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    selected, _ = _select_reference_sky_mask(mask, weather)
    if selected is None:
        return None

    selected = cv2.morphologyEx(
        selected,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )

    ys, xs = np.where(selected > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    h, w = selected.shape[:2]
    raw_x1 = int(xs.min())
    raw_x2 = int(xs.max()) + 1
    raw_y1 = int(ys.min())
    raw_y2 = int(ys.max()) + 1
    crop_w = raw_x2 - raw_x1
    crop_h = raw_y2 - raw_y1
    if weather in WARM_REFERENCE_WEATHERS:
        side_trim = int(round(crop_w * 0.070))
        top_trim = int(round(crop_h * 0.180))
        bottom_trim = int(round(crop_h * 0.035))
    else:
        side_trim = int(round(crop_w * 0.035))
        top_trim = int(round(crop_h * 0.055))
        bottom_trim = int(round(crop_h * 0.020))

    x1 = max(0, raw_x1 + side_trim)
    x2 = min(w, raw_x2 - side_trim)
    y1 = max(0, raw_y1 + top_trim)
    y2 = min(h, raw_y2 - bottom_trim)
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        x1 = max(0, raw_x1)
        x2 = min(w, raw_x2)
        y1 = max(0, raw_y1)
        y2 = min(h, raw_y2)
    if x2 <= x1 or y2 <= y1:
        return None

    sky_crop = reference_image[y1:y2, x1:x2]
    crop_mask = selected[y1:y2, x1:x2]
    if sky_crop.size == 0 or np.count_nonzero(crop_mask) < 40:
        return None

    if weather not in WARM_REFERENCE_WEATHERS:
        sky_crop, crop_mask = _select_clean_reference_sky_span(sky_crop, crop_mask)
        if sky_crop.size == 0 or np.count_nonzero(crop_mask) < 40:
            return None
    else:
        crop_mask = _suppress_reference_vertical_obstacles(sky_crop, crop_mask)

    non_sky_mask = ((crop_mask == 0).astype(np.uint8) * 255)
    if np.count_nonzero(non_sky_mask) > 0:
        radius = max(3, int(round(min(sky_crop.shape[:2]) * 0.012)))
        if weather in WARM_REFERENCE_WEATHERS:
            radius = max(radius, int(round(min(sky_crop.shape[:2]) * 0.026)))
        sky_filled = cv2.inpaint(sky_crop, non_sky_mask, radius, cv2.INPAINT_TELEA)
        alpha = cv2.GaussianBlur(crop_mask.astype(np.float32) / 255.0, (0, 0), 1.1)
        alpha = np.clip(alpha * 1.25, 0.0, 1.0)
        sky_crop = np.clip(
            sky_filled.astype(np.float32) * (1.0 - alpha[..., None])
            + sky_crop.astype(np.float32) * alpha[..., None],
            0,
            255,
        ).astype(np.uint8)

    return sky_crop


def _select_reference_sky_mask(mask: np.ndarray, weather: str) -> tuple[np.ndarray | None, bool]:
    components = _extract_mask_components(mask, min_area_ratio=0.00025)
    if not components:
        return None, False

    selected = np.asarray(components[0]["mask"], dtype=np.uint8)
    if weather not in WARM_REFERENCE_WEATHERS or len(components) < 2:
        return selected, False

    image_h, image_w = mask.shape[:2]
    largest_area = float(components[0]["area"])
    largest_x = int(components[0]["x"])
    largest_w = int(components[0]["w"])
    largest_x2 = largest_x + largest_w
    candidates: list[dict[str, np.ndarray | int | float]] = []
    for component in components:
        x = int(component["x"])
        y = int(component["y"])
        w = int(component["w"])
        h = int(component["h"])
        area = float(component["area"])
        center_y = (y + h * 0.5) / max(float(image_h), 1.0)

        if area < max(80.0, largest_area * 0.12):
            continue
        if w < max(12, int(image_w * 0.012)) or h < max(12, int(image_h * 0.020)):
            continue
        if center_y > 0.82 or y > int(image_h * 0.84):
            continue
        if component is not components[0]:
            x2 = x + w
            horizontal_gap = max(largest_x - x2, x - largest_x2, 0)
            if horizontal_gap > max(18, int(largest_w * 0.08)):
                continue
        candidates.append(component)

    if len(candidates) < 2:
        return selected, False

    merged = np.zeros_like(mask, dtype=np.uint8)
    for component in candidates:
        merged = cv2.bitwise_or(merged, np.asarray(component["mask"], dtype=np.uint8))

    if np.count_nonzero(merged) <= np.count_nonzero(selected):
        return selected, False

    return merged, True


def _reference_vertical_obstacle_columns(sky_crop: np.ndarray) -> np.ndarray:
    h, w = sky_crop.shape[:2]
    if h < 20 or w < 40:
        return np.zeros(w, dtype=bool)

    gray = cv2.cvtColor(sky_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark_cut = float(np.quantile(gray, 0.32))
    dark_column_ratio = (gray < dark_cut).mean(axis=0)

    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    edge_cut = float(np.quantile(sobel_x, 0.88))
    edge_column_ratio = (sobel_x > max(edge_cut, 12.0)).mean(axis=0)

    obstacle_columns = (dark_column_ratio > 0.52) | (
        (dark_column_ratio > 0.34) & (edge_column_ratio > 0.16)
    )
    obstacle_u8 = obstacle_columns.astype(np.uint8)[np.newaxis, :] * 255
    obstacle_u8 = cv2.dilate(
        obstacle_u8,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)),
        iterations=1,
    )[0] > 0
    return obstacle_u8


def _suppress_reference_vertical_obstacles(sky_crop: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    obstacle_columns = _reference_vertical_obstacle_columns(sky_crop)
    if not np.any(obstacle_columns):
        return crop_mask

    original_count = int(np.count_nonzero(crop_mask))
    suppressed = crop_mask.copy()
    suppressed[:, obstacle_columns] = 0
    if np.count_nonzero(suppressed) < max(40, int(original_count * 0.35)):
        return crop_mask
    return suppressed


def _select_clean_reference_sky_span(sky_crop: np.ndarray, crop_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = sky_crop.shape[:2]
    if h < 20 or w < 40:
        return sky_crop, crop_mask

    clear_columns = ~_reference_vertical_obstacle_columns(sky_crop)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_clear in enumerate(clear_columns):
        if is_clear and start is None:
            start = idx
        elif not is_clear and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, w))
    if not runs:
        return sky_crop, crop_mask

    best_start, best_end = max(runs, key=lambda item: item[1] - item[0])
    best_width = best_end - best_start
    if best_width < max(40, int(w * 0.34)) or best_width > int(w * 0.96):
        return sky_crop, crop_mask

    pad = max(0, int(round(best_width * 0.015)))
    best_start = max(0, best_start + pad)
    best_end = min(w, best_end - pad)
    if best_end <= best_start + 8:
        return sky_crop, crop_mask

    return sky_crop[:, best_start:best_end], crop_mask[:, best_start:best_end]


def _build_reference_sky_asset(
    reference_image: np.ndarray | None,
    reference_sky_mask: np.ndarray | None,
    opening_mask: np.ndarray,
    width: int,
    height: int,
    weather: str,
) -> np.ndarray | None:
    texture = _extract_reference_sky_texture(reference_image, reference_sky_mask, weather)
    if texture is None:
        return None

    focus_y = 0.58 if weather in {"sunrise", "sunset"} else 0.48
    layer = _fit_cover(texture, width, height, focus_x=0.50, focus_y=focus_y)

    opening_binary = (opening_mask > 0).astype(np.uint8) * 255
    components = _extract_mask_components(opening_binary, min_area_ratio=0.00025)
    usable_components = [
        component
        for component in components
        if int(component["w"]) >= 8 and int(component["h"]) >= 8
    ]
    if usable_components:
        x1 = max(0, min(int(component["x"]) for component in usable_components))
        y1 = max(0, min(int(component["y"]) for component in usable_components))
        x2 = min(width, max(int(component["x"]) + int(component["w"]) for component in usable_components))
        y2 = min(height, max(int(component["y"]) + int(component["h"]) for component in usable_components))
        if x2 > x1 + 8 and y2 > y1 + 8:
            layer[y1:y2, x1:x2] = _fit_cover(
                texture,
                x2 - x1,
                y2 - y1,
                focus_x=0.50,
                focus_y=focus_y,
            )

    logging.info("[FullSceneGenerator] Using sky texture extracted from reference/%s", REFERENCE_WEATHER_DIRS.get(weather, weather))
    return layer


def _resolve_processing_max_dim(processing_max_dim: int | None) -> int:
    if processing_max_dim is not None:
        return max(1200, int(processing_max_dim))
    raw_value = os.environ.get("PIXELDWELL_SCENE_MAX_DIM", "2200").strip()
    try:
        return max(1200, int(raw_value))
    except ValueError:
        return 2200


def _resolve_sky_relight_intensity() -> float:
    raw_value = os.environ.get("PIXELDWELL_SKY_RELIGHT_INTENSITY", "0.32").strip()
    try:
        return float(np.clip(float(raw_value), 0.0, 1.0))
    except ValueError:
        return 0.32


def _resolve_sky_environment_match_intensity() -> float:
    raw_value = os.environ.get("PIXELDWELL_SKY_ENV_MATCH_INTENSITY", "0.65").strip()
    try:
        return float(np.clip(float(raw_value), 0.0, 1.0))
    except ValueError:
        return 0.65


def _resolve_sky_ground_light_intensity() -> float:
    raw_value = os.environ.get("PIXELDWELL_SKY_GROUND_LIGHT_INTENSITY", "0.72").strip()
    try:
        return float(np.clip(float(raw_value), 0.0, 1.0))
    except ValueError:
        return 0.72


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

    green_content = cv2.inRange(hsv.astype(np.uint8), np.array([25, 14, 18]), np.array([88, 255, 245])) > 0
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
    green_content = cv2.inRange(hsv, np.array([25, 16, 18]), np.array([88, 255, 245]))
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


def _build_secondary_opening_sky_supplement(target_roi_u8: np.ndarray, target_mask_u8: np.ndarray) -> np.ndarray:
    target_mask = (target_mask_u8 > 0).astype(np.uint8) * 255
    h, w = target_mask.shape[:2]
    if np.count_nonzero(target_mask) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    transmission, preserve = _estimate_secondary_opening_transmission(target_roi_u8, target_mask)
    hsv = cv2.cvtColor(target_roi_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(target_roi_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    lightness = lab[:, :, 0] / 255.0

    pale_sky = (lightness > 0.54) & (value > 0.48) & (saturation < 0.34)
    blue_sky = (hue >= 74.0) & (hue <= 134.0) & (saturation > 0.045) & (value > 0.30)
    upper_air = (lightness > 0.60) & (value > 0.55) & (saturation < 0.24)

    hsv_u8 = hsv.astype(np.uint8)
    green_content = cv2.inRange(hsv_u8, np.array([25, 20, 20]), np.array([88, 255, 245])) > 0
    brown_content = cv2.inRange(hsv_u8, np.array([8, 28, 20]), np.array([38, 255, 230])) > 0
    dark_structure = (value < 0.34) & (saturation > 0.06)
    exterior_content = green_content | brown_content | dark_structure

    valid = target_mask > 0
    cutoff = h
    min_valid = max(6, int(w * 0.08))
    band_h = max(6, int(h * 0.035))
    content_rows: list[float] = []
    for y in range(h):
        row_valid = valid[y]
        valid_count = int(np.count_nonzero(row_valid))
        if valid_count < min_valid:
            content_rows.append(0.0)
            continue
        content_rows.append(np.count_nonzero(exterior_content[y][row_valid]) / float(valid_count))

    content_arr = np.array(content_rows, dtype=np.float32)
    for y in range(max(1, int(h * 0.34)), max(1, h - band_h + 1)):
        if float(np.mean(content_arr[y:y + band_h])) > 0.18:
            cutoff = max(1, y - int(round(h * 0.035)))
            break
    cutoff = min(cutoff, max(1, int(round(h * 0.74))))

    candidate = (
        valid
        & (transmission > 0.34)
        & (preserve < 0.86)
        & (pale_sky | blue_sky | upper_air)
        & ~exterior_content
    )
    candidate[cutoff:, :] = False

    candidate_u8 = candidate.astype(np.uint8) * 255
    if np.count_nonzero(candidate_u8) == 0:
        return candidate_u8

    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
        iterations=1,
    )
    candidate_u8 = cv2.GaussianBlur(candidate_u8.astype(np.float32), (0, 0), 0.55)
    candidate_u8 = ((candidate_u8 > 24.0).astype(np.uint8) * 255)
    return cv2.bitwise_and(candidate_u8, target_mask)


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
    night_sky = (
        (hue >= 82.0)
        & (hue <= 136.0)
        & (saturation > 0.055)
        & (value > 0.055)
        & (value < 0.62)
    )
    moon_haze = (lightness > 0.30) & (saturation < 0.30) & (value > 0.18)

    if weather == "night":
        sky_like = night_sky | moon_haze | blue_sky
        top_ratio = 0.66
    elif weather in DIFFUSE_CLOUD_WEATHERS:
        sky_like = gray_cloud | pale_sky | blue_sky
        top_ratio = 0.56
    elif weather == "clear":
        sky_like = blue_sky | pale_sky
        top_ratio = 0.58
    else:
        sky_like = pale_sky | blue_sky
        top_ratio = 0.58

    upper_band = np.zeros((h, w), dtype=np.uint8)
    upper_ratio = 0.88 if weather == "night" else (0.74 if weather in DIFFUSE_CLOUD_WEATHERS else 0.82)
    upper_band[: max(1, int(h * upper_ratio)), :] = 255
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
    elif weather in DIFFUSE_CLOUD_WEATHERS:
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
            elif weather in DIFFUSE_CLOUD_WEATHERS:
                direct_reference_strength = 0.22
            elif weather == "clear":
                direct_reference_strength = 0.0
        elif weather in {"sunrise", "sunset"}:
            direct_reference_strength = 0.10
        elif weather in DIFFUSE_CLOUD_WEATHERS:
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
                if weather in DIFFUSE_CLOUD_WEATHERS:
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
            if weather in DIFFUSE_CLOUD_WEATHERS:
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
        if weather in DIFFUSE_CLOUD_WEATHERS:
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
        elif weather in DIFFUSE_CLOUD_WEATHERS and cloudy_transfer_mask is not None:
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
    alpha = np.clip(dist / 2.8, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.75)
    alpha[cleaned == 0] = 0.0
    return alpha.astype(np.float32)


def _fill_small_replacement_holes(mask: np.ndarray) -> np.ndarray:
    replacement = (mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(replacement) == 0:
        return replacement

    padded = cv2.copyMakeBorder(replacement, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    background = cv2.bitwise_not(padded)
    cv2.floodFill(background, None, (0, 0), 0)
    holes = background[1:-1, 1:-1]
    if np.count_nonzero(holes) == 0:
        return replacement

    mask_area = int(np.count_nonzero(replacement))
    max_hole_area = max(48, int(mask_area * 0.006))
    h, w = replacement.shape[:2]
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    filled = replacement.copy()
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        hole_w = int(stats[label, cv2.CC_STAT_WIDTH])
        hole_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        long_structure = (
            (hole_w > w * 0.035 and hole_h < h * 0.018)
            or (hole_h > h * 0.035 and hole_w < w * 0.018)
        )
        if area <= max_hole_area and not long_structure:
            filled[labels == label] = 255
    return filled


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


def _restore_window_structure(
    original: np.ndarray,
    result: np.ndarray,
    opening_mask: np.ndarray,
    replacement_mask: np.ndarray,
) -> np.ndarray:
    opening_binary = (opening_mask > 0).astype(np.uint8) * 255
    replacement_binary = (replacement_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(replacement_binary) == 0:
        return result
    if np.count_nonzero(opening_binary) == 0:
        opening_binary = cv2.dilate(
            replacement_binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )

    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = cv2.GaussianBlur(np.abs(grad_x) + np.abs(grad_y), (3, 3), 0)

    structure_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    replacement_edge_band = cv2.bitwise_xor(
        cv2.dilate(replacement_binary, structure_kernel, iterations=1),
        cv2.erode(replacement_binary, structure_kernel, iterations=1),
    ) > 0
    dark_frame = ((value < 0.36) & (saturation > 0.035)) | (gray < 52.0)
    pale_frame_edge = (
        (saturation < 0.09)
        & (value > 0.48)
        & (edge_strength > 26.0)
        & replacement_edge_band
    )
    structure = (
        (opening_binary > 0)
        & (replacement_binary > 0)
        & (dark_frame | pale_frame_edge)
    ).astype(np.uint8) * 255
    if np.count_nonzero(structure) == 0:
        return result

    structure = cv2.morphologyEx(
        structure,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    alpha = cv2.GaussianBlur(structure.astype(np.float32) / 255.0, (0, 0), 0.65)
    alpha = np.clip(alpha * 1.12, 0.0, 1.0)
    restored = result.astype(np.float32) * (1.0 - alpha[..., None]) + original.astype(np.float32) * alpha[..., None]
    return np.clip(restored, 0, 255).astype(np.uint8)


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


def _build_sky_replacement_mask(sky_mask: np.ndarray, opening_mask: np.ndarray) -> np.ndarray:
    """
    Build the actual replacement region.

    The opening mask is only a boundary hint. Replacing the full detected door/window
    makes the result look like a poster and destroys the exterior view. The sky mask
    is the source of truth; it is only expanded slightly for seam coverage.
    """
    return _build_sky_replacement_mask_for_image(None, sky_mask, opening_mask)


def _mask_2d(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("mask is required")
    if mask.ndim == 3:
        return mask[:, :, 0]
    return mask


def _augment_sky_mask_from_openings(
    image: np.ndarray,
    sky_mask: np.ndarray,
    opening_mask: np.ndarray,
) -> np.ndarray:
    sky_binary = (_mask_2d(sky_mask) > 0).astype(np.uint8) * 255
    opening_binary = (_mask_2d(opening_mask) > 0).astype(np.uint8) * 255
    if image is None or np.count_nonzero(opening_binary) == 0:
        return sky_binary

    augmented = sky_binary.copy()
    components = _extract_mask_components(opening_binary, min_area_ratio=0.00035)
    image_area = float(max(image.shape[0] * image.shape[1], 1))
    for component in components:
        x = int(component["x"])
        y = int(component["y"])
        component_w = int(component["w"])
        component_h = int(component["h"])
        area_ratio = float(component["area"]) / image_area
        if component_w < 16 or component_h < 16 or area_ratio < 0.00025:
            continue

        roi = image[y:y + component_h, x:x + component_w]
        roi_mask = _mask_2d(np.asarray(component["mask"], dtype=np.uint8))[y:y + component_h, x:x + component_w]
        if roi.size == 0 or np.count_nonzero(roi_mask) == 0:
            continue

        visibility = _estimate_opening_visibility(roi, roi_mask)
        candidate = _build_opening_sky_candidate(roi, visibility, roi_mask)

        candidate_ratio = np.count_nonzero(candidate) / float(max(np.count_nonzero(roi_mask), 1))
        is_secondary_opening = (
            area_ratio < 0.020
            or component_w < image.shape[1] * 0.12
            or component_h < image.shape[0] * 0.24
        )
        if is_secondary_opening or candidate_ratio < 0.018:
            transmission_mask = _build_secondary_opening_sky_supplement(roi, roi_mask)
            if np.count_nonzero(transmission_mask) > 0:
                if is_secondary_opening:
                    candidate = cv2.bitwise_or(candidate, transmission_mask)
                elif np.count_nonzero(transmission_mask) > np.count_nonzero(candidate):
                    candidate = transmission_mask

        if np.count_nonzero(candidate) == 0:
            continue

        pad = max(1, int(round(min(component_w, component_h) * 0.006)))
        if pad > 1:
            candidate = cv2.dilate(
                candidate,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad | 1, pad | 1)),
                iterations=1,
            )
            candidate = cv2.bitwise_and(candidate, roi_mask)
        candidate = _mask_2d(candidate)
        augmented_roi = _mask_2d(augmented[y:y + component_h, x:x + component_w])
        augmented[y:y + component_h, x:x + component_w] = cv2.bitwise_or(augmented_roi, candidate)

    return (augmented > 0).astype(np.uint8) * 255


def _build_non_sky_protect_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0

    green_content = cv2.inRange(hsv, np.array([25, 14, 18]), np.array([88, 255, 245])) > 0
    brown_content = cv2.inRange(hsv, np.array([8, 28, 18]), np.array([38, 255, 230])) > 0
    dark_structure = (value < 0.36) & (saturation > 0.045)
    black_structure = value < 0.16
    return (green_content | brown_content | dark_structure | black_structure).astype(np.uint8) * 255


def _polish_replacement_mask_edges(
    image: np.ndarray,
    replacement_mask: np.ndarray,
    opening_mask: np.ndarray,
) -> np.ndarray:
    replacement = (replacement_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(replacement) == 0:
        return replacement

    allowed = (opening_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(allowed) == 0:
        allowed = np.full(replacement.shape[:2], 255, dtype=np.uint8)

    protect = _build_non_sky_protect_mask(image)
    polished = replacement.copy()
    components = _extract_mask_components(replacement, min_area_ratio=0.00018)
    for component in components:
        x = int(component["x"])
        y = int(component["y"])
        component_w = int(component["w"])
        component_h = int(component["h"])
        if component_w < 24 or component_h < 24:
            continue

        component_roi = np.asarray(component["mask"], dtype=np.uint8)[y:y + component_h, x:x + component_w]
        allowed_roi = allowed[y:y + component_h, x:x + component_w]
        protect_roi = protect[y:y + component_h, x:x + component_w]
        fill_roi = np.zeros_like(component_roi, dtype=np.uint8)

        min_pixels = max(5, int(round(component_w * 0.05)))
        for row_index in range(component_h):
            row = component_roi[row_index] > 0
            active_x = np.where(row)[0]
            if active_x.size < min_pixels:
                continue

            coverage = active_x.size / float(max(component_w, 1))
            if row_index > component_h * 0.86 and coverage < 0.62:
                continue
            if row_index > component_h * 0.66 and coverage < 0.24:
                continue

            pad = max(1, int(round(component_w * 0.006)))
            left = max(0, int(active_x.min()) - pad)
            right = min(component_w - 1, int(active_x.max()) + pad)
            if right <= left:
                continue

            row_fill = np.zeros(component_w, dtype=np.uint8)
            row_fill[left:right + 1] = 255
            row_fill = cv2.bitwise_and(row_fill, allowed_roi[row_index])
            row_fill = np.asarray(row_fill, dtype=np.uint8).reshape(-1)
            row_fill[protect_roi[row_index] > 0] = 0
            fill_roi[row_index] = np.maximum(fill_roi[row_index], row_fill)

        if np.count_nonzero(fill_roi) == 0:
            continue

        polished_roi = polished[y:y + component_h, x:x + component_w]
        polished[y:y + component_h, x:x + component_w] = cv2.bitwise_or(polished_roi, fill_roi)

    polished = cv2.morphologyEx(
        polished,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    polished = cv2.bitwise_and(polished, allowed)
    return (polished > 0).astype(np.uint8) * 255


def _build_sky_replacement_mask_for_image(
    image: np.ndarray | None,
    sky_mask: np.ndarray,
    opening_mask: np.ndarray,
) -> np.ndarray:
    sky_binary = (_mask_2d(sky_mask) > 0).astype(np.uint8) * 255
    opening_binary = (_mask_2d(opening_mask) > 0).astype(np.uint8) * 255
    if np.count_nonzero(sky_binary) == 0:
        if image is not None and np.count_nonzero(opening_binary) > 0:
            sky_binary = _augment_sky_mask_from_openings(image, sky_binary, opening_binary)
        if np.count_nonzero(sky_binary) == 0:
            return sky_binary

    if image is not None and np.count_nonzero(opening_binary) > 0:
        sky_binary = _augment_sky_mask_from_openings(image, sky_binary, opening_binary)

    h, w = sky_binary.shape[:2]
    if np.count_nonzero(opening_binary) > 0:
        constrained = cv2.bitwise_and(sky_binary, opening_binary)
        if np.count_nonzero(constrained) > 0:
            sky_binary = constrained

    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    replacement = cv2.morphologyEx(sky_binary, cv2.MORPH_CLOSE, cleanup_kernel, iterations=1)

    seam_kernel_size = max(3, int(round(min(h, w) * 0.0025))) | 1
    seam_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seam_kernel_size, seam_kernel_size))
    replacement = cv2.dilate(replacement, seam_kernel, iterations=1)

    if np.count_nonzero(opening_binary) > 0:
        replacement = cv2.bitwise_and(replacement, opening_binary)

    # ── Final Safety Check: Reject Fallback Rectangles ──────────────────────
    # If the mask is nearly a perfect rectangle, it is likely a YOLO fallback box.
    # Regenerating a box causes the AI to overwrite walls and frames with "sky".
    x_r, y_r, w_r, h_r = cv2.boundingRect(replacement)
    if w_r > 0 and h_r > 0:
        rect_area = w_r * h_r
        mask_pixels = np.count_nonzero(replacement)
        if mask_pixels / rect_area > 0.96:
            logging.info("[MaskGen] Rejecting rectangular sky mask to prevent box artifact.")
            return np.zeros_like(replacement)

    if image is not None:
        replacement = _polish_replacement_mask_edges(image, replacement, opening_binary)

    replacement = _fill_small_replacement_holes(replacement)
    return (replacement > 0).astype(np.uint8) * 255


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
    if weather == "night":
        floor_gate = np.clip((yy - h * 0.46) / max(h * 0.46, 1.0), 0.0, 1.0)
    elif weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        floor_gate = np.clip((yy - h * 0.48) / max(h * 0.42, 1.0), 0.0, 1.0)
    else:
        floor_gate = np.clip((yy - h * 0.56) / max(h * 0.44, 1.0), 0.0, 1.0)
    beam_gate = _build_floor_beam_mask((h, w), openings)
    if weather == "night":
        candidate_zone = _build_floor_cleanup_zone((h, w), openings)
    elif weather in DIFFUSE_CLOUD_WEATHERS:
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
    if weather == "night":
        background_selector = (floor_gate > 0.10).astype(np.uint8) * 255
    elif weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        background_selector = (floor_gate > 0.08).astype(np.uint8) * 255
    else:
        background_selector = ((floor_gate > 0.18) & (beam_gate < (0.10 if weather in DIFFUSE_CLOUD_WEATHERS else 0.18))).astype(np.uint8) * 255
    background_ring = cv2.bitwise_and(
        background_ring,
        background_selector,
    )
    if np.count_nonzero(background_ring) < max(200, int(h * w * 0.002)):
        return image

    if weather == "night":
        background_l = float(np.quantile(l_coarse[background_ring > 0], 0.30))
    elif weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        background_l = float(np.quantile(l_coarse[background_ring > 0], 0.35))
    else:
        background_l = float(np.median(l_coarse[background_ring > 0]))
    background_a = float(np.median(a_coarse[background_ring > 0]))
    background_b = float(np.median(b_coarse[background_ring > 0]))
    brightness_delta = l_coarse - background_l
    warmth_delta = b_coarse - background_b
    if weather == "night":
        floor_l = l_channel[floor_gate > 0.12]
        high_floor_l = float(np.quantile(floor_l, 0.82)) if floor_l.size else background_l + 18.0
        bright_candidate = (
            (brightness_delta > 5.0)
            | ((brightness_delta > 2.8) & (warmth_delta > 0.3))
            | (l_channel > max(background_l + 13.0, high_floor_l))
        )
        min_component_area = max(220, int(h * w * 0.0014))
        soft_gain = 1.62
    elif weather in DIFFUSE_CLOUD_WEATHERS:
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
    if weather == "night":
        patch_mask = cv2.dilate(
            patch_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
            iterations=1,
        )
    elif weather in PROJECTED_GROUND_LIGHT_WEATHERS:
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
    if weather == "night":
        patch_binary = patch_mask > 0
        target_l[patch_binary] = np.minimum(target_l[patch_binary], background_l + 1.0)
        target_a[patch_binary] = target_a[patch_binary] * 0.50 + background_a * 0.50
        target_b[patch_binary] = np.minimum(target_b[patch_binary], background_b - 1.0)
    elif weather in PROJECTED_GROUND_LIGHT_WEATHERS:
        patch_binary = patch_mask > 0
        target_l[patch_binary] = np.minimum(target_l[patch_binary], background_l + 2.0)
        target_a[patch_binary] = target_a[patch_binary] * 0.55 + background_a * 0.45
        target_b[patch_binary] = np.minimum(target_b[patch_binary], background_b + 1.0)

    l_detail = l_channel - l_coarse
    a_detail = a_channel - a_coarse
    b_detail = b_channel - b_coarse

    if weather == "night":
        new_l = target_l + 0.24 * l_detail
        new_a = target_a + 0.10 * a_detail
        new_b = target_b + 0.08 * b_detail
    elif weather in DIFFUSE_CLOUD_WEATHERS:
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


def _suppress_golden_hour_sun_disk(sky_layer: np.ndarray) -> np.ndarray:
    h, w = sky_layer.shape[:2]
    if h < 24 or w < 24:
        return sky_layer.copy()

    hsv = cv2.cvtColor(sky_layer, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(sky_layer, cv2.COLOR_BGR2LAB).astype(np.float32)
    value = hsv[:, :, 2] / 255.0
    saturation = hsv[:, :, 1] / 255.0
    lightness = lab[:, :, 0] / 255.0
    warmth = lab[:, :, 2]
    red = sky_layer[:, :, 2].astype(np.float32)
    blue = sky_layer[:, :, 0].astype(np.float32)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]

    bright_cutoff = max(0.78, float(np.quantile(value, 0.992)) * 0.94)
    sun_core = (
        (value >= bright_cutoff)
        & (lightness > 0.70)
        & (saturation > 0.12)
        & (warmth > 140.0)
        & (red > blue + 14.0)
        & (yy > 0.38)
    ).astype(np.uint8) * 255
    if np.count_nonzero(sun_core) < max(12, int(h * w * 0.000015)):
        return sky_layer.copy()

    count, labels, stats, _ = cv2.connectedComponentsWithStats((sun_core > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return sky_layer.copy()

    best_label = 0
    best_score = 0.0
    image_area = float(max(h * w, 1))
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < max(8.0, image_area * 0.00001) or area > image_area * 0.06:
            continue
        component_values = value[labels == label]
        score = area * (float(component_values.mean()) if component_values.size else 0.0)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label == 0:
        return sky_layer.copy()

    sun_mask = np.zeros((h, w), dtype=np.uint8)
    sun_mask[labels == best_label] = 255
    kernel_size = max(9, int(round(min(h, w) * 0.035)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    sun_mask = cv2.dilate(sun_mask, kernel, iterations=2)
    sun_mask = cv2.morphologyEx(sun_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    inpaint_radius = max(3, int(round(min(h, w) * 0.014)))
    return cv2.inpaint(sky_layer, sun_mask, inpaint_radius, cv2.INPAINT_TELEA)


def _vertical_overlap_ratio(a: dict[str, int | float], b: dict[str, int | float]) -> float:
    top = max(int(a["y"]), int(b["y"]))
    bottom = min(int(a["y"]) + int(a["h"]), int(b["y"]) + int(b["h"]))
    overlap = max(0, bottom - top)
    return overlap / float(max(1, min(int(a["h"]), int(b["h"]))))


def _horizontal_gap(a: dict[str, int | float], b: dict[str, int | float]) -> int:
    a_left = int(a["x"])
    a_right = int(a["x"]) + int(a["w"])
    b_left = int(b["x"])
    b_right = int(b["x"]) + int(b["w"])
    if a_right < b_left:
        return b_left - a_right
    if b_right < a_left:
        return a_left - b_right
    return 0


def _fit_sky_layer_per_replacement_component(sky_layer: np.ndarray, replacement_mask: np.ndarray) -> np.ndarray:
    replacement = (replacement_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(replacement) == 0:
        return sky_layer

    image_h, image_w = replacement.shape[:2]
    sunless_sky = _suppress_golden_hour_sun_disk(sky_layer)
    secondary_h = max(24, int(round(sunless_sky.shape[0] * 0.62)))
    secondary_source = sunless_sky[:secondary_h, :].copy()
    localized = _fit_cover(secondary_source, image_w, image_h, focus_x=0.50, focus_y=0.48)
    components = _extract_mask_components(replacement, min_area_ratio=0.00004)
    usable_components = [
        component
        for component in components
        if int(component["w"]) >= 8 and int(component["h"]) >= 8
    ]
    if not usable_components:
        return localized

    primary = usable_components[0]
    join_gap = max(18, int(round(image_w * 0.035)))
    primary_group: set[int] = {0}
    changed = True
    while changed:
        changed = False
        grouped = [usable_components[index] for index in primary_group]
        for index, component in enumerate(usable_components):
            if index in primary_group:
                continue
            same_row = any(_vertical_overlap_ratio(component, member) >= 0.42 for member in grouped)
            close_enough = any(_horizontal_gap(component, member) <= join_gap for member in grouped)
            meaningful_piece = float(component["area"]) >= max(80.0, float(primary["area"]) * 0.035)
            if same_row and close_enough and meaningful_piece:
                primary_group.add(index)
                changed = True

    primary_mask = np.zeros_like(replacement, dtype=np.uint8)
    x1 = image_w
    y1 = image_h
    x2 = 0
    y2 = 0
    for index in primary_group:
        component = usable_components[index]
        primary_mask = cv2.bitwise_or(primary_mask, np.asarray(component["mask"], dtype=np.uint8))
        x = int(component["x"])
        y = int(component["y"])
        component_w = int(component["w"])
        component_h = int(component["h"])
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x + component_w)
        y2 = max(y2, y + component_h)

    if x2 > x1 and y2 > y1 and np.count_nonzero(primary_mask) > 0:
        group_w = x2 - x1
        group_h = y2 - y1
        group_region = primary_mask[y1:y2, x1:x2] > 0
        fitted = _fit_cover(sky_layer, group_w, group_h, focus_x=0.50, focus_y=0.56)
        roi = localized[y1:y2, x1:x2]
        roi[group_region] = fitted[group_region]
        localized[y1:y2, x1:x2] = roi

    for index, component in enumerate(usable_components):
        if index in primary_group:
            continue
        x = int(component["x"])
        y = int(component["y"])
        component_w = int(component["w"])
        component_h = int(component["h"])

        component_mask = np.asarray(component["mask"], dtype=np.uint8)[y:y + component_h, x:x + component_w] > 0
        fitted = _fit_cover(secondary_source, component_w, component_h, focus_x=0.50, focus_y=0.48)
        roi = localized[y:y + component_h, x:x + component_w]
        roi[component_mask] = fitted[component_mask]
        localized[y:y + component_h, x:x + component_w] = roi

    return localized


def _masked_lab_mean(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if image is None or mask is None:
        return None
    binary = mask > 0
    if int(np.count_nonzero(binary)) < 200:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[binary].mean(axis=0)


def _masked_lab_stats(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if image is None or mask is None:
        return None
    binary = mask > 0
    if int(np.count_nonzero(binary)) < 200:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab[binary]
    if samples.shape[0] > 8000:
        step = max(1, samples.shape[0] // 8000)
        samples = samples[::step]
    return samples.mean(axis=0), np.maximum(samples.std(axis=0), 1.0)


def _weather_scene_lab_bias(weather: str) -> np.ndarray:
    if weather == "sunrise":
        return np.array([0.0, 4.0, 22.0], dtype=np.float32)
    if weather == "sunset":
        return np.array([-2.0, 7.0, 26.0], dtype=np.float32)
    if weather == "golden_hour":
        return np.array([1.0, 6.0, 34.0], dtype=np.float32)
    if weather == "clear":
        return np.array([2.0, -1.0, -8.0], dtype=np.float32)
    if weather in DIFFUSE_CLOUD_WEATHERS:
        return np.array([-5.0, 0.0, -8.0], dtype=np.float32)
    if weather == "night":
        return np.array([-148.0, 1.0, -52.0], dtype=np.float32)
    return np.array([0.0, 2.0, 8.0], dtype=np.float32)


def _harmonize_sky_layer_to_scene(
    original_image: np.ndarray,
    sky_layer: np.ndarray,
    replacement_mask: np.ndarray,
    opening_mask: np.ndarray,
    weather: str,
) -> np.ndarray:
    replacement_binary = (replacement_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(replacement_binary) < 200:
        return sky_layer

    source_stats = _masked_lab_stats(sky_layer, replacement_binary)
    target_stats = _masked_lab_stats(original_image, replacement_binary)
    if target_stats is None and np.count_nonzero(opening_mask) > 0:
        target_stats = _masked_lab_stats(original_image, opening_mask)
    if source_stats is None or target_stats is None:
        return sky_layer

    source_mean, source_std = source_stats
    target_mean, target_std = target_stats
    target_mean = target_mean + _weather_scene_lab_bias(weather)
    if weather == "night":
        target_mean[0] = float(np.clip(target_mean[0], 22.0, 56.0))
        target_mean[1] = float(np.clip(target_mean[1], 124.0, 138.0))
        target_mean[2] = float(np.clip(target_mean[2], 66.0, 96.0))
    else:
        target_mean[0] = float(np.clip(target_mean[0], 120.0, 238.0))
        target_mean[1] = float(np.clip(target_mean[1], 112.0, 150.0))
        target_mean[2] = float(np.clip(target_mean[2], 104.0, 166.0))

    mean_delta = target_mean - source_mean
    if weather == "night":
        mean_delta[0] = float(np.clip(mean_delta[0], -130.0, -8.0))
        mean_delta[1] = float(np.clip(mean_delta[1], -12.0, 16.0))
        mean_delta[2] = float(np.clip(mean_delta[2], -58.0, 4.0))
    elif weather in {"golden_hour", "sunset"}:
        mean_delta[0] = float(np.clip(mean_delta[0], -24.0, 24.0))
        mean_delta[1] = float(np.clip(mean_delta[1], -10.0, 12.0))
        mean_delta[2] = float(np.clip(mean_delta[2], -8.0, 16.0))
    elif weather in DIFFUSE_CLOUD_WEATHERS:
        mean_delta[0] = float(np.clip(mean_delta[0], -14.0, 10.0))
        mean_delta[1] = float(np.clip(mean_delta[1], -8.0, 8.0))
        mean_delta[2] = float(np.clip(mean_delta[2], -10.0, 8.0))
    else:
        mean_delta[0] = float(np.clip(mean_delta[0], -34.0, 28.0))
        mean_delta[1] = float(np.clip(mean_delta[1], -18.0, 18.0))
        mean_delta[2] = float(np.clip(mean_delta[2], -30.0, 26.0))

    contrast_scale = np.clip(target_std / source_std, 0.72, 1.28)
    contrast_strength = np.array([0.38, 0.18, 0.18], dtype=np.float32)
    mean_strength = np.array([0.86, 0.70, 0.78], dtype=np.float32)
    if weather == "golden_hour":
        contrast_strength = np.array([0.30, 0.12, 0.12], dtype=np.float32)
        mean_strength = np.array([0.48, 0.30, 0.28], dtype=np.float32)
    elif weather == "sunset":
        contrast_strength = np.array([0.32, 0.12, 0.12], dtype=np.float32)
        mean_strength = np.array([0.52, 0.34, 0.34], dtype=np.float32)
    elif weather in {"sunrise", "sunset", "dusk", "dramatic"}:
        mean_strength = np.array([0.80, 0.62, 0.68], dtype=np.float32)
    elif weather == "clear":
        contrast_strength = np.array([0.22, 0.08, 0.08], dtype=np.float32)
        mean_strength = np.array([0.26, 0.14, 0.10], dtype=np.float32)
    elif weather in DIFFUSE_CLOUD_WEATHERS:
        contrast_strength = np.array([0.18, 0.06, 0.06], dtype=np.float32)
        mean_strength = np.array([0.18, 0.12, 0.12], dtype=np.float32)
    elif weather == "night":
        contrast_strength = np.array([0.18, 0.08, 0.12], dtype=np.float32)
        mean_strength = np.array([0.74, 0.38, 0.60], dtype=np.float32)

    lab = cv2.cvtColor(sky_layer, cv2.COLOR_BGR2LAB).astype(np.float32)
    adjusted = lab.copy()
    for channel_index in range(3):
        channel = adjusted[:, :, channel_index]
        channel = source_mean[channel_index] + (channel - source_mean[channel_index]) * (
            1.0 + (contrast_scale[channel_index] - 1.0) * contrast_strength[channel_index]
        )
        channel = channel + mean_delta[channel_index] * mean_strength[channel_index]
        adjusted[:, :, channel_index] = channel

    return cv2.cvtColor(np.clip(adjusted, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _apply_opening_environment_tone(
    original_image: np.ndarray,
    result_image: np.ndarray,
    opening_mask: np.ndarray,
    replacement_mask: np.ndarray,
    sky_layer: np.ndarray,
    weather: str,
) -> np.ndarray:
    if np.count_nonzero(opening_mask) == 0 or np.count_nonzero(replacement_mask) == 0:
        return result_image

    sky_mean = _masked_lab_mean(sky_layer, replacement_mask)
    original_sky_mean = _masked_lab_mean(original_image, replacement_mask)
    if sky_mean is None or original_sky_mean is None:
        return result_image

    delta = sky_mean - original_sky_mean
    if weather == "night":
        delta[0] = float(np.clip(delta[0], -98.0, -8.0))
        delta[1] = float(np.clip(delta[1], -14.0, 18.0))
        delta[2] = float(np.clip(delta[2], -64.0, 6.0))
    else:
        delta[0] = float(np.clip(delta[0], -24.0, 18.0))
        delta[1] = float(np.clip(delta[1], -18.0, 22.0))
        delta[2] = float(np.clip(delta[2], -22.0, 30.0))

    replacement_guard = cv2.dilate(
        (replacement_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    exterior_mask = cv2.bitwise_and((opening_mask > 0).astype(np.uint8) * 255, cv2.bitwise_not(replacement_guard))
    if np.count_nonzero(exterior_mask) < 200:
        return result_image

    h, w = original_image.shape[:2]
    orig_lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    result_lab = cv2.cvtColor(result_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail_gate = np.clip((gray - 42.0) / 95.0, 0.0, 1.0)

    is_rect = _is_mask_rectangular(opening_mask)
    blur_sigma = max(h, w) * 0.035 if is_rect else 1.4
    alpha = cv2.GaussianBlur(exterior_mask.astype(np.float32) / 255.0, (0, 0), blur_sigma)
    alpha = np.clip(alpha * detail_gate, 0.0, 1.0)

    intensity = _resolve_sky_environment_match_intensity()
    if weather in {"sunrise", "sunset"}:
        channel_strength = np.array([0.14, 0.22, 0.28], dtype=np.float32)
    elif weather == "clear":
        channel_strength = np.array([0.08, 0.08, 0.12], dtype=np.float32)
    elif weather in DIFFUSE_CLOUD_WEATHERS:
        channel_strength = np.array([0.16, 0.08, 0.14], dtype=np.float32)
    elif weather == "night":
        channel_strength = np.array([0.70, 0.24, 0.62], dtype=np.float32)
        alpha = np.clip(alpha * 1.35, 0.0, 1.0)
    else:
        channel_strength = np.array([0.10, 0.12, 0.16], dtype=np.float32)

    for channel_index in range(3):
        result_lab[:, :, channel_index] = np.clip(
            result_lab[:, :, channel_index]
            + alpha * delta[channel_index] * channel_strength[channel_index] * intensity,
            0.0,
            255.0,
        )

    return cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _apply_night_exterior_view_tone(
    original_image: np.ndarray,
    result_image: np.ndarray,
    opening_mask: np.ndarray,
    replacement_mask: np.ndarray,
) -> np.ndarray:
    opening_binary = (opening_mask > 0).astype(np.uint8) * 255
    replacement_binary = (replacement_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(opening_binary) == 0 or np.count_nonzero(replacement_binary) == 0:
        return result_image

    sky_guard = cv2.dilate(
        replacement_binary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    exterior_mask = cv2.bitwise_and(opening_binary, cv2.bitwise_not(sky_guard))

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    result_hsv = cv2.cvtColor(result_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    result_gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    result_saturation = result_hsv[:, :, 1] / 255.0
    result_value = result_hsv[:, :, 2] / 255.0
    edge_strength = cv2.GaussianBlur(
        np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)),
        (3, 3),
        0,
    )

    frame_band = cv2.bitwise_and(
        opening_binary,
        cv2.subtract(
            cv2.dilate(replacement_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1),
            cv2.erode(replacement_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1),
        ),
    )
    pale_structure = (
        (frame_band > 0)
        & (saturation < 0.24)
        & (value > 0.34)
        & ((edge_strength > 10.0) | (gray > 130.0))
    ).astype(np.uint8) * 255
    result_halo = (
        (frame_band > 0)
        & (result_value > 0.28)
        & ((result_saturation < 0.42) | (result_gray > 112.0))
        & ((edge_strength > 6.0) | (value > 0.28))
    ).astype(np.uint8) * 255
    pale_structure = cv2.bitwise_or(pale_structure, result_halo)

    tone_mask = cv2.bitwise_or(exterior_mask, pale_structure)
    if np.count_nonzero(tone_mask) == 0:
        return result_image

    detail_gate = np.clip((gray - 18.0) / 126.0, 0.34, 1.0)
    structure_gate = (pale_structure > 0).astype(np.float32)
    is_rect = _is_mask_rectangular(opening_mask)
    blur_sigma = max(h, w) * 0.038 if is_rect else 1.45
    alpha = cv2.GaussianBlur(tone_mask.astype(np.float32) / 255.0, (0, 0), blur_sigma)
    alpha = np.clip(alpha * detail_gate * (1.0 - structure_gate * 0.10), 0.0, 0.94)
    halo_alpha = cv2.GaussianBlur(pale_structure.astype(np.float32) / 255.0, (0, 0), 2.2)
    alpha = np.maximum(alpha, np.clip(halo_alpha * 1.18, 0.0, 0.96))

    lab = cv2.cvtColor(result_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    night_lab = lab.copy()
    l_channel = night_lab[:, :, 0]
    is_structure = structure_gate > 0
    structure_l = l_channel * 0.26 + 12.0
    exterior_l = l_channel * 0.30 + 11.0
    night_lab[:, :, 0] = np.where(is_structure, np.minimum(l_channel, structure_l), np.minimum(l_channel, exterior_l))
    night_lab[:, :, 1] = night_lab[:, :, 1] * 0.86 + 128.0 * 0.14
    night_lab[:, :, 2] = night_lab[:, :, 2] * 0.58 + 90.0 * 0.42

    night_bgr = cv2.cvtColor(np.clip(night_lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    result = result_image.astype(np.float32) * (1.0 - alpha[..., None]) + night_bgr * alpha[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def _suppress_night_sky_fringe(
    result_image: np.ndarray,
    replacement_mask: np.ndarray,
    opening_mask: np.ndarray,
) -> np.ndarray:
    replacement_binary = (replacement_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(replacement_binary) == 0:
        return result_image

    opening_binary = (opening_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(opening_binary) == 0:
        opening_binary = cv2.dilate(
            replacement_binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )

    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    edge_band = cv2.bitwise_and(
        opening_binary,
        cv2.subtract(
            cv2.dilate(replacement_binary, edge_kernel, iterations=1),
            cv2.erode(replacement_binary, edge_kernel, iterations=1),
        ),
    )
    if np.count_nonzero(edge_band) == 0:
        return result_image

    hsv = cv2.cvtColor(result_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    value = hsv[:, :, 2] / 255.0
    saturation = hsv[:, :, 1] / 255.0
    pale_edge = (
        (edge_band > 0)
        & (value > 0.28)
        & (saturation < 0.52)
    ).astype(np.uint8) * 255
    if np.count_nonzero(pale_edge) == 0:
        return result_image

    sky_core = cv2.erode(
        replacement_binary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
        iterations=1,
    ) > 0
    core_values = value[sky_core]
    if core_values.size:
        value_cutoff = float(np.quantile(core_values, 0.72))
        core_mask = sky_core & (value <= value_cutoff)
    else:
        core_mask = sky_core

    if np.count_nonzero(core_mask) > 24:
        target_color = np.median(result_image[core_mask], axis=0).astype(np.float32)
    else:
        target_color = np.array([42.0, 28.0, 15.0], dtype=np.float32)

    local_sky = cv2.GaussianBlur(result_image.astype(np.float32), (0, 0), 8.0)
    target = local_sky * 0.45 + target_color.reshape(1, 1, 3) * 0.55
    alpha = cv2.GaussianBlur(pale_edge.astype(np.float32) / 255.0, (0, 0), 1.4)
    alpha = np.clip(alpha * 0.92, 0.0, 0.92)
    result = result_image.astype(np.float32) * (1.0 - alpha[..., None]) + target * alpha[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_interior_environment_spill(
    original_image: np.ndarray,
    result_image: np.ndarray,
    opening_mask: np.ndarray,
    replacement_mask: np.ndarray,
    openings: list[tuple[int, int, int, int]],
    sky_layer: np.ndarray,
    weather: str,
    relight_intensity: float,
) -> np.ndarray:
    if np.count_nonzero(replacement_mask) == 0:
        return result_image

    sky_mean = _masked_lab_mean(sky_layer, replacement_mask)
    original_sky_mean = _masked_lab_mean(original_image, replacement_mask)
    if sky_mean is None or original_sky_mean is None:
        return result_image

    h, w = result_image.shape[:2]
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    delta = sky_mean - original_sky_mean
    delta[0] = float(np.clip(delta[0], -18.0, 14.0))
    delta[1] = float(np.clip(delta[1], -12.0, 16.0))
    delta[2] = float(np.clip(delta[2], -16.0, 24.0))

    source_mask = opening_mask if np.count_nonzero(opening_mask) > 0 else replacement_mask
    interior_gate = 1.0 - cv2.GaussianBlur((source_mask > 0).astype(np.float32), (0, 0), 2.4)
    spill = _normalized_blur(source_mask, sigma=max(h, w) * 0.105)
    beam = _build_floor_beam_mask((h, w), openings) if openings else np.zeros((h, w), dtype=np.float32)
    floor_gate = np.clip((yy - h * 0.50) / max(h * 0.50, 1.0), 0.0, 1.0)

    influence = spill * 0.11 + beam * floor_gate * 0.08
    if weather in {"sunrise", "sunset"}:
        influence += beam * floor_gate * 0.06
    elif weather in DIFFUSE_CLOUD_WEATHERS:
        influence = spill * 0.08 + beam * floor_gate * 0.03
    elif weather == "clear":
        influence = spill * 0.045 + beam * floor_gate * 0.035

    influence = cv2.GaussianBlur((influence * interior_gate).astype(np.float32), (0, 0), 7.0)
    influence = np.clip(influence * relight_intensity * _resolve_sky_environment_match_intensity(), 0.0, 0.16)
    if float(influence.max()) <= 1e-4:
        return result_image

    result_lab = cv2.cvtColor(result_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    channel_strength = np.array([0.18, 0.36, 0.48], dtype=np.float32)
    if weather == "clear":
        channel_strength = np.array([0.10, 0.18, 0.26], dtype=np.float32)
    elif weather in DIFFUSE_CLOUD_WEATHERS:
        channel_strength = np.array([0.24, 0.20, 0.28], dtype=np.float32)

    for channel_index in range(3):
        result_lab[:, :, channel_index] = np.clip(
            result_lab[:, :, channel_index]
            + influence * delta[channel_index] * channel_strength[channel_index],
            0.0,
            255.0,
        )

    return cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _apply_ground_sunlight_harmony(
    original_image: np.ndarray,
    result_image: np.ndarray,
    openings: list[tuple[int, int, int, int]],
    weather: str,
    reference_image: np.ndarray | None = None,
) -> np.ndarray:
    if weather not in PROJECTED_GROUND_LIGHT_WEATHERS or not openings:
        return result_image

    projection = _build_projected_ground_light(
        original_image,
        openings,
        weather,
        reference_image=reference_image if reference_image is not None else result_image,
    )
    if float(projection.max()) <= 1e-4:
        return result_image

    h, w = result_image.shape[:2]
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    floor_gate = np.clip((yy - h * 0.52) / max(h * 0.42, 1.0), 0.0, 1.0)

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    floor_samples = gray[floor_gate > 0.10]
    if floor_samples.size:
        low = float(np.quantile(floor_samples, 0.38))
        high = float(np.quantile(floor_samples, 0.92))
    else:
        low, high = 0.34, 0.86
    bright_gate = np.clip((gray - low) / max(high - low, 0.06), 0.0, 1.0)

    local_blur = cv2.GaussianBlur(gray, (0, 0), max(8.0, min(h, w) * 0.010))
    shadow_detail = np.clip((local_blur - gray) / 0.18, 0.0, 1.0)
    highlight_detail = np.clip((gray - local_blur) / 0.16, 0.0, 1.0)

    intensity = _resolve_sky_ground_light_intensity()
    light_mask = projection * floor_gate * (0.32 + bright_gate * 0.58 + highlight_detail * 0.22)
    light_mask = cv2.GaussianBlur(light_mask.astype(np.float32), (0, 0), max(5.0, min(h, w) * 0.006))
    light_mask = np.clip(light_mask * intensity, 0.0, 0.38)

    shadow_mask = projection * floor_gate * shadow_detail
    shadow_mask = cv2.GaussianBlur(shadow_mask.astype(np.float32), (0, 0), max(3.0, min(h, w) * 0.004))
    shadow_mask = np.clip(shadow_mask * intensity * 0.18, 0.0, 0.10)

    result = result_image.astype(np.float32) / 255.0
    if weather == "sunrise":
        warm_color = np.array([0.50, 0.72, 1.00], dtype=np.float32)  # BGR
        lift_strength = 0.26
        color_strength = 0.42
    else:
        warm_color = np.array([0.48, 0.70, 1.00], dtype=np.float32)
        lift_strength = 0.24
        color_strength = 0.40

    result = result * (1.0 - shadow_mask[..., None])
    result = np.clip(result * (1.0 + light_mask[..., None] * lift_strength), 0.0, 1.0)
    result = result * (1.0 - light_mask[..., None] * color_strength) + warm_color * (
        light_mask[..., None] * color_strength
    )
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def _apply_night_ground_lighting(
    original_image: np.ndarray,
    result_image: np.ndarray,
    openings: list[tuple[int, int, int, int]],
    opening_mask: np.ndarray,
    reference_image: np.ndarray | None = None,
) -> np.ndarray:
    if not openings:
        return result_image

    cleaned = result_image.copy()

    h, w = cleaned.shape[:2]
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    floor_gate = np.clip((yy - h * 0.50) / max(h * 0.40, 1.0), 0.0, 1.0)
    opening_soft = cv2.GaussianBlur((opening_mask > 0).astype(np.float32), (0, 0), 3.2)
    interior_floor = np.clip(floor_gate * (1.0 - opening_soft * 0.40), 0.0, 1.0)

    hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB).astype(np.float32)
    if reference_image is not None and reference_image.shape[:2] == (h, w):
        ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        reference_floor = ref_lab[:, :, 0][floor_gate > 0.20]
        target_l = float(np.quantile(reference_floor, 0.58)) if reference_floor.size else 58.0
        reference_b = ref_lab[:, :, 2][floor_gate > 0.20]
        target_b = float(np.quantile(reference_b, 0.44)) if reference_b.size else 104.0
    else:
        target_l = 46.0
        target_b = 94.0
    target_l = float(np.clip(target_l, 28.0, 54.0))
    target_b = float(np.clip(target_b, 78.0, 100.0))

    l_channel = lab[:, :, 0]
    too_bright = np.clip((l_channel - target_l) / 56.0, 0.0, 1.0)
    moon_spill = _build_floor_beam_mask((h, w), openings)
    daylight_zone = cv2.GaussianBlur((moon_spill * interior_floor).astype(np.float32), (0, 0), max(22.0, min(h, w) * 0.018))
    tone_alpha = cv2.GaussianBlur((interior_floor * (too_bright * 0.82 + daylight_zone * 0.18)).astype(np.float32), (0, 0), 13.0)
    tone_alpha = np.clip(tone_alpha * 0.68, 0.0, 0.68)
    hard_daylight = cv2.GaussianBlur(np.clip(too_bright * daylight_zone * floor_gate * 1.35, 0.0, 1.0), (0, 0), 7.0)
    tone_alpha = np.maximum(tone_alpha, np.clip(hard_daylight * 0.86, 0.0, 0.86))
    daylight_support = np.clip(moon_spill * 0.58 + opening_soft * floor_gate * 0.70, 0.0, 1.0)
    pale_daylight = (
        (floor_gate > 0.18)
        & (saturation < 0.50)
        & (value > 0.36)
        & (l_channel > target_l + 5.0)
    ).astype(np.float32)
    opening_floor_highlight = (
        (opening_soft > 0.08)
        & (floor_gate > 0.20)
        & (value > 0.32)
        & (l_channel > target_l + 2.0)
    ).astype(np.float32)
    patch_seed = np.maximum(pale_daylight * daylight_support, opening_floor_highlight * floor_gate)
    patch_alpha = cv2.GaussianBlur(patch_seed.astype(np.float32), (0, 0), 8.0)
    patch_alpha = cv2.GaussianBlur(np.clip(patch_alpha * 1.75, 0.0, 1.0), (0, 0), 18.0)
    tone_alpha = np.maximum(tone_alpha, np.clip(patch_alpha * 0.96, 0.0, 0.96))
    original_patch_alpha = np.zeros((h, w), dtype=np.float32)
    if original_image is not None and original_image.shape[:2] == (h, w):
        original_hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        original_saturation = original_hsv[:, :, 1] / 255.0
        original_value = original_hsv[:, :, 2] / 255.0
        original_daylight = (
            (floor_gate > 0.16)
            & (original_value > 0.56)
            & (original_saturation < 0.42)
            & ((daylight_support > 0.04) | (opening_soft > 0.06))
        ).astype(np.float32)
        original_patch_alpha = cv2.GaussianBlur(original_daylight * floor_gate, (0, 0), 9.0)
        original_patch_alpha = cv2.GaussianBlur(np.clip(original_patch_alpha * 1.65, 0.0, 1.0), (0, 0), 22.0)
        tone_alpha = np.maximum(tone_alpha, np.clip(original_patch_alpha * 0.95, 0.0, 0.95))
    lab[:, :, 0] = l_channel * (1.0 - tone_alpha) + (target_l + (l_channel - target_l) * 0.08) * tone_alpha
    lab[:, :, 1] = lab[:, :, 1] * (1.0 - tone_alpha * 0.22) + 128.0 * (tone_alpha * 0.22)
    lab[:, :, 2] = lab[:, :, 2] * (1.0 - tone_alpha * 0.58) + target_b * (tone_alpha * 0.58)

    toned = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    local_glow = _build_opening_local_glow((h, w), openings)
    reflection = cv2.GaussianBlur(
        (moon_spill * 0.34 + local_glow * floor_gate * 0.12).astype(np.float32),
        (0, 0),
        max(16.0, min(h, w) * 0.014),
    )
    reflection = np.clip(reflection * interior_floor, 0.0, 0.07)
    cool_floor = np.array([0.82, 0.72, 0.44], dtype=np.float32)
    toned = np.clip(toned + (1.0 - toned) * cool_floor.reshape(1, 1, 3) * reflection[..., None] * 0.62, 0.0, 1.0)
    flatten_alpha = np.clip(np.maximum(patch_alpha, original_patch_alpha) * floor_gate * 0.42, 0.0, 0.42)
    if float(flatten_alpha.max()) > 1e-4:
        smooth_floor = cv2.GaussianBlur(toned, (0, 0), max(18.0, min(h, w) * 0.014))
        toned = toned * (1.0 - flatten_alpha[..., None]) + smooth_floor * flatten_alpha[..., None]
    return np.clip(toned * 255.0, 0, 255).astype(np.uint8)


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
    elif weather in DIFFUSE_CLOUD_WEATHERS:
        tone_mask = np.clip(room_spill * 0.12 + sun_glow * 0.06 + 0.06, 0.0, 0.20)
        channel_strength = np.array([0.10, 0.10, 0.14], dtype=np.float32)
    elif weather == "clear":
        tone_mask = np.clip(room_spill * 0.06 + beam_mask * 0.04 + floor_gate * 0.03 + 0.02, 0.0, 0.10)
        channel_strength = np.array([0.08, 0.05, 0.04], dtype=np.float32)
    elif weather == "night":
        tone_mask = np.clip(room_spill * 0.22 + floor_gate * 0.08 + 0.10, 0.0, 0.30)
        channel_strength = np.array([0.34, 0.18, 0.32], dtype=np.float32)
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
    elif weather in DIFFUSE_CLOUD_WEATHERS:
        guide = np.clip(room_spill * 0.18 + sun_glow * 0.10 + 0.06, 0.0, 0.24)
    elif weather == "clear":
        guide = np.clip(room_spill * 0.08 + beam_mask * 0.10 + floor_gate * 0.04 + 0.02, 0.0, 0.14)
    elif weather == "night":
        guide = np.clip(room_spill * 0.26 + floor_gate * 0.12 + 0.08, 0.0, 0.32)
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
    if weather == "night":
        global_falloff = np.clip((0.90 - yy / max(h, 1) * 0.12) * float(profile.get("exposure", 1.0)), 0.05, 0.35)
    else:
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
    elif weather == "night":
        tint_amount = np.clip((0.08 + room_spill * 0.18 + sky_halo * 0.05 + floor_gate * 0.035) * interior_gate, 0.0, 0.20)
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
    elif weather == "night":
        warm_overlay[:, :, 2] = 0.72
        warm_overlay[:, :, 1] = 0.72
        warm_overlay[:, :, 0] = 0.56
        direct_sun_mix = np.clip((room_spill * 0.03 + sky_halo * 0.015 + floor_gate * 0.012) * interior_gate, 0.0, 0.04)
    elif weather in DIFFUSE_CLOUD_WEATHERS:
        warm_overlay[:, :, 2] = 1.0
        warm_overlay[:, :, 1] = 0.94
        warm_overlay[:, :, 0] = 0.88
        direct_sun_mix = np.clip(sun_glow * 0.05 + sun_sweep_floor * 0.01, 0.0, 0.08)
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
    elif weather in DIFFUSE_CLOUD_WEATHERS:
        floor_warmth = np.clip(
            room_spill * floor_gate * 0.025
            + sun_glow * floor_gate * 0.012,
            0.0,
            0.06,
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
    elif weather == "night":
        night_grade = np.clip((0.68 + room_spill * 0.22 + floor_gate * 0.10) * interior_gate, 0.0, 0.92)
        lab[:, :, 0] = np.clip(lab[:, :, 0] - night_grade * 22.0, 0.0, 255.0)
        lab[:, :, 1] = np.clip(lab[:, :, 1] - night_grade * 1.2, 0.0, 255.0)
        lab[:, :, 2] = np.clip(lab[:, :, 2] - night_grade * 18.0, 0.0, 255.0)
    else:
        # For sunny/overcast, keep it very subtle to maintain white walls
        lab[:, :, 1] = np.clip(lab[:, :, 1] + room_warmth * 0.5, 0.0, 255.0)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + room_warmth * 0.8, 0.0, 255.0)
    relit = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    relit = _apply_reference_guided_lighting(relit, reference_image, sky_mask, openings, weather)
    relit = _suppress_warm_wall_artifacts(image, relit, openings, weather)
    if original_floor_cleanup_enabled:
        return relit
    if weather == "night":
        return relit
    return _suppress_floor_light_patches(relit, openings, weather=weather)


def _build_regenerative_sky_mask(
    input_path: Path,
    mask_path: Path,
    image_shape: tuple[int, int],
    source_image: np.ndarray | None = None,
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

    regeneration_mask = _build_sky_replacement_mask_for_image(source_image, sky_mask, opening_mask)
    if np.count_nonzero(regeneration_mask) == 0:
        return regeneration_mask

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



def _apply_night_room_darkening(
    result: np.ndarray,
    sky_mask: np.ndarray,
    opening_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Night room darkening matched to reference image.

    Key visual targets from reference:
      - Far walls/ceiling: ~15-18% of daytime brightness, cool blue-gray
      - Near-window walls/floor: ~35-45% brightness (moonlight proximity spill)
      - Bright floor patch directly in front of window (moonlight reflection)
      - Room retains texture — NOT pure black
      - Sky/opening area: UNTOUCHED
    """
    h, w = result.shape[:2]

    # ── 1. Build interior mask (everything that is NOT sky) ─────────────────
    sky_binary = (sky_mask > 0).astype(np.uint8) * 255
    
    # Check if this is a fallback rectangular mask (YOLO bounding box)
    # If it is a perfect rectangle, we don't want to protect the whole thing or it creates a 'box' artifact.
    x_b, y_b, w_b, h_b = cv2.boundingRect(sky_binary)
    is_fallback_rect = False
    if w_b > 0 and h_b > 0:
        rect_area = w_b * h_b
        mask_pixels = np.count_nonzero(sky_binary)
        if mask_pixels / rect_area > 0.98:
            is_fallback_rect = True

    if is_fallback_rect:
        # For fallback rectangles, use no dilation to avoid spreading the 'bright box'
        sky_guard = sky_binary
    else:
        sky_guard = cv2.dilate(
            sky_binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
    
    # We ignore opening_mask here because YOLO's fallback mask is a literal bounding box.
    # If we exempt it from darkening, it leaves a bright rectangular box artifact around the window!
    combined_guard = sky_guard

    interior_mask = (combined_guard == 0).astype(np.float32)
    blur_sigma = max(h, w) * 0.058 if is_fallback_rect else 8.0
    interior_mask = cv2.GaussianBlur(interior_mask, (0, 0), blur_sigma)
    interior_mask = np.clip(interior_mask * 1.08, 0.0, 1.0)

    # ── 2. Uniform Room Darkening ────────────────────────────────────────────
    # The reference image has uniform deep blue/cool lighting, without a massive glowing halo around the window.
    result_f = result.astype(np.float32) / 255.0

    # Apply uniform darkness. 
    # Increased to ~0.82 brightness to make the room significantly brighter based on feedback.
    dark = result_f * 0.82

    # Cool blue-gray tint (BGR): crush red, slight green reduction, boost blue
    dark[:, :, 2] = np.clip(dark[:, :, 2] * 0.68, 0.0, 1.0)   # red   ↓
    dark[:, :, 1] = np.clip(dark[:, :, 1] * 0.80, 0.0, 1.0)   # green ↓
    dark[:, :, 0] = np.clip(dark[:, :, 0] * 1.10, 0.0, 1.0)   # blue  ↑

    # Faint cool ambient so walls don't go pure black
    ambient = np.array([0.050, 0.038, 0.026], dtype=np.float32)  # BGR cool blue
    dark = np.clip(dark + ambient.reshape(1, 1, 3) * interior_mask[..., None], 0.0, 1.0)

    # ── 3. Blend: interior → darkened, sky/opening → untouched ──────────────
    blended = result_f * (1.0 - interior_mask[..., None]) + dark * interior_mask[..., None]
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


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
        source_image=image,
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
        # --- Night mode: brute-force darken the room interior after sky compositing ---
        if weather == "night":
            opening_mask = _build_mask_from_opening_polygons(
                image_shape=(h, w),
                opening_polygons=opening_polygons,
                opening_source_size=opening_source_size,
            )
            result = _apply_night_room_darkening(result, regeneration_mask, opening_mask)
        # ------------------------------------------------------------------------------
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
        inferred_opening_mask = _build_reference_opening_mask(image)
        inferred_area_ratio = np.count_nonzero(inferred_opening_mask) / float(max(h * w, 1))
        support_kernel_size = max(9, int(round(min(h, w) * 0.012))) | 1
        sky_support_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (support_kernel_size, support_kernel_size),
        )
        sky_support = cv2.dilate((sky_mask > 0).astype(np.uint8) * 255, sky_support_kernel, iterations=1)
        support_overlap = np.count_nonzero(cv2.bitwise_and(inferred_opening_mask, sky_support)) / float(
            max(np.count_nonzero(inferred_opening_mask), 1)
        )
        if inferred_area_ratio <= 0.08 and support_overlap >= 0.05:
            opening_mask = inferred_opening_mask
        elif np.count_nonzero(inferred_opening_mask) > 0:
            logging.info(
                "[FullSceneGenerator] Ignoring broad inferred opening mask coverage=%.2f%% overlap=%.2f%%",
                inferred_area_ratio * 100.0,
                support_overlap * 100.0,
            )

    replacement_mask = _build_sky_replacement_mask_for_image(image, sky_mask, opening_mask)
    lighting_mask = cv2.bitwise_or(opening_mask, replacement_mask)
    if np.count_nonzero(lighting_mask) == 0:
        lighting_mask = replacement_mask
    logging.info(
        "[FullSceneGenerator] Sky replacement mask coverage=%.2f%% lighting mask coverage=%.2f%%",
        (np.count_nonzero(replacement_mask) / float(max(h * w, 1))) * 100.0,
        (np.count_nonzero(lighting_mask) / float(max(h * w, 1))) * 100.0,
    )

    use_reference_sky_asset = weather not in {"golden_hour", "sunset"}
    sky_asset = None if weather == "night" or not use_reference_sky_asset else _build_reference_sky_asset(
        reference_image=reference_image,
        reference_sky_mask=reference_sky_mask,
        opening_mask=opening_mask,
        width=w,
        height=h,
        weather=weather,
    )
    using_reference_sky = sky_asset is not None
    if sky_asset is None:
        sky_asset = _load_sky_asset(weather, w, h, sky_assets_dir)
    if weather in {"golden_hour", "sunset"} and not using_reference_sky:
        sky_asset = _fit_sky_layer_per_replacement_component(sky_asset, replacement_mask)
    if weather == "clear":
        clear_replacement_mask = _build_clear_replacement_mask(image, sky_mask, opening_mask)
        if np.count_nonzero(clear_replacement_mask) > 0:
            replacement_mask = cv2.bitwise_or(replacement_mask, clear_replacement_mask)
            if np.count_nonzero(opening_mask) > 0:
                replacement_mask = cv2.bitwise_and(replacement_mask, (opening_mask > 0).astype(np.uint8) * 255)
    sky_asset = _harmonize_sky_layer_to_scene(
        original_image=image,
        sky_layer=sky_asset,
        replacement_mask=replacement_mask,
        opening_mask=opening_mask,
        weather=weather,
    )
    if weather == "clear":
        sky_asset = _enrich_clear_blue_sky_layer(sky_asset, replacement_mask)
        sky_asset = _add_clear_reference_sun_glow(
            sky_layer=sky_asset,
            reference_image=reference_image,
            reference_sky_mask=reference_sky_mask,
            replacement_mask=replacement_mask,
        )
    elif weather == "night":
        if not using_reference_sky:
            sky_asset = _add_night_moon(
                sky_layer=sky_asset,
                replacement_mask=replacement_mask,
            )
    sky_alpha = _build_sky_alpha(replacement_mask)
    openings = _build_opening_stats(opening_mask if np.count_nonzero(opening_mask) > 0 else replacement_mask)

    relit = _apply_room_relighting(image, lighting_mask, openings, weather, reference_image=reference_image)
    relight_intensity = _resolve_sky_relight_intensity()
    if weather == "night":
        relight_intensity = 1.0  # fully apply the night-darkened relit image; no blending with bright original
    result = np.clip(
        image.astype(np.float32) * (1.0 - relight_intensity)
        + relit.astype(np.float32) * relight_intensity,
        0,
        255,
    ).astype(np.uint8)
    result = _apply_sky(result, sky_asset, sky_alpha)
    if np.count_nonzero(opening_mask) > 0:
        if weather != "night":
            result = _restore_clear_non_sky_opening(image, result, opening_mask, replacement_mask)
        result = _apply_opening_environment_tone(
            original_image=image,
            result_image=result,
            opening_mask=opening_mask,
            replacement_mask=replacement_mask,
            sky_layer=sky_asset,
            weather=weather,
        )
        result = _restore_window_structure(
            original=image,
            result=result,
            opening_mask=opening_mask,
            replacement_mask=replacement_mask,
        )
        if weather == "night":
            result = _apply_night_exterior_view_tone(
                original_image=image,
                result_image=result,
                opening_mask=opening_mask,
                replacement_mask=replacement_mask,
            )
            result = _suppress_night_sky_fringe(
                result_image=result,
                replacement_mask=replacement_mask,
                opening_mask=opening_mask,
            )
    result = _apply_interior_environment_spill(
        original_image=image,
        result_image=result,
        opening_mask=opening_mask,
        replacement_mask=replacement_mask,
        openings=openings,
        sky_layer=sky_asset,
        weather=weather,
        relight_intensity=relight_intensity,
    )
    if weather == "night":
        result = _apply_night_ground_lighting(
            original_image=image,
            result_image=result,
            openings=openings,
            opening_mask=opening_mask,
            reference_image=reference_image,
        )
    else:
        result = _apply_ground_sunlight_harmony(
            original_image=image,
            result_image=result,
            openings=openings,
            weather=weather,
            reference_image=reference_image,
        )
    # Night: brute-force darken the room interior as a final pass
    if weather == "night":
        result = _apply_night_room_darkening(result, replacement_mask, opening_mask)

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
