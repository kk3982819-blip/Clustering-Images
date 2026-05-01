from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from light_source_estimator import LightSourceHypothesis


@dataclass(slots=True)
class RayFeatureBundle:
    brightness: np.ndarray
    warmth: np.ndarray
    saturation: np.ndarray
    local_contrast: np.ndarray
    haze: np.ndarray
    source_prior: np.ndarray
    floor_prior: np.ndarray
    sky_prior: np.ndarray
    radial_alignment: np.ndarray
    candidate_prior: np.ndarray


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def _normalized_blur(mask: np.ndarray, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    max_value = float(blurred.max())
    if max_value <= 1e-6:
        return np.zeros_like(blurred, dtype=np.float32)
    return blurred / max_value


def build_floor_prior(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lower_gate = np.clip((yy - h * 0.50) / max(h * 0.50, 1.0), 0.0, 1.0)
    center_bias = 1.0 - np.clip(np.abs(xx - w * 0.5) / max(w * 0.80, 1.0), 0.0, 1.0)
    center_bias = 0.70 + center_bias * 0.30
    return np.clip(lower_gate * center_bias, 0.0, 1.0).astype(np.float32)


def build_sky_prior(shape: tuple[int, int], sky_mask: np.ndarray) -> np.ndarray:
    h, w = shape
    top_gradient = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(h, 1)
    top_gradient = np.repeat(top_gradient, w, axis=1)
    if sky_mask is None or np.count_nonzero(sky_mask) == 0:
        return top_gradient

    sky_binary = (sky_mask > 0).astype(np.uint8) * 255
    sky_soft = _normalized_blur(sky_binary, sigma=max(6.0, min(h, w) * 0.02))
    return np.clip(sky_soft * 0.75 + top_gradient * 0.25, 0.0, 1.0)


def build_source_prior(shape: tuple[int, int], sources: list[LightSourceHypothesis]) -> np.ndarray:
    h, w = shape
    if not sources:
        return np.zeros((h, w), dtype=np.float32)

    prior = np.zeros((h, w), dtype=np.float32)
    for source in sources:
        sx = int(round(source.x))
        sy = int(round(source.y))
        if source.bbox is not None:
            x1, y1, x2, y2 = source.bbox
            opening_w = max(1, x2 - x1 + 1)
            opening_h = max(1, y2 - y1 + 1)
            top = int(np.clip(y2 - opening_h * 0.08, 0, h - 1))
            spread = max(30, int(opening_w * 0.75))
            polygon = np.array(
                [
                    [int(np.clip(x1 + opening_w * 0.12, 0, w - 1)), top],
                    [int(np.clip(x2 - opening_w * 0.12, 0, w - 1)), top],
                    [int(np.clip(x2 + spread, 0, w - 1)), h - 1],
                    [int(np.clip(x1 - spread * 0.45, 0, w - 1)), h - 1],
                ],
                dtype=np.int32,
            )
        else:
            top_width = max(14, int(min(h, w) * 0.03))
            spread = max(40, int((h - sy) * 0.42), int(w * 0.20))
            polygon = np.array(
                [
                    [int(np.clip(sx - top_width, 0, w - 1)), int(np.clip(sy, 0, h - 1))],
                    [int(np.clip(sx + top_width, 0, w - 1)), int(np.clip(sy, 0, h - 1))],
                    [int(np.clip(sx + spread, 0, w - 1)), h - 1],
                    [int(np.clip(sx - spread, 0, w - 1)), h - 1],
                ],
                dtype=np.int32,
            )
        source_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillConvexPoly(source_mask, polygon, float(max(source.score, 0.15)))
        prior = np.maximum(prior, source_mask)

    return _normalized_blur(prior, sigma=max(12.0, min(h, w) * 0.025))


def build_radial_alignment_prior(image: np.ndarray, sources: list[LightSourceHypothesis]) -> np.ndarray:
    h, w = image.shape[:2]
    if not sources:
        return np.zeros((h, w), dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = normalize_map(np.sqrt(dx * dx + dy * dy))
    grad_angle = np.arctan2(dy, dx)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    alignment = np.zeros((h, w), dtype=np.float32)
    for source in sources:
        source_angle = np.arctan2(yy - source.y, xx - source.x)
        boundary_normal = source_angle + np.pi * 0.5
        angle_alignment = np.abs(np.cos(grad_angle - boundary_normal))
        vertical_gate = np.clip((yy - source.y) / max(h * 0.70, 1.0), 0.0, 1.0)
        source_response = angle_alignment * grad_mag * vertical_gate * float(max(source.score, 0.10))
        alignment = np.maximum(alignment, source_response.astype(np.float32))

    return _normalized_blur(alignment, sigma=max(4.0, min(h, w) * 0.01))


def extract_ray_features(
    image: np.ndarray,
    sources: list[LightSourceHypothesis],
    sky_mask: np.ndarray | None = None,
    use_floor_prior: bool = True,
) -> RayFeatureBundle:
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    l_channel = lab[:, :, 0]
    b_channel = lab[:, :, 2]
    saturation_channel = hsv[:, :, 1]
    blur_sigma = max(8.0, min(h, w) * 0.025)

    brightness = normalize_map(l_channel)
    warmth = normalize_map(b_channel)
    saturation = normalize_map(saturation_channel)
    local_contrast = normalize_map(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), blur_sigma)))
    haze = normalize_map(brightness * 0.58 + (1.0 - local_contrast) * 0.32 + (1.0 - saturation) * 0.10)

    source_prior = build_source_prior((h, w), sources)
    floor_prior = build_floor_prior((h, w)) if use_floor_prior else np.zeros((h, w), dtype=np.float32)
    sky_prior = build_sky_prior((h, w), sky_mask if sky_mask is not None else np.zeros((h, w), dtype=np.uint8))
    radial_alignment = build_radial_alignment_prior(image, sources)

    candidate_prior = normalize_map(
        source_prior * 0.34
        + haze * 0.24
        + brightness * 0.18
        + radial_alignment * 0.16
        + warmth * 0.08
    )

    return RayFeatureBundle(
        brightness=brightness,
        warmth=warmth,
        saturation=saturation,
        local_contrast=local_contrast,
        haze=haze,
        source_prior=source_prior,
        floor_prior=floor_prior,
        sky_prior=sky_prior,
        radial_alignment=radial_alignment,
        candidate_prior=candidate_prior,
    )
