from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch is a runtime dependency in this repo
    torch = None
    F = None

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    HAS_DEPTH_TRANSFORMERS = True
except ImportError:
    AutoImageProcessor = None
    AutoModelForDepthEstimation = None
    HAS_DEPTH_TRANSFORMERS = False


_DEPTH_MODEL_CACHE: dict[tuple[str, str], tuple[object, object]] = {}


@dataclass(slots=True)
class DepthPriorResult:
    depth_map: np.ndarray
    far_confidence: np.ndarray
    provider: str
    model_name: str
    used_fallback: bool


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    if float(a.std()) <= 1e-6 or float(b.std()) <= 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _preferred_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_depth_model(model_name_or_path: str, device: str) -> tuple[object, object] | None:
    if not HAS_DEPTH_TRANSFORMERS or torch is None:
        return None

    cache_key = (model_name_or_path, device)
    if cache_key in _DEPTH_MODEL_CACHE:
        return _DEPTH_MODEL_CACHE[cache_key]

    try:
        processor = AutoImageProcessor.from_pretrained(model_name_or_path, local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(model_name_or_path, local_files_only=True)
        model.to(device)
        model.eval()
    except Exception as exc:
        logging.info("[Sunray] Depth model '%s' unavailable locally: %s", model_name_or_path, exc)
        return None

    _DEPTH_MODEL_CACHE[cache_key] = (processor, model)
    return processor, model


def _pseudo_depth_prior(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    blur_sigma = max(8.0, min(image.shape[:2]) * 0.025)
    brightness = _normalize_map(gray)
    local_contrast = _normalize_map(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), blur_sigma)))
    low_saturation = 1.0 - _normalize_map(hsv[:, :, 1])
    atmospheric = _normalize_map(brightness * 0.50 + (1.0 - local_contrast) * 0.32 + low_saturation * 0.18)
    return atmospheric


def estimate_depth_prior(
    image: np.ndarray,
    model_name_or_path: str | Path | None = None,
) -> DepthPriorResult:
    model_name = str(model_name_or_path) if model_name_or_path else "depth-anything-v2-fallback"
    atmospheric = _pseudo_depth_prior(image)

    if model_name_or_path is None:
        return DepthPriorResult(
            depth_map=atmospheric,
            far_confidence=atmospheric,
            provider="heuristic",
            model_name=model_name,
            used_fallback=True,
        )

    loaded = _load_depth_model(str(model_name_or_path), _preferred_device())
    if loaded is None or torch is None or F is None:
        return DepthPriorResult(
            depth_map=atmospheric,
            far_confidence=atmospheric,
            provider="heuristic",
            model_name=model_name,
            used_fallback=True,
        )

    processor, model = loaded
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        inputs = processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(_preferred_device())
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        predicted_depth = outputs.predicted_depth
        if predicted_depth.ndim == 3:
            predicted_depth = predicted_depth.unsqueeze(1)
        depth = F.interpolate(
            predicted_depth,
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().detach().cpu().numpy()
        depth_norm = _normalize_map(depth)
        inv_depth_norm = 1.0 - depth_norm
        if _corr(inv_depth_norm, atmospheric) > _corr(depth_norm, atmospheric):
            depth_norm = inv_depth_norm
        far_confidence = _normalize_map(depth_norm * 0.70 + atmospheric * 0.30)
        return DepthPriorResult(
            depth_map=depth_norm,
            far_confidence=far_confidence,
            provider="depth-anything-compatible",
            model_name=model_name,
            used_fallback=False,
        )
    except Exception as exc:
        logging.warning("[Sunray] Depth inference failed, falling back to heuristic prior: %s", exc)
        return DepthPriorResult(
            depth_map=atmospheric,
            far_confidence=atmospheric,
            provider="heuristic",
            model_name=model_name,
            used_fallback=True,
        )
