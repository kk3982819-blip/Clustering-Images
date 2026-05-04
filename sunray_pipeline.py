from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from depth_prior import DepthPriorResult, estimate_depth_prior
from light_source_estimator import LightSourceEstimation, estimate_light_sources
from ray_feature_extractor import RayFeatureBundle, extract_ray_features, normalize_map


@dataclass(slots=True)
class SunrayPipelineConfig:
    mode: str = "pretrained_baseline"
    segformer_weights: str | None = None
    depth_model_path: str | None = None
    use_sam_refinement: bool = False
    debug: bool = True
    input_size: int = 512


@dataclass(slots=True)
class SunrayPipelineResult:
    success: bool
    mode_used: str
    binary_mask: np.ndarray
    probability_map: np.ndarray
    alpha_map: np.ndarray
    scene_type: str
    segformer_used: bool
    sam_refined: bool
    metadata: dict


_SEGFORMER_CACHE: dict[tuple[str, int], object] = {}


def _empty_result(shape: tuple[int, int], mode: str) -> SunrayPipelineResult:
    h, w = shape
    zeros_u8 = np.zeros((h, w), dtype=np.uint8)
    zeros_f = np.zeros((h, w), dtype=np.float32)
    return SunrayPipelineResult(
        success=False,
        mode_used=mode,
        binary_mask=zeros_u8,
        probability_map=zeros_f,
        alpha_map=zeros_f,
        scene_type="ambiguous",
        segformer_used=False,
        sam_refined=False,
        metadata={},
    )


def _resize_for_pipeline(image: np.ndarray, max_dim: int | None) -> tuple[np.ndarray, tuple[int, int]]:
    if max_dim is None or max_dim <= 0:
        return image, image.shape[:2]

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, (h, w)

    scale = max_dim / float(longest)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    return resized, (h, w)


def _build_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    color = np.zeros_like(overlay)
    color[:, :, 1] = 255
    color[:, :, 2] = 255
    alpha = (mask.astype(np.float32) / 255.0) * 0.45
    blended = overlay.astype(np.float32) * (1.0 - alpha[..., None]) + color.astype(np.float32) * alpha[..., None]
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.clip(blended, 0, 255).astype(np.uint8)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 2)
    return result


def _write_float_map(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(values, 0.0, 1.0)
    cv2.imwrite(str(path), (image * 255.0).astype(np.uint8))


def _filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def _keep_components_touching_seed(
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    kept = np.zeros_like(candidate_mask)
    if np.count_nonzero(seed_mask) == 0:
        return kept

    seed_dilate = cv2.dilate(
        seed_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=1,
    )
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        if np.count_nonzero(seed_dilate[component]) > 0:
            kept[component] = 255
    return kept


def _select_detached_floor_spill_components(
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray,
    features: RayFeatureBundle,
    score_map: np.ndarray,
    min_area: int,
) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    kept = np.zeros_like(candidate_mask)
    if np.count_nonzero(seed_mask) == 0:
        return kept

    h, w = candidate_mask.shape[:2]
    seed_binary = (seed_mask > 0).astype(np.uint8) * 255
    inverse_seed = cv2.bitwise_not(seed_binary)
    seed_distance = cv2.distanceTransform(inverse_seed, cv2.DIST_L2, 5)
    corridor_mask = (features.source_prior > 0.055).astype(np.uint8) * 255
    low_texture = 1.0 - features.local_contrast
    max_detached_distance = max(140.0, min(h, w) * 0.42)

    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component = labels == label
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        center_y = y + height * 0.5
        aspect_ratio = width / float(max(height, 1))
        if center_y < h * 0.44 or aspect_ratio < 0.60:
            continue

        min_seed_distance = float(seed_distance[component].min())
        if min_seed_distance > max_detached_distance:
            continue

        corridor_ratio = np.count_nonzero(corridor_mask[component]) / float(max(area, 1))
        source_mean = float(features.source_prior[component].mean())
        floor_mean = float(features.floor_prior[component].mean())
        haze_mean = float(features.haze[component].mean())
        score_mean = float(score_map[component].mean())
        texture_mean = float(low_texture[component].mean())
        detached_score = (
            score_mean * 0.34
            + source_mean * 0.24
            + floor_mean * 0.18
            + haze_mean * 0.14
            + texture_mean * 0.10
        )

        if corridor_ratio >= 0.26 and source_mean >= 0.03 and floor_mean >= 0.10 and detached_score >= 0.135:
            kept[component] = 255

    return kept


def _binarize_probability(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability.astype(np.float32), 0.0, 1.0)
    if np.count_nonzero(clipped > 0) == 0:
        return np.zeros(clipped.shape, dtype=np.uint8)

    high_percentile = float(np.percentile(clipped, 92))
    threshold = float(np.clip(max(0.42, high_percentile * 0.72), 0.42, 0.68))
    binary = (clipped >= threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.GaussianBlur(binary, (0, 0), 1.5)
    binary = (binary > 64).astype(np.uint8) * 255
    h, w = binary.shape[:2]
    return _filter_small_components(binary, min_area=max(80, int(h * w * 0.00045)))


def _expand_indoor_floor_extent(
    image: np.ndarray,
    probability: np.ndarray,
    seed_mask: np.ndarray,
    features: RayFeatureBundle,
) -> tuple[np.ndarray, np.ndarray]:
    if np.count_nonzero(seed_mask) == 0:
        return seed_mask, probability

    h, w = seed_mask.shape[:2]
    floor_gate = features.floor_prior > 0.10
    primary_source_gate = features.source_prior > 0.03
    spill_source_gate = features.source_prior > 0.012
    if not np.any(floor_gate & spill_source_gate):
        return seed_mask, probability

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0] / 255.0
    l_coarse = cv2.GaussianBlur(l_channel, (0, 0), max(7.0, min(h, w) * 0.018))
    low_texture = 1.0 - features.local_contrast

    seed_binary = (seed_mask > 0).astype(np.uint8) * 255
    dilated_seed = cv2.dilate(
        seed_binary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1,
    )
    reference_zone = floor_gate & (features.source_prior < 0.09) & (dilated_seed == 0)
    if np.count_nonzero(reference_zone) < max(180, int(h * w * 0.0009)):
        reference_zone = floor_gate & (dilated_seed == 0)

    if np.count_nonzero(reference_zone) > 0:
        baseline_l = float(np.percentile(l_coarse[reference_zone], 58))
    else:
        baseline_l = float(np.percentile(l_coarse, 50))

    brightness_delta = np.clip(l_coarse - baseline_l, -0.25, 1.0)
    positive_delta = normalize_map(np.clip(brightness_delta, 0.0, None))
    seed_values = probability[seed_binary > 0]
    seed_floor_brightness = l_coarse[seed_binary > 0]
    seed_low_threshold = float(np.clip(np.percentile(seed_values, 20) * 0.58, 0.18, 0.36)) if seed_values.size else 0.22
    brightness_floor = float(np.percentile(seed_floor_brightness, 15) - 0.08) if seed_floor_brightness.size else baseline_l + 0.01

    soft_extent = normalize_map(
        probability * 0.32
        + features.source_prior * 0.18
        + features.floor_prior * 0.14
        + features.haze * 0.13
        + positive_delta * 0.11
        + low_texture * 0.07
        + features.warmth * 0.05
    )

    low_confidence_candidate = (
        floor_gate
        & primary_source_gate
        & (l_coarse >= brightness_floor)
        & (
            (probability >= seed_low_threshold)
            | (soft_extent >= 0.29)
            | ((features.haze >= 0.38) & (positive_delta >= 0.16))
        )
    ).astype(np.uint8) * 255

    low_confidence_candidate = cv2.morphologyEx(
        low_confidence_candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    )
    low_confidence_candidate = cv2.morphologyEx(
        low_confidence_candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )

    spill_candidate = (
        floor_gate
        & spill_source_gate
        & (l_coarse >= baseline_l + 0.006)
        & (
            (soft_extent >= 0.18)
            | ((low_texture >= 0.50) & (positive_delta >= 0.05))
            | (features.haze >= 0.34)
        )
    ).astype(np.uint8) * 255
    spill_candidate = cv2.morphologyEx(
        spill_candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    )
    spill_candidate = cv2.morphologyEx(
        spill_candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    expanded_mask = _keep_components_touching_seed(
        candidate_mask=low_confidence_candidate,
        seed_mask=seed_binary,
        min_area=max(120, int(h * w * 0.00055)),
    )
    detached_spill_mask = _select_detached_floor_spill_components(
        candidate_mask=spill_candidate,
        seed_mask=seed_binary,
        features=features,
        score_map=np.maximum(probability, soft_extent),
        min_area=max(90, int(h * w * 0.00038)),
    )
    expanded_mask = cv2.bitwise_or(expanded_mask, detached_spill_mask)
    if np.count_nonzero(expanded_mask) == 0:
        return seed_mask, probability

    expanded_mask = cv2.bitwise_or(expanded_mask, seed_binary)
    expanded_mask = cv2.GaussianBlur(expanded_mask, (0, 0), 2.0)
    expanded_mask = (expanded_mask > 40).astype(np.uint8) * 255
    expanded_mask = _filter_small_components(expanded_mask, min_area=max(120, int(h * w * 0.00055)))

    expanded_probability = probability.copy()
    expanded_pixels = expanded_mask > 0
    expanded_probability[expanded_pixels] = np.maximum(
        expanded_probability[expanded_pixels],
        np.clip(soft_extent[expanded_pixels] * 0.78, 0.0, 1.0),
    )
    return expanded_mask, normalize_map(expanded_probability)


def _heuristic_probability(
    features: RayFeatureBundle,
    depth_prior: DepthPriorResult,
    scene_type: str,
) -> np.ndarray:
    low_texture = 1.0 - features.local_contrast
    if scene_type == "indoor_opening":
        base = normalize_map(
            features.source_prior * 0.30
            + features.floor_prior * 0.20
            + features.brightness * 0.15
            + features.haze * 0.13
            + features.radial_alignment * 0.10
            + features.warmth * 0.07
            + depth_prior.far_confidence * 0.05
        )
        gate = (
            (features.source_prior > 0.12)
            & ((features.floor_prior > 0.15) | (features.haze > 0.44))
            & (features.brightness > 0.46)
        ).astype(np.float32)
    elif scene_type == "outdoor_sky":
        base = normalize_map(
            features.source_prior * 0.24
            + features.sky_prior * 0.22
            + features.haze * 0.18
            + low_texture * 0.14
            + features.radial_alignment * 0.12
            + depth_prior.far_confidence * 0.10
        )
        gate = (
            (features.sky_prior > 0.18)
            & (features.source_prior > 0.08)
            & (features.haze > 0.35)
        ).astype(np.float32)
    else:
        base = normalize_map(
            features.candidate_prior * 0.48
            + features.sky_prior * 0.15
            + features.floor_prior * 0.10
            + low_texture * 0.10
            + depth_prior.far_confidence * 0.17
        )
        gate = ((features.source_prior > 0.08) | (features.haze > 0.48)).astype(np.float32)

    fused = np.clip(base * (0.62 + gate * 0.38), 0.0, 1.0)
    fused = cv2.GaussianBlur(fused.astype(np.float32), (0, 0), max(2.5, min(fused.shape[:2]) * 0.006))
    return normalize_map(fused)


def _load_segformer_bundle(weights_path: str, input_size: int):
    cache_key = (weights_path, input_size)
    if cache_key in _SEGFORMER_CACHE:
        return _SEGFORMER_CACHE[cache_key]

    from models.segformer_sunray import load_sunray_model

    bundle = load_sunray_model(weights_path=weights_path, input_size=input_size, prefer_transformer=True, local_files_only=True)
    _SEGFORMER_CACHE[cache_key] = bundle
    return bundle


def _predict_segformer_probability(image: np.ndarray, weights_path: str | None, input_size: int) -> np.ndarray | None:
    if not weights_path:
        return None
    candidate = Path(weights_path)
    if not candidate.exists():
        logging.info("[Sunray] SegFormer weights not found at %s; using heuristic pipeline.", weights_path)
        return None

    try:
        from models.segformer_sunray import predict_probability_map

        bundle = _load_segformer_bundle(str(candidate), input_size=input_size)
        return predict_probability_map(bundle, image)
    except Exception as exc:
        logging.warning("[Sunray] SegFormer inference failed; falling back to heuristics: %s", exc)
        return None


def _mask_components(mask: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = np.zeros_like(mask)
        component[labels == label] = 255
        components.append((component, (x, y, x + width, y + height)))
    return components


def _refine_with_sam(image: np.ndarray, coarse_mask: np.ndarray) -> tuple[np.ndarray, bool]:
    try:
        from mask_generator import HAS_MODELS, _get_model
    except Exception:
        return coarse_mask, False

    if not HAS_MODELS:
        return coarse_mask, False

    components = [
        (component, box)
        for component, box in _mask_components(coarse_mask)
        if np.count_nonzero(component) >= max(120, int(coarse_mask.size * 0.00035))
    ]
    if not components:
        return coarse_mask, False

    boxes = np.array([box for _, box in components], dtype=np.float32)
    try:
        sam_model = _get_model("sam_b.pt")
        results = sam_model(image, bboxes=boxes, verbose=False)[0]
    except Exception as exc:
        logging.info("[Sunray] SAM refinement unavailable: %s", exc)
        return coarse_mask, False

    if results.masks is None:
        return coarse_mask, False

    refined = np.zeros_like(coarse_mask)
    raw_masks = results.masks.data.cpu().numpy()
    for (component, _), raw_mask in zip(components, raw_masks, strict=False):
        candidate = (raw_mask > 0.5).astype(np.uint8) * 255
        if candidate.shape[:2] != coarse_mask.shape[:2]:
            candidate = cv2.resize(candidate, (coarse_mask.shape[1], coarse_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlap = np.count_nonzero((candidate > 0) & (component > 0))
        component_area = max(1, np.count_nonzero(component > 0))
        union = np.count_nonzero((candidate > 0) | (component > 0))
        overlap_ratio = overlap / float(component_area)
        iou = overlap / float(max(union, 1))
        if overlap_ratio >= 0.25 or iou >= 0.14:
            refined = cv2.bitwise_or(refined, candidate)

    if np.count_nonzero(refined) == 0:
        return coarse_mask, False

    refined = cv2.bitwise_or(refined, coarse_mask)
    refined = cv2.GaussianBlur(refined, (0, 0), 1.5)
    refined = (refined > 64).astype(np.uint8) * 255
    refined = _filter_small_components(refined, min_area=max(80, int(coarse_mask.size * 0.00045)))
    return refined, True


def _resolve_config(mode: str | None = None, debug: bool | None = None) -> SunrayPipelineConfig:
    mode_value = (mode or os.environ.get("SUNRAY_PIPELINE_MODE") or "pretrained_baseline").strip().lower()
    segformer_weights = os.environ.get("SUNRAY_SEGFORMER_WEIGHTS") or None
    depth_model_path = os.environ.get("SUNRAY_DEPTH_MODEL_PATH") or None
    use_sam = mode_value == "segformer_fused_sam" or os.environ.get("SUNRAY_USE_SAM_REFINEMENT", "").strip() in {"1", "true", "yes"}
    if debug is None:
        debug_value = os.environ.get("SUNRAY_PIPELINE_DEBUG", "").strip().lower() not in {"0", "false", "no"}
    else:
        debug_value = bool(debug)
    return SunrayPipelineConfig(
        mode=mode_value,
        segformer_weights=segformer_weights,
        depth_model_path=depth_model_path,
        use_sam_refinement=use_sam,
        debug=debug_value,
    )


def _write_artifacts(
    image: np.ndarray,
    output_mask_path: Path,
    result: SunrayPipelineResult,
    features: RayFeatureBundle | None,
    estimation: LightSourceEstimation | None,
    depth_prior: DepthPriorResult | None,
) -> None:
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_mask_path), result.binary_mask)

    debug_prefix = output_mask_path.with_suffix("")
    _write_float_map(debug_prefix.with_name(f"{debug_prefix.stem}_prob.png"), result.probability_map)
    _write_float_map(debug_prefix.with_name(f"{debug_prefix.stem}_alpha.png"), result.alpha_map)
    cv2.imwrite(str(debug_prefix.with_name(f"{debug_prefix.stem}_overlay.jpg")), _build_overlay(image, result.binary_mask))

    if features is not None:
        _write_float_map(debug_prefix.with_name(f"{debug_prefix.stem}_source_prior.png"), features.source_prior)
        _write_float_map(debug_prefix.with_name(f"{debug_prefix.stem}_candidate_prior.png"), features.candidate_prior)
        _write_float_map(debug_prefix.with_name(f"{debug_prefix.stem}_radial_alignment.png"), features.radial_alignment)
    if depth_prior is not None:
        _write_float_map(debug_prefix.with_name(f"{debug_prefix.stem}_depth.png"), depth_prior.depth_map)
    if estimation is not None and np.count_nonzero(estimation.sky_mask) > 0:
        cv2.imwrite(str(debug_prefix.with_name(f"{debug_prefix.stem}_sky_mask.png")), estimation.sky_mask)

    metadata_path = debug_prefix.with_name(f"{debug_prefix.stem}_metadata.json")
    metadata_path.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")


def run_sunray_pipeline(
    image_path: str | Path,
    output_mask_path: str | Path,
    mode: str | None = None,
    debug: bool | None = None,
    fast_light_sources: bool | None = None,
    processing_max_dim: int | None = None,
) -> SunrayPipelineResult:
    image_path = Path(image_path)
    output_mask_path = Path(output_mask_path)
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        return _empty_result((1, 1), mode or "pretrained_baseline")
    image, original_shape = _resize_for_pipeline(original_image, processing_max_dim)

    config = _resolve_config(mode, debug=debug)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    if config.mode == "legacy_floor_patch":
        from sunray_detector import generate_legacy_sunray_mask

        success = generate_legacy_sunray_mask(image_path, output_mask_path)
        binary_mask = cv2.imread(str(output_mask_path), cv2.IMREAD_GRAYSCALE)
        if binary_mask is None:
            binary_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        probability = (binary_mask > 0).astype(np.float32)
        return SunrayPipelineResult(
            success=success,
            mode_used="legacy_floor_patch",
            binary_mask=binary_mask,
            probability_map=probability,
            alpha_map=probability,
            scene_type="indoor_opening",
            segformer_used=False,
            sam_refined=False,
            metadata={"mode": "legacy_floor_patch", "success": bool(success)},
        )

    estimation = estimate_light_sources(
        image_path,
        image=image,
        workdir=output_mask_path.parent,
        fast_mode=fast_light_sources,
    )
    features = extract_ray_features(
        image=image,
        sources=estimation.sources,
        sky_mask=estimation.sky_mask,
        use_floor_prior=estimation.scene_type == "indoor_opening",
    )
    depth_prior = estimate_depth_prior(image, model_name_or_path=config.depth_model_path)
    heuristic_prob = _heuristic_probability(features, depth_prior, estimation.scene_type)

    segformer_prob = None
    segformer_used = False
    if config.mode in {"segformer_fused", "segformer_fused_sam"}:
        segformer_prob = _predict_segformer_probability(image, config.segformer_weights, config.input_size)
        segformer_used = segformer_prob is not None

    if segformer_prob is not None:
        probability = normalize_map(
            segformer_prob * 0.58
            + heuristic_prob * 0.26
            + features.source_prior * 0.08
            + depth_prior.far_confidence * 0.08
        )
    else:
        probability = heuristic_prob

    alpha_map = cv2.GaussianBlur(probability.astype(np.float32), (0, 0), max(1.4, min(image.shape[:2]) * 0.004))
    binary_mask = _binarize_probability(alpha_map)
    if estimation.scene_type == "indoor_opening" and np.count_nonzero(binary_mask) > 0:
        binary_mask, probability = _expand_indoor_floor_extent(
            image=image,
            probability=probability,
            seed_mask=binary_mask,
            features=features,
        )
        alpha_map = np.maximum(alpha_map, probability * (binary_mask > 0).astype(np.float32))
        alpha_map = normalize_map(alpha_map)
    sam_refined = False
    if config.use_sam_refinement and np.count_nonzero(binary_mask) > 0:
        binary_mask, sam_refined = _refine_with_sam(image, binary_mask)
        alpha_map = np.maximum(alpha_map, (binary_mask > 0).astype(np.float32) * 0.50)
        alpha_map = normalize_map(alpha_map)

    if np.count_nonzero(binary_mask) == 0:
        binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        alpha_map = np.zeros(image.shape[:2], dtype=np.float32)

    if binary_mask.shape[:2] != original_shape:
        binary_mask = cv2.resize(binary_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        probability = cv2.resize(probability.astype(np.float32), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha_map = cv2.resize(alpha_map.astype(np.float32), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        probability = normalize_map(np.clip(probability, 0.0, 1.0))
        alpha_map = normalize_map(np.clip(alpha_map, 0.0, 1.0))

    metadata = {
        "mode_requested": config.mode,
        "mode_used": config.mode,
        "scene_type": estimation.scene_type,
        "segformer_used": segformer_used,
        "sam_refined": sam_refined,
        "depth_provider": depth_prior.provider,
        "depth_model_name": depth_prior.model_name,
        "sources": [
            {
                "x": round(source.x, 2),
                "y": round(source.y, 2),
                "score": round(source.score, 4),
                "source_type": source.source_type,
                "bbox": list(source.bbox) if source.bbox else None,
            }
            for source in estimation.sources
        ],
        "mask_coverage_pct": round(float(np.count_nonzero(binary_mask)) / float(max(binary_mask.size, 1)) * 100.0, 4),
        "processing_shape": [int(image.shape[0]), int(image.shape[1])],
        "original_shape": [int(original_shape[0]), int(original_shape[1])],
    }

    result = SunrayPipelineResult(
        success=bool(np.count_nonzero(binary_mask) > 0),
        mode_used=config.mode,
        binary_mask=binary_mask,
        probability_map=probability.astype(np.float32),
        alpha_map=alpha_map.astype(np.float32),
        scene_type=estimation.scene_type,
        segformer_used=segformer_used,
        sam_refined=sam_refined,
        metadata=metadata,
    )

    if config.debug:
        _write_artifacts(original_image, output_mask_path, result, features, estimation, depth_prior)
    else:
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_mask_path), result.binary_mask)

    return result


def generate_sunray_mask(
    image_path: str | Path,
    output_mask_path: str | Path,
    mode: str | None = None,
    debug: bool | None = None,
    fast_light_sources: bool | None = None,
    processing_max_dim: int | None = None,
) -> bool:
    result = run_sunray_pipeline(
        image_path=image_path,
        output_mask_path=output_mask_path,
        mode=mode,
        debug=debug,
        fast_light_sources=fast_light_sources,
        processing_max_dim=processing_max_dim,
    )
    return result.success
