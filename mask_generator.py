import logging
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import SAM, YOLO
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False

# --- Model Singletons for Performance ---
_CACHED_MODELS = {}

def _get_model(model_name: str):
    """Load model once and cache it in RAM to avoid re-loading overhead."""
    if model_name not in _CACHED_MODELS:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # For Apple Silicon
        if not torch.cuda.is_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        
        logging.info("[MaskGen] Loading %s on %s...", model_name, device)
        if "world" in model_name:
            m = YOLO(model_name)
            m.set_classes(["window", "sliding glass door", "glass door"])
        else:
            m = SAM(model_name)
        
        m.to(device)
        _CACHED_MODELS[model_name] = m
    return _CACHED_MODELS[model_name]



def _odd(value: int) -> int:
    return max(3, value | 1)


def _as_binary_mask(mask: np.ndarray | None, shape: tuple[int, int] | None = None) -> np.ndarray:
    if shape is not None:
        h, w = shape
    elif mask is not None:
        h, w = mask.shape[:2]
    else:
        raise ValueError("shape is required when mask is None")

    if mask is None:
        return np.zeros((h, w), dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def _component_masks(mask: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    binary = _as_binary_mask(mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    masks: list[np.ndarray] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = np.zeros_like(binary)
        component[labels == label] = 255
        masks.append(component)
    return masks


def _find_window_boxes(img: np.ndarray) -> np.ndarray:
    if not HAS_MODELS:
        return np.empty((0, 4), dtype=np.float32)

    try:
        # Switch from 'l' (Large) to 's' (Small) for much faster inference
        model = _get_model("yolov8s-world.pt")
        result = model(img, conf=0.01, iou=0.45, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)
        return result.boxes.xyxy.cpu().numpy()
    except Exception as exc:
        logging.warning("[MaskGen] Window detection failed: %s", exc)
        return np.empty((0, 4), dtype=np.float32)


def _segment_openings(img: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    if len(boxes) == 0:
        return []

    opening_masks: list[np.ndarray] = []

    if HAS_MODELS:
        try:
            sam_model = _get_model("sam_b.pt")
            results = sam_model(img, bboxes=boxes, verbose=False)[0]
            if results.masks is not None:
                for raw_mask in results.masks.data.cpu().numpy():
                    opening_masks.append(_as_binary_mask(raw_mask > 0.5, (h, w)))
        except Exception as exc:
            logging.warning("[MaskGen] Opening segmentation failed: %s", exc)

    if opening_masks:
        return opening_masks

    # Fall back to inset rectangles so the frame itself is not treated as sky.
    fallback_masks: list[np.ndarray] = []
    for box in boxes.astype(int):
        x1, y1, x2, y2 = box.tolist()
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        inset_x = max(2, int(bw * 0.04))
        inset_y = max(2, int(bh * 0.04))
        left = max(0, x1 + inset_x)
        top = max(0, y1 + inset_y)
        right = min(w, x2 - inset_x)
        bottom = min(h, y2 - inset_y)
        if right <= left or bottom <= top:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
        fallback_masks.append(mask)
    return fallback_masks


def _keep_top_connected_components(candidate: np.ndarray, opening_core: np.ndarray) -> np.ndarray:
    keep = np.zeros_like(candidate)
    if np.count_nonzero(candidate) == 0:
        return keep

    height = candidate.shape[0]
    opening_pixels = max(1, np.count_nonzero(opening_core))
    min_area = max(30, int(opening_pixels * 0.002))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)

    for label in range(1, component_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        ys, _ = np.where(labels == label)
        if ys.size == 0:
            continue

        top_y = int(ys.min())
        if top_y <= max(3, int(height * 0.28)):
            keep[labels == label] = 255

    return keep


def _filter_neutral_sky_components(
    neutral_mask: np.ndarray,
    blue_mask: np.ndarray,
    allowed_mask: np.ndarray,
) -> np.ndarray:
    """
    Keep white/gray cloud pixels only when they are supported by nearby blue sky.
    This prevents bright walls, trim, and window frames from being treated as sky.
    """
    neutral = _as_binary_mask(neutral_mask)
    blue = _as_binary_mask(blue_mask, neutral.shape[:2])
    allowed = _as_binary_mask(allowed_mask, neutral.shape[:2])

    if np.count_nonzero(neutral) == 0 or np.count_nonzero(blue) == 0:
        return np.zeros_like(neutral)

    h, w = neutral.shape[:2]
    min_dim = max(1, min(h, w))
    support_kernel_size = _odd(max(5, int(round(min_dim * 0.008))))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (support_kernel_size, support_kernel_size),
    )
    blue_support = cv2.dilate(blue, support_kernel, iterations=1)

    boundary_width = max(3, int(round(min_dim * 0.003)))
    allowed_dist = cv2.distanceTransform(allowed, cv2.DIST_L2, 3)
    allowed_boundary = ((allowed_dist > 0) & (allowed_dist <= boundary_width)).astype(np.uint8) * 255

    allowed_area = max(1, int(np.count_nonzero(allowed)))
    large_boundary_area = max(80, int(allowed_area * 0.0015))
    filtered = np.zeros_like(neutral)
    allowed_ys, allowed_xs = np.where(allowed > 0)
    if allowed_ys.size > 0 and allowed_xs.size > 0:
        allowed_x1, allowed_x2 = int(allowed_xs.min()), int(allowed_xs.max()) + 1
        allowed_y1, allowed_y2 = int(allowed_ys.min()), int(allowed_ys.max()) + 1
        scale_w = max(1, allowed_x2 - allowed_x1)
        scale_h = max(1, allowed_y2 - allowed_y1)
    else:
        allowed_x1, allowed_y1 = 0, 0
        scale_w, scale_h = w, h

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(neutral, connectivity=8)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 12:
            continue

        comp_x = int(stats[label, cv2.CC_STAT_LEFT])
        comp_y = int(stats[label, cv2.CC_STAT_TOP])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        fill_ratio = area / float(max(1, comp_w * comp_h))
        vertical_strip = (
            fill_ratio > 0.42
            and comp_h > scale_h * 0.30
            and comp_w < scale_w * 0.14
        )
        edge_strip = (
            fill_ratio > 0.42
            and comp_h > scale_h * 0.24
            and (
                comp_x <= allowed_x1 + scale_w * 0.04
                or comp_x + comp_w >= allowed_x1 + scale_w * 0.96
            )
        )
        top_strip = (
            fill_ratio > 0.45
            and comp_w > scale_w * 0.22
            and comp_h < scale_h * 0.08
            and comp_y <= allowed_y1 + scale_h * 0.08
        )
        if vertical_strip or edge_strip or top_strip:
            continue

        component = labels == label
        near_blue = int(np.count_nonzero(blue_support[component]))
        if near_blue == 0:
            continue

        near_ratio = near_blue / float(area)
        boundary_ratio = np.count_nonzero(allowed_boundary[component]) / float(area)

        if near_ratio < 0.015:
            continue
        if boundary_ratio > 0.02 and area > large_boundary_area and near_ratio < 0.50:
            continue

        filtered[component] = 255

    return filtered


def _find_sky_floor_y(
    roi_hsv: np.ndarray,
    roi_mask: np.ndarray,
    blue_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> int:
    """
    Find the first row where an opening changes from sky to exterior content.
    This keeps bright balcony floors, grass, trees, and distant ground from
    being accepted as white/hazy sky just because they touch blue pixels above.
    """
    h, w = roi_mask.shape[:2]
    if h < 24 or w < 24:
        return h

    valid = roi_mask > 0
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels < 100:
        return h

    hue, sat, val = cv2.split(roi_hsv)

    green_land = cv2.inRange(roi_hsv, np.array([25, 18, 25]), np.array([88, 255, 245]))
    brown_land = cv2.inRange(roi_hsv, np.array([8, 25, 25]), np.array([36, 255, 230]))
    dark_land = (((val < 105) & (sat > 18)).astype(np.uint8) * 255)
    exterior_content = cv2.bitwise_or(cv2.bitwise_or(green_land, brown_land), dark_land)

    blue = (blue_mask > 0).astype(np.uint8) * 255
    candidate = (candidate_mask > 0).astype(np.uint8) * 255

    scan_start = max(1, int(h * 0.26))
    scan_end = min(h, int(h * 0.94))
    band_h = max(8, int(round(h * 0.018)))
    step = max(3, band_h // 2)
    min_valid = max(20, int(w * band_h * 0.08))

    for y in range(scan_start, max(scan_start, scan_end - band_h + 1), step):
        band_valid = valid[y:y + band_h]
        valid_count = int(np.count_nonzero(band_valid))
        if valid_count < min_valid:
            continue

        band_content = exterior_content[y:y + band_h]
        band_blue = blue[y:y + band_h]
        land_ratio = np.count_nonzero(band_content[band_valid] > 0) / float(valid_count)
        blue_ratio = np.count_nonzero(band_blue[band_valid] > 0) / float(valid_count)

        next_y = min(h, y + band_h * 3)
        next_valid = valid[y:next_y]
        next_valid_count = int(np.count_nonzero(next_valid))
        if next_valid_count == 0:
            continue
        next_content = exterior_content[y:next_y]
        next_land_ratio = np.count_nonzero(next_content[next_valid] > 0) / float(next_valid_count)

        if land_ratio >= 0.20 and next_land_ratio >= 0.16 and blue_ratio < 0.48:
            return max(0, y - int(round(h * 0.022)))

    # Secondary guard: if real blue evidence ends and a large neutral area keeps
    # going below it, cut before that lower neutral fill. This catches overexposed
    # balcony/floor regions that are smooth and nearly white.
    blue_rows = []
    candidate_rows = []
    for y in range(h):
        row_valid = valid[y]
        valid_count = int(np.count_nonzero(row_valid))
        if valid_count < max(8, int(w * 0.04)):
            blue_rows.append(0.0)
            candidate_rows.append(0.0)
            continue
        blue_rows.append(np.count_nonzero(blue[y][row_valid] > 0) / float(valid_count))
        candidate_rows.append(np.count_nonzero(candidate[y][row_valid] > 0) / float(valid_count))

    blue_rows_arr = np.array(blue_rows, dtype=np.float32)
    candidate_rows_arr = np.array(candidate_rows, dtype=np.float32)
    strong_blue_rows = np.where(blue_rows_arr > 0.12)[0]
    if strong_blue_rows.size > max(4, int(h * 0.035)):
        bottom_blue = int(strong_blue_rows.max())
        lower_start = min(h, bottom_blue + max(10, int(h * 0.08)))
        if lower_start < h:
            lower_candidate = np.mean(candidate_rows_arr[lower_start:])
            lower_blue = np.mean(blue_rows_arr[lower_start:])
            if lower_candidate > 0.45 and lower_blue < 0.04:
                return min(h, bottom_blue + max(8, int(h * 0.045)))

    return h


def _extract_sky_from_opening(
    bgr_img: np.ndarray,
    hsv_img: np.ndarray,
    gray_img: np.ndarray,
    opening_mask: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(opening_mask)
    ys, xs = np.where(opening_mask > 0)
    if ys.size == 0 or xs.size == 0:
        return result

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    roi_mask = opening_mask[y1:y2, x1:x2]
    roi_bgr = bgr_img[y1:y2, x1:x2]
    roi_hsv = hsv_img[y1:y2, x1:x2]
    roi_gray = gray_img[y1:y2, x1:x2]
    roi_h, roi_w = roi_mask.shape[:2]

    # 1. Pixel-level Sky Candidates (Color + Texture)
    hsv_blue = cv2.inRange(roi_hsv, np.array([80, 18, 60]), np.array([155, 255, 255]))
    blue_channel, green_channel, red_channel = cv2.split(roi_bgr.astype(np.int16))
    blue_dominant = (
        (blue_channel > green_channel + 6)
        & (blue_channel > red_channel + 10)
        & (blue_channel > 105)
    ).astype(np.uint8) * 255
    is_blue = cv2.bitwise_and(hsv_blue, blue_dominant)
    is_bright = cv2.inRange(roi_hsv, np.array([0, 0, 188]), np.array([180, 70, 255]))
    is_haze = cv2.inRange(roi_hsv, np.array([65, 4, 125]), np.array([165, 55, 255]))
    neutral_sky = cv2.bitwise_or(is_bright, is_haze)

    # Texture-based smoothness
    laplacian = np.abs(cv2.Laplacian(roi_gray, cv2.CV_32F))
    texture = cv2.GaussianBlur(laplacian, (5, 5), 0)
    # Adaptive texture threshold: relaxed to allow more cloud texture
    v_channel = roi_hsv[:, :, 2].astype(np.float32) / 255.0
    adaptive_thresh = 45.0 + v_channel * 65.0 
    is_smooth = (texture < adaptive_thresh).astype(np.uint8) * 255

    sobel_x = np.abs(cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3))
    edge_mag = cv2.GaussianBlur(sobel_x + sobel_y, (5, 5), 0)
    edge_thresh = 55.0 + v_channel * 55.0
    low_edge = (edge_mag < edge_thresh).astype(np.uint8) * 255

    # 2. Reject obvious non-sky (Green, Dark, High-Edge density)
    green_reject = cv2.inRange(roi_hsv, np.array([25, 10, 10]), np.array([88, 255, 235]))
    dark_reject  = cv2.inRange(roi_hsv, np.array([0, 0, 0]),   np.array([180, 255, 95]))
    
    # Combined sky candidate. Blue pixels are direct evidence; neutral pixels
    # are only kept when they behave like clouds inside that blue region.
    sky_reject = cv2.bitwise_or(green_reject, dark_reject)
    blue_candidate = cv2.bitwise_and(is_blue, is_smooth)
    blue_candidate = cv2.bitwise_and(blue_candidate, low_edge)
    blue_candidate = cv2.bitwise_and(blue_candidate, cv2.bitwise_not(sky_reject))
    blue_candidate = cv2.bitwise_and(blue_candidate, roi_mask)

    neutral_candidate = cv2.bitwise_and(neutral_sky, is_smooth)
    neutral_candidate = cv2.bitwise_and(neutral_candidate, low_edge)
    neutral_candidate = cv2.bitwise_and(neutral_candidate, cv2.bitwise_not(sky_reject))
    neutral_candidate = cv2.bitwise_and(neutral_candidate, roi_mask)
    neutral_candidate = _filter_neutral_sky_components(neutral_candidate, blue_candidate, roi_mask)

    candidate = cv2.bitwise_or(blue_candidate, neutral_candidate)

    treeline_y = _find_treeline_y(roi_gray)
    sky_floor_y = _find_sky_floor_y(roi_hsv, roi_mask, blue_candidate, candidate)
    treeline_y = min(treeline_y, sky_floor_y)
    if treeline_y < roi_h:
        candidate[treeline_y:, :] = 0

    # 3. Geometric pruning: Sky usually starts from the top or is high up
    # We use a relaxed horizon scan but don't force a hard cutoff
    canny_edges = cv2.Canny(cv2.GaussianBlur(roi_gray, (5, 5), 0), 20, 60)
    
    # Fill small gaps but keep it tight
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, close_k, iterations=1)
    
    # Final check: Must be part of a significant component that is "upper" in the window
    n_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(refined, connectivity=8)
    final_sky_roi = np.zeros_like(refined)
    
    min_area = max(50, int(roi_h * roi_w * 0.005))
    for i in range(1, n_comp):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
            
        top_y = stats[i, cv2.CC_STAT_TOP]
        component = labels == i
        blue_support = np.count_nonzero(is_blue[component]) / float(area)

        # Reject weakly blue neutral patches that are low in the opening and
        # likely belong to balcony floors, walls, or interior reflections.
        if blue_support < 0.12 and top_y > roi_h * 0.65 and area < (roi_h * roi_w * 0.20):
            continue

        # Relaxed: If the component starts in the upper 85% or is at least 10% of window area
        if top_y < roi_h * 0.85 or area > (roi_h * roi_w * 0.10):
            final_sky_roi[component] = 255

    if np.count_nonzero(final_sky_roi) == 0:
        return result

    result[y1:y2, x1:x2] = final_sky_roi
    return result


def _extract_global_sky_fallback(hsv_img: np.ndarray, gray_img: np.ndarray) -> np.ndarray:
    h, w = gray_img.shape[:2]
    sky_blue = cv2.inRange(hsv_img, np.array([78, 18, 90]), np.array([140, 255, 255]))
    sky_white = cv2.inRange(hsv_img, np.array([0, 0, 192]), np.array([180, 60, 255]))
    green_reject = cv2.inRange(hsv_img, np.array([28, 20, 20]), np.array([88, 255, 220]))
    dark_reject = cv2.inRange(hsv_img, np.array([0, 0, 0]), np.array([180, 255, 85]))

    texture = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray_img, cv2.CV_32F)), (5, 5), 0)
    smooth_blue = (texture < 85.0).astype(np.uint8) * 255
    smooth_white = (texture < 30.0).astype(np.uint8) * 255

    blue_candidate = cv2.bitwise_and(sky_blue, smooth_blue)
    white_candidate = cv2.bitwise_and(sky_white, smooth_white)

    if np.count_nonzero(blue_candidate) > max(50, (h * w) // 300):
        support = cv2.dilate(
            blue_candidate,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(int(min(h, w) * 0.03)), _odd(int(min(h, w) * 0.03)))),
            iterations=1,
        )
        white_candidate = cv2.bitwise_and(white_candidate, support)

    candidate = cv2.bitwise_or(blue_candidate, white_candidate)
    candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(green_reject))
    candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(dark_reject))

    top_gate = np.zeros((h, w), dtype=np.uint8)
    top_gate[: max(1, int(h * 0.92)), :] = 255
    candidate = cv2.bitwise_and(candidate, top_gate)

    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cleanup_kernel, iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cleanup_kernel, iterations=1)

    keep = np.zeros_like(candidate)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    min_area = max(80, int(h * w * 0.00012))
    for label in range(1, component_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        ys, _ = np.where(labels == label)
        if ys.size == 0:
            continue

        top_y = int(ys.min())
        bottom_y = int(ys.max())
        if top_y <= int(h * 0.50) and bottom_y <= int(h * 0.90):
            keep[labels == label] = 255

    return keep


def _load_annotation_region_masks(
    image_path: Path,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clusters_path = Path("uploads_temp/clustered/clusters.json")
    if not clusters_path.exists():
        empty = np.zeros(shape, dtype=np.uint8)
        return empty, empty.copy(), empty.copy()

    try:
        data = json.loads(clusters_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("[MaskGen] Could not read clusters.json: %s", exc)
        empty = np.zeros(shape, dtype=np.uint8)
        return empty, empty.copy(), empty.copy()

    sky_mask = np.zeros(shape, dtype=np.uint8)
    opening_mask = np.zeros(shape, dtype=np.uint8)
    exclude_mask = np.zeros(shape, dtype=np.uint8)
    image_name = image_path.name

    for entries in data.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("image") != image_name:
                continue
            for ann in entry.get("annotations", []):
                boundary = ann.get("boundary") or []
                if len(boundary) < 3:
                    continue

                contour = np.array(boundary, dtype=np.int32).reshape((-1, 1, 2))
                labels = {str(label.get("label", "")).strip().lower() for label in ann.get("labels", [])}

                if labels & {"sky", "cloud"}:
                    cv2.drawContours(sky_mask, [contour], -1, 255, -1)
                if labels & {"window", "sliding glass door", "glass door", "door"}:
                    cv2.drawContours(opening_mask, [contour], -1, 255, -1)
                if labels & {
                    "balcony", "balcony railing", "floor", "left wall",
                    "right wall", "ceiling", "tree", "grass", "plant",
                    "railing", "fence", "wall", "ground",
                }:
                    cv2.drawContours(exclude_mask, [contour], -1, 255, -1)

    return sky_mask, opening_mask, exclude_mask

    # ── Post-annotation verification: remove non-sky pixels ──────────
    # Even if annotations say "sky", verify each pixel is actually sky-colored
    img = cv2.imread(str(image_path))
    if img is not None:
        sky_mask = _verify_sky_pixels(img, sky_mask)
    return sky_mask


def _refine_annotation_sky_mask(
    image_path: Path,
    sky_mask: np.ndarray,
    opening_mask: np.ndarray,
    exclude_mask: np.ndarray,
) -> np.ndarray:
    sky_mask = _as_binary_mask(sky_mask)
    if np.count_nonzero(sky_mask) == 0:
        return sky_mask

    opening_mask = _as_binary_mask(opening_mask, sky_mask.shape[:2])
    exclude_mask = _as_binary_mask(exclude_mask, sky_mask.shape[:2])

    if np.count_nonzero(opening_mask) > 0:
        sky_mask = cv2.bitwise_and(sky_mask, opening_mask)
    if np.count_nonzero(exclude_mask) > 0:
        sky_mask = cv2.bitwise_and(sky_mask, cv2.bitwise_not(exclude_mask))

    img = cv2.imread(str(image_path))
    if img is None:
        return sky_mask

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    hard_dark = ((gray < 48) | ((val < 62) & (sat > 12))).astype(np.uint8) * 255
    green_content = cv2.inRange(hsv, np.array([25, 22, 18]), np.array([88, 255, 245]))
    brown_content = cv2.inRange(hsv, np.array([8, 32, 18]), np.array([36, 255, 220]))
    non_sky_content = cv2.bitwise_or(hard_dark, cv2.bitwise_or(green_content, brown_content))

    sky_mask = cv2.bitwise_and(sky_mask, cv2.bitwise_not(non_sky_content))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, close_k, iterations=1)
    sky_mask = _fill_small_holes(sky_mask, max_hole_fraction=0.18)

    sky_mask = cv2.bitwise_and(sky_mask, cv2.bitwise_not(non_sky_content))
    if np.count_nonzero(exclude_mask) > 0:
        sky_mask = cv2.bitwise_and(sky_mask, cv2.bitwise_not(exclude_mask))
    verified = _verify_sky_pixels(img, sky_mask)
    if np.count_nonzero(verified) > 0:
        sky_mask = verified
    return sky_mask


def _load_annotation_sky_mask(image_path: Path, shape: tuple[int, int]) -> np.ndarray:
    sky_mask, opening_mask, exclude_mask = _load_annotation_region_masks(image_path, shape)
    return _refine_annotation_sky_mask(image_path, sky_mask, opening_mask, exclude_mask)


def _find_treeline_y(gray_roi: np.ndarray) -> int:
    """
    Use Canny edges to find the treeline/horizon boundary inside a window.
    Returns the Y coordinate below which content is NOT sky (trees, ground, etc).
    """
    h, w = gray_roi.shape[:2]
    if h < 20 or w < 20:
        return h

    # Adaptive Canny: higher contrast areas need higher thresholds
    mean_val = np.mean(gray_roi)
    std_val = np.std(gray_roi)
    low_t = max(10, int(mean_val - std_val))
    high_t = min(200, int(mean_val + std_val))

    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges = cv2.Canny(blurred, low_t, high_t)

    # Scan rows from top to bottom; find the first row with significant edge density
    # 0.12 threshold: catches treelines while ignoring some noise
    for y in range(int(h * 0.15), int(h * 0.90)):
        row_edge_count = np.count_nonzero(edges[y, :])
        if row_edge_count / max(w, 1) > 0.12:
            # Check a small band to ensure it's not a single stray line
            band_end = min(h, y + max(4, int(h * 0.03)))
            band_density = np.count_nonzero(edges[y:band_end, :]) / max((band_end - y) * w, 1)
            if band_density > 0.05:
                return y
    return h


def _verify_sky_pixels(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Strict pixel-level verification: remove any masked pixels that are NOT
    actually sky-colored (blue/bright white and smooth texture).
    Uses adaptive texture gating and structural rejection.
    """
    h, w = mask.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Color Segmentation - Broadened to avoid patches
    hsv_blue = cv2.inRange(hsv, np.array([80, 18, 60]), np.array([155, 255, 255]))
    blue_channel, green_channel, red_channel = cv2.split(img_bgr.astype(np.int16))
    blue_dominant = (
        (blue_channel > green_channel + 6)
        & (blue_channel > red_channel + 10)
        & (blue_channel > 105)
    ).astype(np.uint8) * 255
    is_blue = cv2.bitwise_and(hsv_blue, blue_dominant)
    is_bright = cv2.inRange(hsv, np.array([0, 0, 188]), np.array([180, 70, 255]))
    is_haze = cv2.inRange(hsv, np.array([65, 4, 125]), np.array([165, 55, 255]))
    neutral_sky = cv2.bitwise_or(is_bright, is_haze)

    # 2. Adaptive Texture Rejection
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    texture = cv2.GaussianBlur(laplacian, (5, 5), 0)
    v_f = hsv[:, :, 2].astype(np.float32) / 255.0
    # Brighter clouds can have more texture than clear blue sky
    adaptive_texture_thresh = 40.0 + v_f * 65.0
    is_smooth = (texture < adaptive_texture_thresh).astype(np.uint8) * 255

    # 3. Structural Rejection (Edges/Railings)
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edge_mag = cv2.GaussianBlur(sobel_x + sobel_y, (5, 5), 0)
    # Higher threshold for edges in bright areas
    adaptive_edge_thresh = 50.0 + v_f * 40.0
    no_edges = (edge_mag < adaptive_edge_thresh).astype(np.uint8) * 255

    # 4. Content Rejection
    # Dark rejection (railings, mullions): Relaxed to V < 65 to avoid eating into dark blue sky
    is_dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 65]))
    # Green/brown rejection: slightly tightened Hue range
    is_non_sky = cv2.inRange(hsv, np.array([25, 15, 10]), np.array([88, 255, 245]))
    
    # Combined verification. Do not allow generic bright, smooth surfaces to
    # pass unless nearby blue pixels support them as clouds/haze.
    reject = cv2.bitwise_or(is_dark, is_non_sky)
    blue_candidate = cv2.bitwise_and(is_blue, is_smooth)
    blue_candidate = cv2.bitwise_and(blue_candidate, no_edges)
    blue_candidate = cv2.bitwise_and(blue_candidate, cv2.bitwise_not(reject))
    blue_candidate = cv2.bitwise_and(blue_candidate, mask)

    neutral_candidate = cv2.bitwise_and(neutral_sky, is_smooth)
    neutral_candidate = cv2.bitwise_and(neutral_candidate, no_edges)
    neutral_candidate = cv2.bitwise_and(neutral_candidate, cv2.bitwise_not(reject))
    neutral_candidate = cv2.bitwise_and(neutral_candidate, mask)
    neutral_candidate = _filter_neutral_sky_components(neutral_candidate, blue_candidate, mask)

    verified = cv2.bitwise_or(blue_candidate, neutral_candidate)

    # 5. Local Treeline Gating
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    treeline_gate = np.zeros((h, w), dtype=np.uint8)
    for label in range(1, component_count):
        rx, ry, rw, rh = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        if rw < 5 or rh < 5:
            continue
        gray_roi = gray[ry:ry+rh, rx:rx+rw]
        tl_y = _find_treeline_y(gray_roi)
        treeline_gate[ry:ry + tl_y, rx:rx + rw] = 255

    if np.count_nonzero(treeline_gate) > 0:
        verified = cv2.bitwise_and(verified, treeline_gate)

    result = cv2.bitwise_and(mask, verified)

    # 6. Final Smoothing (Preserving Details but Filling Gaps)
    # Increased to 5x5 to ensure small patches/holes are filled
    cleanup = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, cleanup, iterations=1)
    
    return result


def _fill_small_holes(mask: np.ndarray, max_hole_fraction: float = 0.03) -> np.ndarray:
    """
    Fill only small internal holes (pin-holes, thin gaps) without expanding
    the outer boundary of the mask.  Uses flood-fill from the border to
    identify the background, then fills anything that ISN'T background and
    ISN'T already mask — but only if the hole is small enough.
    """
    h, w = mask.shape[:2]
    total_mask_area = max(1, np.count_nonzero(mask))
    max_hole_area = int(total_mask_area * max_hole_fraction)

    # Flood-fill from the image border to find the "outside"
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    inv = cv2.bitwise_not(padded)
    cv2.floodFill(inv, None, (0, 0), 0)
    holes = inv[1:-1, 1:-1]  # internal holes only

    # Only fill holes that are smaller than max_hole_area
    n_comp, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    filled = mask.copy()
    for lbl in range(1, n_comp):
        area = int(hole_stats[lbl, cv2.CC_STAT_AREA])
        hole_w = int(hole_stats[lbl, cv2.CC_STAT_WIDTH])
        hole_h = int(hole_stats[lbl, cv2.CC_STAT_HEIGHT])
        long_slot = (
            (hole_w > w * 0.03 and hole_h < h * 0.025)
            or (hole_h > h * 0.03 and hole_w < w * 0.025)
        )
        if area <= max_hole_area and not long_slot:
            filled[hole_labels == lbl] = 255
    return filled


def _remove_structural_false_sky(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Final mask cleanup for architectural false positives: gray trim, mullion
    edges, and blind-like striped regions that can pass color checks.
    """
    cleaned = (mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(cleaned) == 0:
        return cleaned

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    dull_gray = ((sat <= 8) & (val < 210)).astype(np.uint8) * 255
    cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(dull_gray))

    dist = cv2.distanceTransform((cleaned > 0).astype(np.uint8), cv2.DIST_L2, 3)
    weak_edge = ((dist > 0) & (dist <= 3.0) & (sat <= 12) & (val < 225))
    cleaned[weak_edge] = 0

    if np.count_nonzero(cleaned) == 0:
        return cleaned

    result = np.zeros_like(cleaned)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    image_area = img_bgr.shape[0] * img_bgr.shape[1]
    min_area = max(120, int(image_area * 0.00002))

    for label in range(1, component_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component_roi = labels[y:y + h, x:x + w] == label
        roi_gray = gray[y:y + h, x:x + w]
        roi_hsv = hsv[y:y + h, x:x + w]

        sobel_y = np.abs(cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3))
        horizontal_edges = (sobel_y > 35.0) & component_roi
        row_pixels = np.count_nonzero(component_roi, axis=1)
        valid_rows = row_pixels > max(5, w * 0.05)
        if np.any(valid_rows):
            row_edge_ratio = np.count_nonzero(horizontal_edges, axis=1) / np.maximum(1, row_pixels)
            stripe_score = float(np.mean(row_edge_ratio[valid_rows] > 0.08))
        else:
            stripe_score = 0.0

        mean_sat = float(np.mean(roi_hsv[:, :, 1][component_roi]))
        aspect = w / float(max(1, h))
        fill_ratio = area / float(max(1, w * h))
        blind_like = (
            (stripe_score > 0.28 and mean_sat < 45.0)
            or (stripe_score > 0.36 and aspect > 1.35)
        )
        border_strip = (
            mean_sat < 25.0
            and fill_ratio > 0.35
            and (
                (aspect > 5.0 and h < img_bgr.shape[0] * 0.025)
                or (aspect < 0.20 and h > img_bgr.shape[0] * 0.08)
            )
        )
        if blind_like or border_strip:
            continue

        result[labels == label] = 255

    if np.count_nonzero(result) == 0:
        return result

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, small_kernel, iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, small_kernel, iterations=1)
    return result


def generate_sky_mask(image_path: Path, output_mask_path: Path) -> bool:
    """
    Generate a strict sky mask that stays inside window openings and only keeps
    sky that is connected to the top of each opening.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    h, w = img.shape[:2]
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    annotation_sky, annotation_opening, _annotation_exclude = _load_annotation_region_masks(
        Path(image_path),
        (h, w),
    )
    annotation_mask = _refine_annotation_sky_mask(
        Path(image_path),
        annotation_sky,
        annotation_opening,
        _annotation_exclude,
    )
    auto_mask = np.zeros((h, w), dtype=np.uint8)

    annotated_openings = _component_masks(
        annotation_opening,
        min_area=max(120, int(h * w * 0.00002)),
    )

    # Always run YOLO to find any windows that weren't manually annotated.
    boxes = _find_window_boxes(img)
    opening_masks = annotated_openings + _segment_openings(img, boxes)

    # Build a containment boundary: union of all SAM window openings.
    # The final sky mask must NEVER extend outside this boundary.
    containment = np.zeros((h, w), dtype=np.uint8)
    for om in opening_masks:
        containment = cv2.bitwise_or(containment, om)

    for opening_mask in opening_masks:
        sky_region = _extract_sky_from_opening(img, hsv_img, gray_img, opening_mask)
        if np.count_nonzero(sky_region) > 0:
            auto_mask = cv2.bitwise_or(auto_mask, sky_region)

    fallback_mask = np.zeros((h, w), dtype=np.uint8)
    if np.count_nonzero(annotation_mask) == 0:
        auto_area = int(np.count_nonzero(auto_mask))
        min_useful_auto_area = max(1500, int(h * w * 0.0012))
        if auto_area < min_useful_auto_area:
            fallback_mask = _extract_global_sky_fallback(hsv_img, gray_img)
            if np.count_nonzero(fallback_mask) > auto_area:
                auto_mask = cv2.bitwise_or(auto_mask, fallback_mask)

    # Constrain only auto-detected sky to model-detected openings. Manually
    # annotated sky was already clipped to annotated openings and refined.
    if np.count_nonzero(containment) > 0 and np.count_nonzero(fallback_mask) == 0:
        auto_mask = cv2.bitwise_and(auto_mask, containment)

    final_mask = cv2.bitwise_or(annotation_mask, auto_mask)

    if np.count_nonzero(final_mask) == 0:
        cv2.imwrite(str(output_mask_path), final_mask)
        return False

    # ── Reject dark pixels (railing bars, mullions, frames) ──────────
    gray_gate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Stricter dark threshold (55 instead of 65) to avoid eating into the sky
    dark_struct = (gray_gate < 55).astype(np.uint8) * 255
    # No dilation, or very small, to keep edges sharp
    final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(dark_struct))
    final_mask = _remove_structural_false_sky(img, final_mask)

    # ── Remove small noise components ────────────────────────────────
    clean_mask = np.zeros_like(final_mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    min_area = max(150, int(h * w * 0.0004))
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == label] = 255

    # ── Conservative hole fill ───────────────────────────────────────
    # Fill internal pin-holes (< 5% of mask area)
    clean_mask = _fill_small_holes(clean_mask, max_hole_fraction=0.05)

    # Light smoothing: use 5x5 to ensure patches are closed
    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, smooth_k, iterations=1)

    cv2.imwrite(str(output_mask_path), clean_mask)
    pct = np.count_nonzero(clean_mask) / float(h * w) * 100.0
    logging.info("[MaskGen] Sky mask saved - %.2f%% of image", pct)
    return np.count_nonzero(clean_mask) > 0


def generate_object_mask(image_path, output_mask_path, points):
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if HAS_MODELS:
        try:
            sam_model = SAM("sam_b.pt")
            result = sam_model(img, points=points, labels=[1] * len(points), verbose=False)[0]
            if result.masks is not None:
                mask = (result.masks.data[0].cpu().numpy() * 255).astype(np.uint8)
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(output_mask_path), mask)
                return True
        except Exception:
            pass

    for px, py in points:
        cv2.circle(mask, (int(px), int(py)), 20, 255, -1)

    cv2.imwrite(str(output_mask_path), mask)
    return True


def generate_sunray_mask(image_path: Path, output_mask_path: Path, mode: str | None = None) -> bool:
    from sunray_pipeline import generate_sunray_mask as _generate_sunray_mask

    return _generate_sunray_mask(Path(image_path), Path(output_mask_path), mode=mode)
