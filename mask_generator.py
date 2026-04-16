import cv2
import numpy as np
from pathlib import Path
import logging

try:
    from ultralytics import YOLO, SAM
except ImportError:
    pass


def _find_window_regions(img):
    """
    Use YOLO-World to detect windows and sliding glass doors.
    Returns list of bounding boxes [x1,y1,x2,y2].
    """
    try:
        yolo_model = YOLO("yolov8l-world.pt")
        # Broader classes to ensure detection in various lighting
        yolo_model.set_classes(["window", "sliding glass door", "glass door", "door", "glass", "balcony"])
        results = yolo_model(img, conf=0.005, iou=0.5, verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy()
        return bboxes
    except Exception as e:
        logging.warning(f"Window detection failed: {e}")
        return np.empty((0, 4))


def _detect_blue_sky_in_region(img, region_mask):
    """
    Detect strictly BLUE or BRIGHT sky pixels within a given region.
    Extremely relaxed for desaturated/overexposed skies.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Broad Sky Color Mask - Blue/Cyan/White/Gray (Very inclusive)
    # H: 70-170, S: 0-255, V: 100-255
    lower_sky = np.array([70, 0, 100]) 
    upper_sky = np.array([170, 255, 255])
    sky_color = cv2.inRange(hsv, lower_sky, upper_sky)

    # 2. Texture Mask - Essential to keep it "clean" (excludes railings)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = cv2.GaussianBlur(np.abs(laplacian), (5, 5), 0)
    # Relaxed variance (30.0) but still excludes sharp edges
    smooth_mask = (variance < 30.0).astype(np.uint8) * 255

    # Combine: Color + Smoothness
    sky_candidate = cv2.bitwise_and(sky_color, smooth_mask)

    # Restrict to the allowed region only
    sky_in_region = cv2.bitwise_and(sky_candidate, region_mask)

    # 3. Morphological cleanup for "smooth finish"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sky_in_region = cv2.morphologyEx(sky_in_region, cv2.MORPH_OPEN, kernel, iterations=1)
    sky_in_region = cv2.morphologyEx(sky_in_region, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Final smoothing
    sky_in_region = cv2.GaussianBlur(sky_in_region, (15, 15), 0)
    _, sky_in_region = cv2.threshold(sky_in_region, 128, 255, cv2.THRESH_BINARY)

    return sky_in_region


def _refine_with_sam(img, coarse_mask):
    """
    Use SAM with point prompts sampled from the coarse mask
    to get pixel-perfect sky boundaries.
    """
    try:
        ys, xs = np.where(coarse_mask > 0)
        if len(ys) == 0:
            return coarse_mask

        # Sample up to 5 well-spread points from the detected sky region
        n_points = min(5, len(ys))
        indices = np.linspace(0, len(ys) - 1, n_points, dtype=int)
        points = [[int(xs[i]), int(ys[i])] for i in indices]
        labels = [1] * n_points

        sam_model = SAM("sam_b.pt")
        sam_results = sam_model(img, points=points, labels=labels, verbose=False)[0]

        if sam_results.masks is None:
            return coarse_mask

        masks = sam_results.masks.data.cpu().numpy()
        combined = np.zeros(coarse_mask.shape, dtype=np.uint8)
        for m in masks:
            m = m.astype(np.uint8)
            if m.shape != coarse_mask.shape:
                m = cv2.resize(m, (coarse_mask.shape[1], coarse_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            combined = cv2.bitwise_or(combined, m * 255)

        return combined
    except Exception as e:
        logging.warning(f"SAM refinement failed: {e}")
        return coarse_mask


def generate_sky_mask(image_path: Path, output_mask_path: Path) -> bool:
    """
    Generates a binary mask isolating ONLY the sky visible through windows.
    For interior photos: sky is restricted to within detected window/door regions.
    For exterior photos: sky is detected in the upper portion of the image.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    orig_h, orig_w = img.shape[:2]

    # Work at manageable resolution
    max_dim = 1600
    scale = 1.0
    proc_img = img
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        proc_img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))

    proc_h, proc_w = proc_img.shape[:2]

    # ── Step 1: Detect window/door regions ──────────────────────────────
    logging.info("[MaskGen] Step 1: Detecting window/door regions...")
    window_bboxes = _find_window_regions(proc_img)

    if len(window_bboxes) > 0:
        # Build a mask of ONLY the window interiors
        window_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
        significant_windows = 0
        img_area = proc_h * proc_w
        
        for bbox in window_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            w_area = (x2 - x1) * (y2 - y1)
            # Only count as 'window' if it's reasonably large (> 0.5% of image)
            if w_area > img_area * 0.005: 
                window_mask[max(0, y1):min(proc_h, y2), max(0, x1):min(proc_w, x2)] = 255
                significant_windows += 1

        if significant_windows > 0:
            logging.info(f"[MaskGen] Step 2: Detecting blue sky within {significant_windows} window regions...")
            sky_mask = _detect_blue_sky_in_region(proc_img, window_mask)
        else:
            logging.info("[MaskGen] Only tiny window regions found — falling back to global detection.")
            # Fallback logic below...
            window_bboxes = [] # Reset to trigger exterior fallback

    if len(window_bboxes) == 0:
        # No windows found — likely an exterior image
        logging.info("[MaskGen] No windows detected — treating as exterior image.")
        # For exterior: detect blue sky in the upper 50% of the image
        upper_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
        upper_mask[0:int(proc_h * 0.5), :] = 255
        sky_mask = _detect_blue_sky_in_region(proc_img, upper_mask)

    # ── Step 3: Remove noise — keep only significant sky patches ────────
    min_area = proc_h * proc_w * 0.002
    contours, _ = cv2.findContours(sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(sky_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    sky_pixels = np.count_nonzero(clean_mask)
    if sky_pixels == 0:
        logging.warning("[MaskGen] No sky pixels found.")
        cv2.imwrite(str(output_mask_path), clean_mask)
        return False

    # ── Step 4: Refine edges with SAM ───────────────────────────────────
    logging.info(f"[MaskGen] Step 3: Refining {sky_pixels} sky pixels with SAM...")
    refined = _refine_with_sam(proc_img, clean_mask)

    # CRITICAL: Re-intersect with window regions so SAM doesn't leak onto walls
    if len(window_bboxes) > 0:
        refined = cv2.bitwise_and(refined, window_mask)

    # ── Step 5: Scale back to original resolution ───────────────────────
    if scale != 1.0:
        final_mask = cv2.resize(refined, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        final_mask = refined

    _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite(str(output_mask_path), final_mask)
    pct = np.count_nonzero(final_mask) / (orig_h * orig_w) * 100
    logging.info(f"[MaskGen] Sky mask saved — {pct:.1f}% of image is sky.")
    return pct > 0.1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_img = Path(r"uploads_temp\clustered\cluster_0\DSC04222_HDR.jpg")
    out_mask = Path("output_test_sky/test_mask.png")
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    ok = generate_sky_mask(test_img, out_mask)
    print(f"Result: {'Success' if ok else 'No sky found'}")
