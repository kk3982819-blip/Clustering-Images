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
        # Only look for transparent glass openings to avoid appliance/wall leakage
        yolo_model.set_classes([
            "window", "sliding glass door", "glass door"
        ])
        results = yolo_model(img, conf=0.05, iou=0.45, verbose=False)[0]
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

    # 1. Broad Sky Color Mask
    # Blue/Cyan range
    lower_blue = np.array([70, 30, 80]) 
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # NEW: White/Overexposed range (High Value, Low Saturation)
    # Stricter saturation floor (S > 15) to avoid dead-grey/off-white walls
    lower_white = np.array([0, 15, 200]) 
    upper_white = np.array([180, 80, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    sky_color = cv2.bitwise_or(blue_mask, white_mask)

    # 2. Texture Mask - Essential to keep it "clean" (excludes railings/walls)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = cv2.GaussianBlur(np.abs(laplacian), (5, 5), 0)
    
    # Blue sky can be slightly more textured (clouds), but white walls must be VERY smooth
    # to be considered sky.
    smooth_blue = (variance < 60.0).astype(np.uint8) * 255
    smooth_white = (variance < 25.0).astype(np.uint8) * 255
    
    # Combine: Color + Smoothness (Per-type)
    sky_blue = cv2.bitwise_and(blue_mask, smooth_blue)
    sky_white = cv2.bitwise_and(white_mask, smooth_white)
    sky_candidate = cv2.bitwise_or(sky_blue, sky_white)

    # Restrict to the allowed region only
    sky_in_region = cv2.bitwise_and(sky_candidate, region_mask)

    # Remove small noise before blurring
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sky_in_region = cv2.morphologyEx(sky_in_region, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Tighter smoothing to prevent mask "bloating" into window frames
    # We use a smaller kernel and a higher threshold to keep it crisp
    sky_in_region = cv2.GaussianBlur(sky_in_region, (3, 3), 0)
    _, sky_in_region = cv2.threshold(sky_in_region, 160, 255, cv2.THRESH_BINARY)

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

        # Find contours to ensure we sample from ALL distinct windows
        contours, _ = cv2.findContours(coarse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        labels = []
        
        # Sort contours by area to process the largest ones first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for cnt in contours[:5]: # Take top 5 distinct sky regions
            if cv2.contourArea(cnt) < 100:
                continue
            
            # Use image moments to find the center of the contour
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Check if the centroid is actually inside the mask (contours can be C-shaped)
                if coarse_mask[cY, cX] > 0:
                    points.append([cX, cY])
                    labels.append(1)
        
        # If contour sampling didn't yield enough points, fallback to random sampling
        if len(points) < 2:
            n_points = min(5 - len(points), len(ys))
            if n_points > 0:
                indices = np.linspace(0, len(ys) - 1, n_points, dtype=int)
                for i in indices:
                    points.append([int(xs[i]), int(ys[i])])
                    labels.append(1)

        sam_model = SAM("sam_b.pt")
        combined = np.zeros(coarse_mask.shape, dtype=np.uint8)
        
        # Run SAM individually for each point so it treats disconnected windows as separate objects
        for p, l in zip(points, labels):
            sam_results = sam_model(img, points=[p], labels=[l], verbose=False)[0]
            if sam_results.masks is not None:
                masks = sam_results.masks.data.cpu().numpy()
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

    # ── Step 1b: Smart Fallback for missed windows ──────────────────────
    # Instead of raw color, look for "High Contrast" regions that look like 
    # sky-through-glass. Sky is usually much brighter than interior walls.
    if len(window_bboxes) == 0 or True: # Always check for supplemental regions
        hsv_full = cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)
        # Sky is usually the BRIGHTEST thing in interior photos
        bright_mask = (hsv_full[:, :, 2] > 220).astype(np.uint8) * 255
        
        # Must be blue OR extremely white (overexposed sky)
        # Blue/Cyan
        low_b = np.array([70, 15, 80])
        upp_b = np.array([145, 255, 255])
        # White - relaxed slightly to catch hazy sky
        low_w = np.array([0, 0, 220])
        upp_w = np.array([180, 50, 255])
        
        sky_col = cv2.bitwise_or(cv2.inRange(hsv_full, low_b, upp_b), cv2.inRange(hsv_full, low_w, upp_w))
        candidate = cv2.bitwise_and(bright_mask, sky_col)
        
        # ── VERTICAL BIAS: Sky is almost never on the floor ──────────────
        # Ignore anything in the bottom 35% of the frame
        candidate[int(proc_h * 0.65):, :] = 0
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
        
        cnts, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extra = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area > proc_h * proc_w * 0.001: # 0.1% of image
                # NEW: Strict check for narrow fallback strips
                # If it's a "ribbon" shape, it must have clear blue saturation
                # to distinguish from bright white walls/columns.
                x, y, w_box, h_box = cv2.boundingRect(c)
                ar = w_box / float(h_box) if h_box > 0 else 0
                is_narrow = ar < 0.25 or ar > 4.0
                if is_narrow:
                    # Sample saturation in this region
                    region_hsv = hsv_full[y:y+h_box, x:x+w_box]
                    avg_sat = np.mean(region_hsv[:, :, 1])
                    if avg_sat < 15: # Too grey to be sky
                        continue
                
                if 0.05 < ar < 5.0: # Catch narrow panes but only if they pass the sat check above
                    # Store the ACTUAL contour instead of a bbox
                    extra.append(c)
        
        if extra:
            # We mix bboxes (from YOLO) and contours (from fallback) in one list
            # The loop below handles both
            if len(window_bboxes) > 0:
                # Convert YOLO bboxes to a list so we can append contours
                wb_list = list(window_bboxes)
                wb_list.extend(extra)
                window_bboxes = wb_list
            else:
                window_bboxes = extra
            logging.info(f"[MaskGen] Supplemental fallback found {len(extra)} region(s).")

    if len(window_bboxes) > 0:
        # Build a mask of ONLY the window interiors
        window_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
        significant_windows = 0
        img_area = proc_h * proc_w
        
        for obj in window_bboxes:
            # Check if this is a bounding box (from YOLO) or a contour (from fallback)
            # YOLO bboxes are 1D arrays of length 4. Contours are 3D (N,1,2).
            is_bbox = False
            if isinstance(obj, (list, np.ndarray)):
                arr = np.array(obj)
                if arr.ndim == 1 and len(arr) == 4:
                    is_bbox = True
            
            if is_bbox:
                # YOLO Bounding Box
                x1, y1, x2, y2 = map(int, obj)
                w_area = (x2 - x1) * (y2 - y1)
                
                # Safety: ignore things clearly on the floor
                if y1 > proc_h * 0.85:
                    continue
                    
                if w_area > img_area * 0.0001: 
                    window_mask[max(0, y1):min(proc_h, y2), max(0, x1):min(proc_w, x2)] = 255
                    significant_windows += 1
            else:
                # It's a contour (from fallback)
                cnt = obj
                if cv2.contourArea(cnt) > img_area * 0.0001:
                    cv2.drawContours(window_mask, [cnt], -1, 255, -1)
                    significant_windows += 1

        if significant_windows > 0:
            logging.info(f"[MaskGen] Step 2: Detecting blue sky within {significant_windows} window regions...")
            sky_mask = _detect_blue_sky_in_region(proc_img, window_mask)
        else:
            logging.info("[MaskGen] Only tiny window regions found — falling back to global detection.")
            # Fallback logic below...
            window_bboxes = [] # Reset to trigger exterior fallback

    if len(window_bboxes) == 0:
        # No windows found — fallback to global detection ONLY if image is very bright overall
        # and has blue-sky signatures, otherwise assume it's a window-less room.
        avg_v = np.mean(cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)[:,:,2])
        if avg_v > 130:
            logging.info("[MaskGen] No windows detected — trying global detection (Exterior Mode).")
            upper_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
            upper_mask[0:int(proc_h * 0.55), :] = 255 # Up to 55% for exterior shots
            sky_mask = _detect_blue_sky_in_region(proc_img, upper_mask)
        else:
            logging.info("[MaskGen] No windows and dim image — assuming no sky visible.")
            sky_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)

    # ── Step 3: Remove noise — keep only significant sky patches ────────
    min_area = proc_h * proc_w * 0.0002 # 0.02% of image area (very sensitive)
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
    
    # NEW: Multi-point sampling for better coverage of large windows
    refined = np.zeros_like(clean_mask)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sam_model = SAM("sam_b.pt")
    
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        if cv2.contourArea(cnt) < 100: continue
        
        # Sample center + 4 quadrant points
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        
        x,y,w,h = cv2.boundingRect(cnt)
        candidate_points = [
            [cX, cY],
            [x + w//4, y + h//4],
            [x + 3*w//4, y + h//4],
            [x + w//4, y + 3*h//4],
            [x + 3*w//4, y + 3*h//4]
        ]
        
        valid_points = []
        for p in candidate_points:
            if cv2.pointPolygonTest(cnt, (float(p[0]), float(p[1])), False) >= 0:
                valid_points.append(p)
        
        if valid_points:
            labels = [1] * len(valid_points)
            res = sam_model(proc_img, points=valid_points, labels=labels, verbose=False)[0]
            if res.masks is not None:
                for m in res.masks.data.cpu().numpy():
                    m = (m > 0.5).astype(np.uint8) * 255
                    refined = cv2.bitwise_or(refined, m)
    
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


def generate_object_mask(image_path: Path, output_mask_path: Path, regions: list) -> bool:
    """
    Generates a binary mask for objects specified by points or bboxes.
    regions: List of [[x, y], ...] or [[x1, y1, x2, y2], ...]
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    orig_h, orig_w = img.shape[:2]
    sam_model = SAM("sam_b.pt")

    combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    for region in regions:
        points = []
        labels = []
        bboxes = None

        if len(region) == 2: # Point [x, y]
            points = [region]
            labels = [1]
        elif len(region) == 4: # BBox [x1, y1, x2, y2]
            bboxes = [region]
        
        results = sam_model(img, points=points, labels=labels, bboxes=bboxes, verbose=False)[0]

        if results.masks is not None:
            for m in results.masks.data.cpu().numpy():
                m = (m > 0.5).astype(np.uint8) * 255
                if m.shape != (orig_h, orig_w):
                    m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                combined_mask = cv2.bitwise_or(combined_mask, m)

    if np.count_nonzero(combined_mask) == 0:
        return False

    # ── Mask Expansion ────────────────────────────────────────────────
    kernel_size = int(max(orig_h, orig_w) * 0.005) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded_mask = cv2.dilate(combined_mask, kernel, iterations=2)

    cv2.imwrite(str(output_mask_path), expanded_mask)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_img = Path(r"uploads_temp\clustered\cluster_0\DSC04222_HDR.jpg")
    out_mask = Path("output_test_sky/test_mask.png")
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    ok = generate_sky_mask(test_img, out_mask)
    print(f"Result: {'Success' if ok else 'No sky found'}")
