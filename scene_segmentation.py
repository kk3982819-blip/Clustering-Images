import os
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO, SAM
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    exit(1)

OUTPUT_DIR = Path("output_scene_segmentation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The master list of EVERYTHING to detect from user request
PROMPTS = [
    # Structure
    "ceiling", "wall", "floor", "door", "window", "sliding glass door",

    # Outdoor
    "sky", "cloud", "tree", "grass", "balcony", "balcony railing", "plant",

    # Fixtures
    "ceiling light", "smoke detector", "air vent", "power outlet", "light switch",

    # Kitchen
    "kitchen cabinet", "countertop", "sink", "faucet", "kitchen island",

    # Appliances
    "refrigerator", "microwave", "oven", "stove", "dishwasher",

    # Furniture
    "sofa", "chair", "table", "bed", "wardrobe", "tv",

    # Bathroom
    "toilet", "bathtub", "shower", "mirror", "bathroom sink"
]

STRUCTURAL_CLASSES = {
    "ceiling", "wall", "floor",
    "door", "window", "sliding glass door",
    "balcony", "balcony railing",
    "kitchen cabinet", "countertop", "kitchen island",
    "sky", "cloud"
}
np.random.seed(101) # Seed for colors
COLORS = {p: np.random.randint(40, 255, 3).tolist() for p in PROMPTS}

def get_label_name(prompt_idx, bbox, img_width):
    # Post-processing heuristics
    name = PROMPTS[prompt_idx]
    if name == "wall":
        center_x = (bbox[0] + bbox[2]) / 2
        return "Left Wall" if center_x < (img_width / 2) else "Right Wall"
    if name in ["baseboard", "skirting board"]:
        return "Baseboard / Skirting"
    if name in ["refrigerator", "fridge"]:
        return "Refrigerator"
    if name in ["television", "tv"]:
        return "Television"
    return name.title()

def draw_styled_label(img, label, bbox, color):
    # Get top-left of bounding box
    x1, y1, x2, y2 = map(int, bbox)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Text box padding
    pad_x, pad_y = 4, 4
    
    # Draw background rectangle
    rect_x1 = max(0, x1)
    rect_y1 = max(0, y1 - text_height - (pad_y * 2))
    rect_x2 = rect_x1 + text_width + (pad_x * 2)
    rect_y2 = rect_y1 + text_height + (pad_y * 2)
    
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
    # Draw text
    text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
    cv2.putText(img, label, (rect_x1 + pad_x, rect_y2 - pad_y - baseline // 2), font, font_scale, text_color, thickness)

def process_image(image_path: Path):
    print(f"Processing {image_path.name}...")
    img = cv2.imread(str(image_path))
    if img is None:
        return
        
    orig_h, orig_w = img.shape[:2]
    max_dim = 1600 # Use higher res for small objects like light switches
    scale = 1.0
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))

    print("  -> Running high-detail detector (YOLO-World Large)...")
    # Using 'l' or 'x' is much better for tiny objects
    yolo_model = YOLO("yolov8l-world.pt")
    yolo_model.set_classes(PROMPTS)
    
    # We use very low confidence (0.01) to pick up everything (tiny details and tough backgrounds).
    # We increase IOU to 0.75 in YOLO to let it find nested elements. We will manually filter synonyms!
    results = yolo_model(img, conf=0.01, iou=0.75, verbose=False)[0]
    
    raw_bboxes = results.boxes.xyxy.cpu().numpy()
    raw_class_ids = results.boxes.cls.cpu().numpy().astype(int)
    raw_confscores = results.boxes.conf.cpu().numpy()

    # --- CUSTOM DEDUPLICATION & FILTERING ---
    # Because our keyword dictionary contains many synonyms ("tv", "television", "wall", "painted wall"),
    # we need to prevent drawing 4 identical overlapping boundaries for the exact same semantic object.
    bboxes = []
    class_ids = []
    confscores = []
    
    img_area = img.shape[0] * img.shape[1]
    
    # Process from highest confidence to lowest
    sorted_indices = np.argsort(raw_confscores)[::-1]
    
    for idx_sorted in sorted_indices:
        b1 = raw_bboxes[idx_sorted]
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        
        # 1. Skip ambient boxes ("room interior") that cover essentially the entire image and just make it messy
        if (a1 / img_area) > 0.92:
            continue
            
        duplicate = False
        # 2. Check overlap heavily against already-accepted higher-confidence boxes
        for fidx, b2 in enumerate(bboxes):
            a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            
            xx1 = max(b1[0], b2[0])
            yy1 = max(b1[1], b2[1])
            xx2 = min(b1[2], b2[2])
            yy2 = min(b1[3], b2[3])
            
            inter_w = max(0, xx2 - xx1)
            inter_h = max(0, yy2 - yy1)
            inter_a = inter_w * inter_h
            
            if inter_a > 0:
                iou = inter_a / float(a1 + a2 - inter_a)
                # If these boxes overlap by >65%, they are likely synonyms aiming at the exact same object.
                if iou > 0.65:
                    duplicate = True
                    break
        
        if not duplicate:
            bboxes.append(b1)
            class_ids.append(raw_class_ids[idx_sorted])
            confscores.append(raw_confscores[idx_sorted])
            
    bboxes = np.array(bboxes) if len(bboxes) > 0 else np.empty((0, 4))
    class_ids = np.array(class_ids)
    confscores = np.array(confscores)

    if len(bboxes) == 0:
        print("  -> No objects detected.")
        return

    print(f"  -> Detected {len(bboxes)} distinct elements (Filtered redundant synonyms). Finding boundaries with Segment Anything Base...")
    # use sam_b for better masks than mobile_sam
    sam_model = SAM("sam_b.pt") 
    sam_results = sam_model(img, bboxes=bboxes, verbose=False)[0]
    
    if sam_results.masks is None:
        return

    masks = sam_results.masks.data.cpu().numpy()
    annotated_img = img.copy()
    
    for i in range(len(bboxes)):
        class_id = class_ids[i]
        prompt_name = PROMPTS[class_id]
        label = get_label_name(class_id, bboxes[i], img.shape[1])
        color = COLORS[prompt_name]
        
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # smoothing pass to remove noise and jagged edges
        mask_u8 = (m * 255).astype(np.uint8)
        mask_u8 = cv2.medianBlur(mask_u8, 7)
            
        # Extract boundary contour
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        # Find the largest contour for this mask
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Apply polygonal approximation if it's a structural element (to get perfectly straight lines)
        if prompt_name in STRUCTURAL_CLASSES:
            epsilon = 0.015 * cv2.arcLength(largest_contour, True) # 1.5% error margin for smoother architectural lines
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
        # Draw the main outline (thick)
        cv2.drawContours(annotated_img, [largest_contour], -1, color, 2)
        
        # Draw bounding box for context (thin outer edge)
        x1, y1, x2, y2 = map(int, bboxes[i])
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 1)
        
        # Draw floating text label
        draw_styled_label(annotated_img, label, bboxes[i], color)

    out_path = OUTPUT_DIR / f"{image_path.stem}_detailed.jpg"
    cv2.imwrite(str(out_path), annotated_img)
    print(f"  -> Saved to {out_path}!")

def main():
    input_dir = Path("input")
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    for img_path in image_files:
        process_image(img_path)

if __name__ == "__main__":
    main()
