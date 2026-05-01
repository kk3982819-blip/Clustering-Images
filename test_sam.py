import cv2
import numpy as np
from ultralytics import SAM
from pathlib import Path

img_path = Path(r"c:\Users\Mohammed kaif M\OneDrive\Desktop\clustering images\uploads_temp\clustered\cluster_0\DSC04222_HDR.jpg")
img = cv2.imread(str(img_path))
orig_h, orig_w = img.shape[:2]

sam_model = SAM("sam_b.pt")
print("Running SAM with point...")
results = sam_model(img, points=[orig_w // 2, int(orig_h * 0.05)], labels=[1], verbose=False)[0]

if results.masks is not None:
    print(f"Success! Mask shape: {results.masks.data.shape}")
else:
    print("Failed to get mask from point.")
