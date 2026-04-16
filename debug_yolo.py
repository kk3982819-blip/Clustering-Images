from ultralytics import YOLO
import cv2
from pathlib import Path

img_path = Path(r"c:\Users\Mohammed kaif M\OneDrive\Desktop\clustering images\uploads_temp\clustered\cluster_0\DSC04222_HDR.jpg")
img = cv2.imread(str(img_path))

yolo_model = YOLO("yolov8l-world.pt")
yolo_model.set_classes(["sky", "cloud", "tree", "building"])
results = yolo_model(img, conf=0.01, iou=0.5, verbose=False)[0]

for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy(), results.boxes.conf.cpu().numpy()):
    name = yolo_model.names[int(cls)]
    print(f"Found {name} with conf {conf} at {box}")
