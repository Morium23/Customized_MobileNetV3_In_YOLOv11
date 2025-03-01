import torch
from ultralytics import YOLO
import sys
import os
# Add the parent directory
sys.path.append(os.path.abspath("D:/Last_dance"))
from custom_mobilenetv3 import MobileNetV3Backbone
# Load the YOLOv11 model
model = YOLO("yolo11n.pt")  # Load YOLOv11 model architecture
# Replace the default backbone with MobileNetv3
model.model.backbone = MobileNetV3Backbone(pretrained=True)
#print model architecture
print(model.model.backbone)
# Train the model
model.train(
    data="D:/Last_dance/scripts/data.yaml",  # Dataset configuration
    epochs=10,
    imgsz=640,
    batch=16,
    workers = 1,
    device="cpu"  # GPU ID (use "cpu" for CPU training)
)