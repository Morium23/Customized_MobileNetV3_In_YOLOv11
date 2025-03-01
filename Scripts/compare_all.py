from ultralytics import YOLO
import torch
import time
# Load a sample image
from PIL import Image
import numpy as np


# Load normal YOLOv11 model
#model_default = YOLO('yolo11n.pt')  # Replace with the version you want to benchmark
model_default = YOLO("D:/Last_dance/runs/10epo_9classes_final_yolo/weights/best.pt")
model_custom = YOLO("D:/Last_dance/runs/10epo_9classes_final_normal/weights/best.pt")
model_custom2 = YOLO("D:/Last_dance/runs/10epo_custom_HardSwish/weights/best.pt")

image_path = 'D:/Last_dance/cica.jpg'  # Replace with your test image path
input_image = Image.open(image_path)

# Convert image to tensor (assuming input size 640x640 for YOLOv11)
input_tensor = np.array(input_image.resize((640, 640))).transpose(2, 0, 1)  # Resize to (3, 640, 640)
input_tensor = torch.tensor(input_tensor).unsqueeze(0).float() / 255.0  # Add batch dimension and normalize

# Benchmark default YOLOv8 model
start_time = time.time()
pred_default = model_default(input_tensor)
default_inference_time = time.time() - start_time

# Benchmark custom YOLOv8 model
start_time = time.time()
pred_custom = model_custom(input_tensor)
custom_inference_time = time.time() - start_time

# Benchmark custom YOLOv8 model
start_time = time.time()
pred_custom2 = model_custom2(input_tensor)
custom_inference_time2 = time.time() - start_time

# Print results
print(f"Default YOLOv11 Inference Time: {default_inference_time:.4f} seconds")
print(f"Custom YOLOv11 (Normal_MobileNetv3) Inference Time: {custom_inference_time:.4f} seconds")
print(f"Custom YOLOv11 (Custom_MobileNet) Inference Time: {custom_inference_time2:.4f} seconds")


# Number of inferences to test throughput
num_inferences = 200

# Throughput for default YOLOv11
start_time = time.time()
for _ in range(num_inferences):
    pred_default = model_default(input_tensor)
default_total_time = time.time() - start_time
default_fps = num_inferences / default_total_time

# Throughput for custom YOLOv11
start_time = time.time()
for _ in range(num_inferences):
    pred_custom = model_custom(input_tensor)
custom_total_time = time.time() - start_time
custom_fps = num_inferences / custom_total_time

# Throughput for custom YOLOv11
start_time = time.time()
for _ in range(num_inferences):
    pred_custom = model_custom2(input_tensor)
custom_total_time2 = time.time() - start_time
custom_fps2 = num_inferences / custom_total_time2

# Print results
print(f"Default YOLOv11 Throughput: {default_fps:.2f} FPS")
print(f"Custom YOLOv11 (Normal_Mobilenetv3) Throughput: {custom_fps:.2f} FPS")
print(f"Custom YOLOv11 (Custom_MobileNetv3) Throughput: {custom_fps2:.2f} FPS")
