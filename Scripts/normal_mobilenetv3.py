import torch
import torch.nn as nn
import time
from torchvision.models import mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights

class MobileNetV3Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3Backbone, self).__init__()
        # Load MobileNetV3
        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        
        # Extract the feature layers
        self.stage1 = nn.Sequential(*mobilenet.features[:3])  # Low-level features
        self.stage2 = nn.Sequential(*mobilenet.features[3:6])  # Mid-level features
        self.stage3 = nn.Sequential(*mobilenet.features[6:])   # High-level features

        # Adjust output channels to match YOLOv8's neck
        self.adjust1 = nn.Conv2d(24, 64, kernel_size=3, padding=1)  
        self.adjust2 = nn.Conv2d(40, 128, kernel_size=3, padding=1) 
        self.adjust3 = nn.Conv2d(960, 256, kernel_size=3, padding=1)


    def forward(self, x):

        start_time = time.time()  # Start time measurement
        # Extract features at different scales
        f1 = self.stage1(x)  # Large-scale feature map(56*56)
        f2 = self.stage2(f1)  # Medium-scale feature map (28*28)
        f3 = self.stage3(f2)  # Small-scale feature map(7*7)

        print(f"Before Adjustment:")
        print(f"  f1 shape: {f1.shape}")
        print(f"  f2 shape: {f2.shape}")
        print(f"  f3 shape: {f3.shape}")

         # Adjust feature map channels
        f1 = self.adjust1(f1)
        f2 = self.adjust2(f2)
        f3 = self.adjust3(f3)

        # Print feature maps after adjustment
        print(f"After Adjustment:")
        print(f"  f1 shape: {f1.shape}")
        print(f"  f2 shape: {f2.shape}")
        print(f"  f3 shape: {f3.shape}")
        
        end_time = time.time()  # End time measurement
        print(f"Total Execution Time: {end_time - start_time:.6f} seconds")

        return [f1, f2, f3]  # Return feature maps for YOLOv8 neck

# Instantiate the backbone
mobilenetv3_backbone = MobileNetV3Backbone(pretrained=True)
# Test the model with a dummy input
with torch.no_grad():
    sample_input = torch.randn(1, 3, 224, 224)  # (Batch=1, Channels=3, Height=224, Width=224)
    outputs = mobilenetv3_backbone(sample_input)