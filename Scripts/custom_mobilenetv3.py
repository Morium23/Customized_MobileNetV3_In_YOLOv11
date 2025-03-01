import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights

class MobileNetV3Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3Backbone, self).__init__()

        # Load MobileNetV3 with reduced width multiplier
        #width_mult = 0.75  # Reduce depth multiplier to decrease computations
        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        # Extract the feature layers with fewer layers for efficiency
        self.stage1 = nn.Sequential(*mobilenet.features[:3])  # Reduced low-level features
        self.stage2 = nn.Sequential(*mobilenet.features[3:5])  # Reduced mid-level features
        self.stage3 = nn.Sequential(*mobilenet.features[5:8])  # Reduced high-level features

        self.adjust1 = nn.Sequential(nn.Conv2d(24,64, kernel_size=1, groups=4),     # Pointwise grouped convolution
                                     nn.BatchNorm2d(64),                            # Normalize feature distribution
                                     nn.Hardswish(inplace=True))                         # Provide non-linearity
        self.adjust2 = nn.Sequential(nn.Conv2d(40,128, kernel_size=1, groups=8),    #40->128,10 channels & 32 filters per group
                                     nn.BatchNorm2d(128),
                                     nn.Hardswish(inplace=True))
        self.adjust3 = nn.Sequential(nn.Conv2d(80,256, kernel_size=1, groups=16),
                                     nn.BatchNorm2d(256),
                                     nn.Hardswish(inplace=True))


    def forward(self, x):
        # Extract features at different scales
        f1 = self.stage1(x)  # Large-scale feature map
        f2 = self.stage2(f1)  # Medium-scale feature map
        f3 = self.stage3(f2)  # Small-scale feature map

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
        
        return [f1, f2, f3]  # Return feature maps for YOLOv8 neck

# Instantiate the backbone
if __name__ == "__main__":
    model = MobileNetV3Backbone(pretrained=True)
    sample_input = torch.randn(1, 3, 224, 224)  # Reduced input resolution
    outputs = model(sample_input)
