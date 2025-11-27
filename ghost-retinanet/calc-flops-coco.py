import torch
from thop import profile
from ghostnet_faster_rcnn import GhostNetBackbone 

def measure_backbone_flops():
    model = GhostNetBackbone(width_mult=1.0)
    model.eval()
    
    # standard 224x224 as per paper
    input_tensor = torch.randn(1, 3, 224, 224)
    
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    
    print(f"GhostNet Backbone (1.0x) FLOPs: {flops / 1e6:.2f} M")
    print(f"GhostNet Backbone (1.0x) Params: {params / 1e6:.2f} M")
    
    print("\n--- Paper Comparison (Table 8) ---")
    print("MobileNetV2 Backbone FLOPs:  300 M ")
    print("MobileNetV3 Backbone FLOPs:  219 M ")
    print("GhostNet 1.1x Backbone FLOPs: 164 M ")

if __name__ == "__main__":
    measure_backbone_flops()
