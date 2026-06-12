# models/backbone.py
import torch.nn as nn
from mambavision import create_model

class MambaVisionStudent(nn.Module):
    def __init__(self, target_dim=768):
        super().__init__()
        self.backbone = create_model('mamba_vision_T', pretrained=True, num_classes=0)
        self.intercepted_features = None
        self.target_dim = target_dim 
        
        def hook_fn(module, input, output):
            self.intercepted_features = output  
            
        if hasattr(self.backbone, 'norm'):
            self.backbone.norm.register_forward_hook(hook_fn)
        else:
            raise ValueError("❌ 找不到 backbone 的 'norm' 層！")

        self.proj = nn.Linear(640, target_dim)

    def forward(self, x):
        _ = self.backbone(x) 
        feats = self.intercepted_features
        B, C, H, W = feats.shape
        feats = feats.flatten(2).transpose(1, 2)
        feats = self.proj(feats)
        final_feats = feats.transpose(1, 2).reshape(B, self.target_dim, H, W)
        return final_feats