# models/fewshot_matcher.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.rope_utils import compute_2d_sincos, apply_rotary_pos_emb

class RoPEVisualSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        grid_size = int(math.sqrt(N))
        H = W = grid_size
        num_extra_tokens = N - (H * W) 
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        sin, cos = compute_2d_sincos(H, W, self.head_dim, device=x.device)
        
        q_extra, q_spatial = q[:, :, :num_extra_tokens, :], q[:, :, num_extra_tokens:, :]
        k_extra, k_spatial = k[:, :, :num_extra_tokens, :], k[:, :, num_extra_tokens:, :]
        
        q_spatial_rot = apply_rotary_pos_emb(q_spatial, sin, cos)
        k_spatial_rot = apply_rotary_pos_emb(k_spatial, sin, cos)
        
        q_rotated = torch.cat([q_extra, q_spatial_rot], dim=2)
        k_rotated = torch.cat([k_extra, k_spatial_rot], dim=2)
        
        attn = (q_rotated @ k_rotated.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class URMAdapter(nn.Module):
    def __init__(self, visual_dim, text_dim=512, num_heads=8):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, visual_dim)
        
        self.visual_self_attn = RoPEVisualSelfAttention(visual_dim, num_heads)
        self.visual_norm = nn.LayerNorm(visual_dim)

        self.cross_attn = nn.MultiheadAttention(visual_dim, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(visual_dim)

    def forward(self, visual_patches, text_embeds):
        self_attn_output = self.visual_self_attn(visual_patches)
        enriched_patches = self.visual_norm(self_attn_output + visual_patches)
        
        Q = self.text_proj(text_embeds).unsqueeze(1) 
        K = enriched_patches
        V = enriched_patches
        
        cross_attn_output, _ = self.cross_attn(Q, K, V)
        fused_features = self.cross_norm(cross_attn_output + Q) 
        
        return fused_features.squeeze(1)

class URMFewShotMatcher(nn.Module):
    # 🌟 移除 backbone_name 參數，改為直接指定維度
    def __init__(self, visual_dim=768, text_dim=512):
        super().__init__()
        
        # 移除自我實例化的 CLIP 骨幹，因為特徵會由外部 (Mamba 或 Teacher) 傳入
        self.visual_dim = visual_dim
        
        # URM 跨模態大腦 (接收從 Mamba 映射過來的 768 維，以及 CLIP Text Encoder 的維度)
        self.adapter = URMAdapter(visual_dim=self.visual_dim, text_dim=text_dim)
        
        # 投影層：將 (Support CLS + URM 局部特徵) 降維回原本的視覺維度
        self.metric_proj = nn.Linear(self.visual_dim * 2, self.visual_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # 🌟 關鍵修改：將輸入從 images 改為 Mamba 萃取出的 4D 空間特徵
    def forward(self, support_feats_2d, query_feats_2d, text_embeddings, K_shot=5):
        # 輸入形狀預期為 (B, C, H, W)
        # 例如 MambaVision 轉換後為 (N * K_shot, 768, 14, 14) 和 (Q, 768, 14, 14)
        
        # 為了相容原本的 Transformer (B, L, C) 架構，將 4D (B, C, H, W) 展平為 3D
        B_supp, C, H, W = support_feats_2d.shape
        N = B_supp // K_shot
        
        supp_patches = support_feats_2d.flatten(2).transpose(1, 2) # [N*K, H*W, C]
        query_patches = query_feats_2d.flatten(2).transpose(1, 2)  # [Q, H*W, C]

        # 模擬原本 CLIP 的 CLS Token (用 Global Average Pooling 代替)
        supp_cls = supp_patches.mean(dim=1) # [N*K, C]
        query_cls = query_patches.mean(dim=1) # [Q, C]

        # --- 2. Query 處理：使用 GAP 特徵當作全局表示 ---
        query_g = query_cls 

        # --- 3. Support 處理 ---
        # 聚合 K-shot 的特徵 (求平均)
        # 先將 supp_patches 轉成 [N, K, H*W, C]，再對 K 维度求平均
        supp_patches = supp_patches.view(N, K_shot, supp_patches.size(1), -1).mean(dim=1) 
        supp_cls = supp_cls.view(N, K_shot, -1).mean(dim=1)

        # 透過 URMAdapter (2D RoPE) 找出專家字典中描述的病斑細節
        fused_local = self.adapter(supp_patches, text_embeddings) # [N, Dim]
        
        # 融合 Support 的「全局畫面」與「局部病斑」
        fused_prototypes = torch.cat([supp_cls, fused_local], dim=-1) # [N, 2 * Dim]
        fused_prototypes = self.metric_proj(fused_prototypes)            # 降回 [N, Dim]
        
        # --- 4. 餘弦相似度測量 ---
        fused_prototypes = F.normalize(fused_prototypes, p=2, dim=-1)
        query_g = F.normalize(query_g, p=2, dim=-1)
        
        logits = self.logit_scale * (query_g @ fused_prototypes.t())
        return logits
    
    def get_fused_prototypes(self, support_feats_2d, text_embeddings, K_shot):
        """事先計算並快取 89-way 的多模態融合原型"""
        B_supp, C, H, W = support_feats_2d.shape
        N = B_supp // K_shot
        
        supp_patches = support_feats_2d.flatten(2).transpose(1, 2)
        supp_cls = supp_patches.mean(dim=1)
        
        supp_patches = supp_patches.view(N, K_shot, supp_patches.size(1), -1).mean(dim=1) 
        supp_cls = supp_cls.view(N, K_shot, -1).mean(dim=1)

        # URM 跨模態對齊
        fused_local = self.adapter(supp_patches, text_embeddings) 
        fused_prototypes = torch.cat([supp_cls, fused_local], dim=-1) 
        fused_prototypes = self.metric_proj(fused_prototypes)            
        
        return F.normalize(fused_prototypes, p=2, dim=-1)

    def get_query_features(self, query_feats_2d):
        """快速提取 Query 全域特徵"""
        query_patches = query_feats_2d.flatten(2).transpose(1, 2)
        query_g = query_patches.mean(dim=1)
        return F.normalize(query_g, p=2, dim=-1)