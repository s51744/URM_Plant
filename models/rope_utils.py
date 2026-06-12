# models/rope_utils.py
import torch
import math

def compute_2d_sincos(H, W, dim, temperature=10000., device='cpu'):
    """
    動態計算 2D RoPE 的 sin 與 cos 矩陣
    """
    half_dim = dim // 2
    # 修正錯誤：這裡現在有 import torch，不會報錯了
    inv_freq = 1.0 / (temperature ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))
    
    grid_h = torch.arange(H, device=device).float()
    grid_w = torch.arange(W, device=device).float()
    
    emb_h = torch.einsum('i,j->ij', grid_h, inv_freq)
    emb_w = torch.einsum('i,j->ij', grid_w, inv_freq)
    
    # 將 H 和 W 的編碼拼接
    emb = torch.cat([emb_h.unsqueeze(1).repeat(1, W, 1), 
                     emb_w.unsqueeze(0).repeat(H, 1, 1)], dim=-1)
    
    emb = emb.view(-1, half_dim)
    sin = emb.sin()
    cos = emb.cos()
    
    return sin, cos

def apply_rotary_pos_emb(x, sin, cos):
    """
    將 2D RoPE 旋轉矩陣施加到特徵上
    x shape: [Batch, Heads, Seq_Len, Head_Dim]
    """
    # 將特徵分為偶數維度與奇數維度以進行旋轉
    x1, x2 = x[..., 0::2], x[..., 1::2]
    
    # 調整 sin, cos 的形狀以支援 Broadcasting
    sin = sin.view(1, 1, sin.shape[0], sin.shape[1])
    cos = cos.view(1, 1, cos.shape[0], cos.shape[1])
    
    # 旋轉矩陣運算
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot