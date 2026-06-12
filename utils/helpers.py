# utils/helpers.py
import random
import numpy as np
import torch

def set_seed(seed=42):
    """鎖死所有的隨機種子，確保實驗可完全復現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_class_name(name):
    """清理資料夾名稱，轉化為 CLIP 看得懂的 Prompt"""
    return str(name).replace('___', ' ').replace('_', ' ')