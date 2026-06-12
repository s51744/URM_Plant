# utils/clip_teacher.py
import torch
from transformers import CLIPModel, CLIPTokenizer

def load_clip_teacher(model_name="openai/clip-vit-base-patch32", device="cuda"):
    """動態載入不同的 CLIP 模型作為知識蒸餾的教師"""
    print(f"📥 載入 CLIP 教師模型: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    
    # 教師模型必須永遠凍結 (不參與梯度更新)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    return model, tokenizer

def get_clip_text_embeddings(model, tokenizer, texts, device="cuda"):
    """使用指定的 CLIP 模型提取正規化後的文字特徵"""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        
        # 🌟 防呆機制：相容不同版本的 transformers API
        if hasattr(model, 'get_text_features'):
            features = model.get_text_features(**inputs)
        else:
            features = model(**inputs)
            
        # 如果 features 是一個物件 (BaseModelOutputWithPooling)，把它裡面的 Tensor 抽出來
        if hasattr(features, 'pooler_output'):
            features = features.pooler_output
        elif hasattr(features, 'text_embeds'):
            features = features.text_embeds
        elif isinstance(features, tuple):
            features = features[0]
            
        # 正規化 (L2 Normalize)，這在計算餘弦相似度或 URM 匹配時非常重要
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        
    return features