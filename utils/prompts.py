# utils/prompts.py
import torch
import clip

def generate_text_embeddings(device, class_names):
    """
    使用 CLIP 生成植物病害的文字特徵向量 (Text Embeddings)
    """
    print("Loading OpenAI CLIP model for Text Embeddings...")
    model, _ = clip.load("ViT-B/32", device=device)
    
    # 建構 Prompt 模板，增強語言語意
    prompts = [f"A photo of a plant leaf with {disease}" for disease in class_names]
    
    # 進行 Tokenize 並推論
    text_tokens = clip.tokenize(prompts).to(device)
    
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        # 正規化特徵，對 Cosine 相似度或 Attention 計算很重要
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # 【新增這行】將 CLIP 輸出的 float16 強制轉為標準 float32
        text_embeddings = text_embeddings.float()
        
    print(f"Generated text embeddings shape: {text_embeddings.shape}") 
    return text_embeddings