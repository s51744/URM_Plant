# test_fewshot_siglip2.py
import argparse
import torch
from tqdm import tqdm
import numpy as np
import scipy.stats 
from transformers import AutoProcessor, AutoModel # 🌟 載入 SigLIP2

torch.serialization.add_safe_globals([argparse.Namespace])

from utils.helpers import set_seed, clean_class_name
from models.backbone import MambaVisionStudent
from data.fewshot import EpisodicSampler
from models.fewshot_matcher import URMFewShotMatcher

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a) if n > 1 else 0.0
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1) if n > 1 else 0.0
    return m, h

def main():
    parser = argparse.ArgumentParser(description="URM KD Few-Shot Inference (SigLIP2 Baseline)")
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--q_query', type=int, default=15)
    parser.add_argument('--test_episodes', type=int, default=600)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(42)

    print(f"🚀 啟動 URM 推論測試 (SigLIP2 基準) | {args.n_way}-Way {args.k_shot}-Shot")

    test_sampler = EpisodicSampler(args.test_data)
    
    # 🌟 載入 SigLIP2
    model_id = "google/siglip2-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_id)
    siglip_model = AutoModel.from_pretrained(model_id).to(args.device)
    siglip_model.eval()

    # 自動抓取維度
    teacher_vision_dim = siglip_model.config.vision_config.hidden_size
    text_dim = siglip_model.config.text_config.hidden_size

    student_vision = MambaVisionStudent(target_dim=teacher_vision_dim).to(args.device)
    urm_matcher = URMFewShotMatcher(visual_dim=teacher_vision_dim, text_dim=text_dim).to(args.device)

    checkpoint = torch.load(args.weights, map_location=args.device)
    student_vision.load_state_dict(checkpoint['student_state_dict'])
    urm_matcher.load_state_dict(checkpoint['urm_state_dict'])
    
    student_vision.eval()
    urm_matcher.eval()

    episode_accuracies = []
    test_pbar = tqdm(range(args.test_episodes), desc="Testing")

    with torch.no_grad():
        for _ in test_pbar:
            support_imgs, query_imgs, query_labels, class_names = test_sampler.sample_episode(
                args.n_way, args.k_shot, args.q_query
            )
            
            support_imgs = support_imgs.to(args.device)
            query_imgs = query_imgs.to(args.device)
            query_labels = query_labels.to(args.device)
            
            N_way, K_shot, C_img, H_img, W_img = support_imgs.shape
            support_imgs_flat = support_imgs.view(N_way * K_shot, C_img, H_img, W_img)
            if query_imgs.dim() == 5:
                query_imgs = query_imgs.view(-1, C_img, H_img, W_img)
                
            all_imgs = torch.cat([support_imgs_flat, query_imgs], dim=0)
            
            descriptions = [f"A photo of a leaf infected with {clean_class_name(c)}." if "healthy" not in c.lower() 
                else f"A photo of a healthy {clean_class_name(c).replace('healthy', '').strip()} leaf." 
                for c in class_names]
            
            # 🌟 透過 SigLIP2 Processor 處理文字
            text_inputs = processor(text=descriptions, padding=True, return_tensors="pt", truncation=True).to(args.device)
            text_embeds = siglip_model.get_text_features(**text_inputs)
            
            with torch.autocast(device_type=args.device, dtype=torch.float16):
                student_feats_2d = student_vision(all_imgs)
                s_len = support_imgs_flat.size(0)
                s_feats_2d, q_feats_2d = student_feats_2d[:s_len], student_feats_2d[s_len:]
                
                logits = urm_matcher(s_feats_2d, q_feats_2d, text_embeds, K_shot=args.k_shot) 
            
            preds = logits.argmax(dim=1)
            acc = (preds == query_labels).float().mean().item() * 100
            episode_accuracies.append(acc)
            test_pbar.set_postfix({'Acc': f"{acc:.2f}%"})

    mean_acc, ci95 = mean_confidence_interval(episode_accuracies)
    print("\n" + "="*50)
    print(f"🎉 最終評估結果 (SigLIP2 教師) | {args.n_way}-Way {args.k_shot}-Shot")
    print(f"📊 平均準確率: {mean_acc:.2f}% +- {ci95:.2f}%")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()