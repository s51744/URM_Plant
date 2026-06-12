# test_fewshot.py
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import scipy.stats 

# 🌟 關鍵新增：將 argparse.Namespace 加入 PyTorch 2.6 的安全白名單
torch.serialization.add_safe_globals([argparse.Namespace])

# 🌟 完美的模組化引入
from utils.helpers import set_seed, clean_class_name
from models.backbone import MambaVisionStudent
from data.fewshot import EpisodicSampler
from models.fewshot_matcher import URMFewShotMatcher
from utils.clip_teacher import load_clip_teacher, get_clip_text_embeddings

def mean_confidence_interval(data, confidence=0.95):
    """計算符合頂會少樣本論文論文標準的 95% 信賴區間"""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a) if n > 1 else 0.0
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1) if n > 1 else 0.0
    return m, h

def main():
    parser = argparse.ArgumentParser(description="URM KD Few-Shot Inference")
    parser.add_argument('--test_data', type=str, required=True, help='Path to Test set')
    parser.add_argument('--weights', type=str, required=True, help='Path to saved .pth weights')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32')
    
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--q_query', type=int, default=15)
    parser.add_argument('--test_episodes', type=int, default=600)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # 🌟 規範一：測試期同樣在第一時間鎖死隨機種子，確保 600 回合大考的考題與發牌完全固定！
    set_seed(42)

    print(f"🚀 啟動 URM 推論測試 | 裝置: {args.device} | {args.n_way}-Way {args.k_shot}-Shot")

    # 1. 初始化資料集與模型
    test_sampler = EpisodicSampler(args.test_data)
    
    # 2. 載入 Teacher 並動態安全抓取文字與視覺維度，支援跨模型無痛切換
    clip_teacher, clip_tokenizer = load_clip_teacher(args.clip_model, args.device)
    teacher_vision_dim = clip_teacher.config.vision_config.hidden_size
    text_dim = clip_teacher.config.text_config.hidden_size # 自動適配 Base(512) 或 Large(768)

    # 3. 初始化學生與適配器組件
    student_vision = MambaVisionStudent(target_dim=teacher_vision_dim).to(args.device)
    urm_matcher = URMFewShotMatcher(visual_dim=teacher_vision_dim, text_dim=text_dim).to(args.device)

    # 4. 📥 載入權重檔案並嚴格設定為 eval 狀態，防止統計量污染
    print(f"📥 載入權重檔案: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=args.device)
    student_vision.load_state_dict(checkpoint['student_state_dict'])
    urm_matcher.load_state_dict(checkpoint['urm_state_dict'])
    print(f"✅ 成功載入！此權重在驗證集的最佳分數為: {checkpoint['best_val_acc']:.2f}%")

    student_vision.eval()
    urm_matcher.eval()
    clip_teacher.eval()

    episode_accuracies = []

    # 5. 開始 600 回合的標準少樣本測試
    test_pbar = tqdm(range(args.test_episodes), desc="Testing")

    with torch.no_grad():
        for _ in test_pbar:
            # 發牌機抽牌
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
            
            # 優化文字 Prompt，顯著拉開「健康」與「病害」的語意特徵距離
            descriptions = [f"A photo of a leaf infected with {clean_class_name(c)}." if "healthy" not in c.lower() 
                else f"A photo of a healthy {clean_class_name(c).replace('healthy', '').strip()} leaf." 
                for c in class_names]
            
            # 獲取文字特徵 (URM 需要)
            text_embeds = get_clip_text_embeddings(clip_teacher, clip_tokenizer, descriptions, args.device)
            
            # Student 提取視覺特徵與 URM 預測
            with torch.autocast(device_type=args.device, dtype=torch.float16):
                student_feats_2d = student_vision(all_imgs)
                s_len = support_imgs_flat.size(0)
                s_feats_2d, q_feats_2d = student_feats_2d[:s_len], student_feats_2d[s_len:]
                
                logits = urm_matcher(s_feats_2d, q_feats_2d, text_embeds, K_shot=args.k_shot) 
            
            # 計算單回合 Accuracy
            preds = logits.argmax(dim=1)
            acc = (preds == query_labels).float().mean().item() * 100
            episode_accuracies.append(acc)
            
            test_pbar.set_postfix({'Acc': f"{acc:.2f}%"})

    # 6. 結算最終成績與信賴區間
    mean_acc, ci95 = mean_confidence_interval(episode_accuracies)
    print("\n" + "="*50)
    print(f"🎉 最終少樣本評估結果 ({args.n_way}-Way {args.k_shot}-Shot)")
    print(f"📊 平均準確率 (Accuracy): {mean_acc:.2f}% +- {ci95:.2f}%")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()