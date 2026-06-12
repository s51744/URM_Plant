# train_fewshot.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from utils.helpers import set_seed, clean_class_name
from models.backbone import MambaVisionStudent
from data.fewshot import EpisodicSampler
from models.fewshot_matcher import URMFewShotMatcher
from utils.clip_teacher import load_clip_teacher, get_clip_text_embeddings

# 將 argparse.Namespace 加入 PyTorch 2.6 的安全白名單
torch.serialization.add_safe_globals([argparse.Namespace])

def parse_args():
    parser = argparse.ArgumentParser(description="URM KD Few-Shot Training (MambaVision-T)")
    parser.add_argument('--source_data', type=str, required=True, help='Path to Train set (e.g. ./train)')
    parser.add_argument('--target_data', type=str, required=True, help='Path to Test/Val set (e.g. ./test)')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32')
    
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--q_query', type=int, default=15)
    
    parser.add_argument('--train_episodes', type=int, default=100)
    parser.add_argument('--val_episodes', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kd_weight', type=float, default=0.5, help='KD Loss 的權重')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    # 🌟 規範一：在程式啟動的最源頭鎖死亂數，全域控制發牌機的可復現性
    set_seed(42)
    
    args = parse_args()
    print(f"🔥 啟動 URM 知識蒸餾訓練 (MambaVision) | 裝置: {args.device} | {args.n_way}-Way 🔥")
    os.makedirs('checkpoints', exist_ok=True)

    # 1. 載入遵循 FSL 規範的樣本互斥發牌機
    source_sampler = EpisodicSampler(args.source_data)
    target_sampler = EpisodicSampler(args.target_data)

    # 2. 載入 Teacher (CLIP) -> 完全凍結
    clip_teacher, clip_tokenizer = load_clip_teacher(args.clip_model, args.device)
    for param in clip_teacher.parameters():
        param.requires_grad = False
    clip_teacher.eval()
    
    teacher_vision_dim = clip_teacher.config.vision_config.hidden_size

    # 3. 載入自 models/backbone.py 的 Student 模型
    student_vision = MambaVisionStudent(target_dim=teacher_vision_dim).to(args.device)
    
    # 4. 載入 URM 適配器並自動判定維度，防呆避免 Large 模型崩潰
    text_dim = clip_teacher.config.text_config.hidden_size
    urm_matcher = URMFewShotMatcher(visual_dim=teacher_vision_dim, text_dim=text_dim).to(args.device)
    
    # 5. 優化器只帶走 Student 與 URM 的 learnable 參數
    trainable_params = [p for p in student_vision.parameters() if p.requires_grad] + \
                       [p for p in urm_matcher.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = nn.MSELoss()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---------- Train Phase ----------
        student_vision.train()
        urm_matcher.train()
        train_loss, train_acc = 0.0, 0.0
        
        accumulation_steps = 4 
        optimizer.zero_grad()
        
        train_pbar = tqdm(range(args.train_episodes), desc=f"Epoch {epoch} [Train]")
        for step_idx in train_pbar:
            
            # 抽牌與資料處理
            support_imgs, query_imgs, query_labels, class_names = source_sampler.sample_episode(
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
            descriptions = [f"A photo of a {clean_class_name(c)}." for c in class_names]
            
            # AMP 混合精度運算
            with torch.autocast(device_type=args.device, dtype=torch.float16):
                # Teacher 提取標準答案
                with torch.no_grad():
                    text_embeds = get_clip_text_embeddings(clip_teacher, clip_tokenizer, descriptions, args.device)
                    vision_outputs = clip_teacher.vision_model(pixel_values=all_imgs)
                    teacher_feats_tensor = vision_outputs.pooler_output

                # Student 提取特徵
                student_feats_2d = student_vision(all_imgs)
                student_feats_1d = student_feats_2d.mean(dim=[2, 3])
                
                s_len = support_imgs_flat.size(0)
                s_feats_2d, q_feats_2d = student_feats_2d[:s_len], student_feats_2d[s_len:]
                
                # URM 預測
                logits = urm_matcher(s_feats_2d, q_feats_2d, text_embeds, K_shot=args.k_shot) 
                
                # 雙重 Loss 計算
                loss_cls = criterion_cls(logits, query_labels)
                loss_kd = criterion_kd(student_feats_1d, teacher_feats_tensor)
                loss = (loss_cls + args.kd_weight * loss_kd) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            # 梯度累積更新
            if (step_idx + 1) % accumulation_steps == 0 or (step_idx + 1) == args.train_episodes:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += (loss.item() * accumulation_steps)
            preds = logits.argmax(dim=1)
            train_acc += (preds == query_labels).float().mean().item()
            train_pbar.set_postfix({'L_cls': f"{loss_cls.item():.3f}", 'L_kd': f"{loss_kd.item():.3f}"})

        # ---------- Val Phase ----------
        student_vision.eval()
        urm_matcher.eval()
        val_acc = 0.0
        
        val_pbar = tqdm(range(args.val_episodes), desc=f"Epoch {epoch} [Val]")
        with torch.no_grad(): 
            for _ in val_pbar:
                support_imgs, query_imgs, query_labels, class_names = target_sampler.sample_episode(
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
                descriptions = [f"A photo of a {clean_class_name(c)}." for c in class_names]
                
                text_embeds = get_clip_text_embeddings(clip_teacher, clip_tokenizer, descriptions, args.device)
                
                student_feats_2d = student_vision(all_imgs)
                s_len = support_imgs_flat.size(0)
                s_feats_2d, q_feats_2d = student_feats_2d[:s_len], student_feats_2d[s_len:]
                
                logits = urm_matcher(s_feats_2d, q_feats_2d, text_embeds, K_shot=args.k_shot) 
                
                preds = logits.argmax(dim=1)
                val_acc += (preds == query_labels).float().mean().item()
                
        # ---------- 結算與存檔邏輯 ----------
        avg_train_acc = (train_acc / args.train_episodes) * 100
        avg_val_acc = (val_acc / args.val_episodes) * 100
        
        print(f"\n📊 Epoch {epoch} 結算 -> Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}%")
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_path = f"checkpoints/best_mamba_urm_{args.n_way}way.pth"
            
            torch.save({
                'epoch': epoch,
                'student_state_dict': student_vision.state_dict(),
                'urm_state_dict': urm_matcher.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)
            print(f"🌟 發現新高分！模型已儲存至 {save_path} (Acc: {best_val_acc:.2f}%)")
        print("-" * 60)

if __name__ == '__main__':
    main()