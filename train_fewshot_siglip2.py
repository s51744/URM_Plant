# train_fewshot_siglip2.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
from transformers import AutoProcessor, AutoModel # 🌟 載入 SigLIP2 需要

from utils.helpers import set_seed, clean_class_name
from models.backbone import MambaVisionStudent
from data.fewshot import EpisodicSampler
from models.fewshot_matcher import URMFewShotMatcher

torch.serialization.add_safe_globals([argparse.Namespace])

def parse_args():
    parser = argparse.ArgumentParser(description="URM KD Few-Shot Training (SigLIP2 Baseline)")
    parser.add_argument('--source_data', type=str, required=True)
    parser.add_argument('--target_data', type=str, required=True)
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--q_query', type=int, default=15)
    parser.add_argument('--train_episodes', type=int, default=100)
    parser.add_argument('--val_episodes', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kd_weight', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    set_seed(42)
    args = parse_args()
    print(f"🔥 啟動 URM 知識蒸餾訓練 (SigLIP2 教師) | {args.n_way}-Way 🔥")
    os.makedirs('checkpoints', exist_ok=True)

    source_sampler = EpisodicSampler(args.source_data)
    target_sampler = EpisodicSampler(args.target_data)

    # 🌟 載入 SigLIP2 處理器與模型
    model_id = "google/siglip2-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_id)
    siglip_model = AutoModel.from_pretrained(model_id).to(args.device)
    siglip_model.eval()

    # 自動抓取 SigLIP2 的視覺與文字維度 (Base版通常為 768)
    teacher_vision_dim = siglip_model.config.vision_config.hidden_size
    text_dim = siglip_model.config.text_config.hidden_size

    student_vision = MambaVisionStudent(target_dim=teacher_vision_dim).to(args.device)
    urm_matcher = URMFewShotMatcher(visual_dim=teacher_vision_dim, text_dim=text_dim).to(args.device)
    
    trainable_params = [p for p in student_vision.parameters() if p.requires_grad] + \
                       [p for p in urm_matcher.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = nn.MSELoss()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        student_vision.train()
        urm_matcher.train()
        train_loss, train_acc = 0.0, 0.0
        accumulation_steps = 4 
        optimizer.zero_grad()
        
        train_pbar = tqdm(range(args.train_episodes), desc=f"Epoch {epoch} [Train]")
        for step_idx in train_pbar:
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
            
            with torch.autocast(device_type=args.device, dtype=torch.float16):
                with torch.no_grad():
                    # 🌟 SigLIP2 文字特徵提取
                    text_inputs = processor(text=descriptions, padding=True, return_tensors="pt", truncation=True).to(args.device)
                    text_embeds = siglip_model.get_text_features(**text_inputs)
                    
                    # 🌟 SigLIP2 視覺特徵提取 (強制縮放至 224x224)
                    imgs_224 = F.interpolate(all_imgs, size=(224, 224), mode='bicubic', align_corners=False)
                    vision_outputs = siglip_model.vision_model(pixel_values=imgs_224)
                    teacher_target_1d = vision_outputs.pooler_output

                student_feats_2d = student_vision(all_imgs)
                student_feats_1d = student_feats_2d.mean(dim=[2, 3]) 
                
                s_len = support_imgs_flat.size(0)
                s_feats_2d, q_feats_2d = student_feats_2d[:s_len], student_feats_2d[s_len:]
                
                logits = urm_matcher(s_feats_2d, q_feats_2d, text_embeds, K_shot=args.k_shot) 
                
                loss_cls = criterion_cls(logits, query_labels)
                loss_kd = criterion_kd(student_feats_1d, teacher_target_1d)
                loss = (loss_cls + args.kd_weight * loss_kd) / accumulation_steps
            
            scaler.scale(loss).backward()
            
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
                
                # 🌟 SigLIP2 驗證期文字特徵
                text_inputs = processor(text=descriptions, padding=True, return_tensors="pt", truncation=True).to(args.device)
                text_embeds = siglip_model.get_text_features(**text_inputs)
                
                student_feats_2d = student_vision(all_imgs)
                s_len = support_imgs_flat.size(0)
                s_feats_2d, q_feats_2d = student_feats_2d[:s_len], student_feats_2d[s_len:]
                
                logits = urm_matcher(s_feats_2d, q_feats_2d, text_embeds, K_shot=args.k_shot) 
                
                preds = logits.argmax(dim=1)
                val_acc += (preds == query_labels).float().mean().item()
                
        avg_val_acc = (val_acc / args.val_episodes) * 100
        print(f"\n📊 Epoch {epoch} 結算 -> Val Acc: {avg_val_acc:.2f}%")
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_path = f"checkpoints/best_mamba_urm_siglip2_{args.n_way}way.pth"
            torch.save({
                'epoch': epoch,
                'student_state_dict': student_vision.state_dict(),
                'urm_state_dict': urm_matcher.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)
            print(f"🌟 新高分！已儲存至 {save_path} (Acc: {best_val_acc:.2f}%)")
        print("-" * 60)

if __name__ == '__main__':
    main()