# data/fewshot.py
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class EpisodicSampler:
    """
    符合標準 Few-Shot Learning 論文規範的隨機回合發牌機 (Sample-level Split)
    支援 PlantVillage 與 PlantDoc 架構
    """
    def __init__(self, data_dir, image_size=224, allowed_classes=None):
        print(f"⚙️ 正在初始化少樣本發牌機: {data_dir}")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 讀取目標資料夾 (確保不論是 train 還是 test，其內部類別順序保持絕對一致)
        self.dataset = datasets.ImageFolder(root=data_dir, transform=self.transform)
        
        if allowed_classes is not None:
            self.classes = [c for c in self.dataset.classes if c in allowed_classes]
        else:
            self.classes = self.dataset.classes
        self.classes.sort()  # 🌟 關鍵：強制排序，確保 Train 和 Test 的類別字串順序完全對齊
            
        # 將每個類別名稱與它在該資料夾底下的所有圖片 index 綁定
        self.class_to_indices = {c: [] for c in self.classes}
        for idx, (_, class_idx) in enumerate(self.dataset.samples):
            class_name = self.dataset.classes[class_idx]
            if class_name in self.classes:
                self.class_to_indices[class_name].append(idx)

    def sample_episode(self, n_way, k_shot, q_query):
        """
        嚴格抽取一個少樣本任務 (Episode)
        確保 Support 參考集與 Query 考題集來自同一個資料夾且圖片完全不重複
        """
        # 1. 隨機挑選 N-Way 個病害類別 (例如從 PlantDoc 的 27 個類別中盲抽 5 種)
        sampled_classes = random.sample(self.classes, n_way)
        
        support_images = []
        query_images = []
        query_labels = []  # 分配 0 ~ n_way-1 的局部相對標籤

        for i, class_name in enumerate(sampled_classes):
            indices = self.class_to_indices[class_name]
            
            # 2. 嚴格安全防呆：如果 Test 資料夾某些類別圖片太少，則使用 choices，否則用無重複的 sample
            if len(indices) < (k_shot + q_query):
                sampled_indices = random.choices(indices, k=k_shot + q_query)
            else:
                sampled_indices = random.sample(indices, k=k_shot + q_query)
                
            # 3. 嚴格切分：前 K 張當作現場參考書，後 Q 張當作考題
            support_idx = sampled_indices[:k_shot]
            query_idx = sampled_indices[k_shot:]
            
            # 讀取並打包 Support 圖片
            support_images.append(torch.stack([self.dataset[idx][0] for idx in support_idx]))
            
            # 讀取並打包 Query 圖片
            for q_i in query_idx:
                query_images.append(self.dataset[q_i][0])
                query_labels.append(i)  # 給予當前 Way 的相對標籤 (0 ~ N-1)
                
        # 4. 整理成符合 Mamba 2D 與 URM Matcher 要求的 Tensor 規格
        support_images = torch.stack(support_images) # [N_way, K_shot, 3, H, W]
        query_images = torch.stack(query_images)     # [N_way * Q_query, 3, H, W]
        query_labels = torch.tensor(query_labels, dtype=torch.long) # [N_way * Q_query]
        
        return support_images, query_images, query_labels, sampled_classes