# URM-MambaVision: 結合 SigLIP2 蒸餾與 2D RoPE 幾何對齊的少樣本植物病害分類系統

本專案實作了一套基於標準少樣本學習（Standard Episodic Few-Shot Learning）範式的植物病害識別系統。我們提出以邊緣輕量化狀態空間模型 **MambaVision-T** 作為 Student 網路，結合最新 **SigLIP2** 強大的圖文特徵空間進行知識蒸餾。

為了克服真實野外環境的極端雜訊，我們外掛了獨創的 **URM (Universal Representation Matcher)** 適配器大腦。它能夠拒絕傳統的特徵平均化，完整保留多模態語意，並透過 2D 旋轉位置編碼（2D RoPE）進行細粒度的空間對齊，在保留輕量化優勢的同時，實現 SOTA 等級的野外病害分類能力。

---

## 🚀 1. 核心技術創新 (Key Innovations)

* **邊緣輕量化骨幹 (MambaVision-T)**：採用最新 Mamba 與 ViT 混合架構，精準攔截 4D 空間特徵圖，極大化部署於農業邊緣裝置的潛力。
* **SigLIP2 知識蒸餾 (Knowledge Distillation)**：訓練期引入 `google/siglip2-base-patch16-224` 作為視覺導師，透過 1D 全域特徵的 MSE Loss，將大模型通用常識「灌頂」給輕量化學生。
* **2D RoPE 跨模態對齊大腦 (URM Matcher)**：在不破壞病斑二維空間結構的前提下，進行跨模態交叉注意力（Cross-Attention）矩陣連連看，拉開健康葉片與不同罹病形狀之間的幾何距離。
* **標準 Episodic 評估機制**：系統嚴格遵循 Few-Shot 互斥規範，測試期模型面對從未見過的全新病葉照片，必須純粹依賴現場發放的 K 張 Support 參考圖進行即時推論。

---

## 📂 2. 快速啟動 (Quick Start)

### A. 執行知識蒸餾與 URM 訓練
使用最新的 SigLIP2 腳本啟動 Episodic 訓練（以 5-Way 5-Shot 為例）：

```bash
python train_fewshot_siglip2.py \
    --source_data /path/to/dataset/train \
    --target_data /path/to/dataset/val \
    --epochs 40 \
    --n_way 5 \
    --k_shot 5 \
    --q_query 15
```

```bash
python test_fewshot.py \
    --test_data /path/to/dataset/test \
    --weights checkpoints/model.pth \
    --test_episodes 5 \
    --n_way 5