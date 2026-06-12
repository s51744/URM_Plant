#!/bin/bash

# ==========================================================
# 🌿 URM MambaVision 雙導師 (DINOv2 + SigLIP) 終極實驗腳本
# ==========================================================

LOG_FILE="fsl_dual_teacher_results.log"

echo "==========================================================" > $LOG_FILE
echo "🧪 URM Dual-Teacher FSL (PlantVillage & PlantDoc) Report" >> $LOG_FILE
echo "Generated on: $(date)" >> $LOG_FILE
echo "==========================================================" >> $LOG_FILE

# 定義實驗資料集 (名稱:Train路徑:Val路徑:Test路徑)
# PlantDoc 因為沒有 val，所以 val 與 test 皆指向 test 資料夾
DATASETS=(
    "PlantVillage:/home/fishlord/dataset/PlantVillage_split/train:/home/fishlord/dataset/PlantVillage_split/val:/home/fishlord/dataset/PlantVillage_split/test"
    "PlantDoc:/home/fishlord/dataset/PlantDoc-Dataset/train:/home/fishlord/dataset/PlantDoc-Dataset/test:/home/fishlord/dataset/PlantDoc-Dataset/test"
)

for DATASET_INFO in "${DATASETS[@]}"; do
    # 解析字串獲取路徑
    IFS=":" read -r DATASET_NAME TRAIN_DATA VAL_DATA TEST_DATA <<< "${DATASET_INFO}"

    echo -e "\n\n==========================================================" | tee -a $LOG_FILE
    echo "📂 啟動資料集測試: $DATASET_NAME" | tee -a $LOG_FILE
    echo "==========================================================" | tee -a $LOG_FILE

    # 設定獨立的權重檔名防呆機制
    DEFAULT_CKPT="checkpoints/best_mamba_urm_5way.pth"
    TARGET_CKPT="checkpoints/best_mamba_${DATASET_NAME}_DualTeacher.pth"

    # --------------------------------------------------------
    # 1. 執行雙導師蒸餾訓練 (Train / Val Phase)
    # --------------------------------------------------------
    echo "--- 🏋️ Step 1: Training (20 Epochs) ---" | tee -a $LOG_FILE
    
    # 執行並將詳細 log 存入變數
    TRAIN_LOG=$(python train_fewshot.py \
        --source_data $TRAIN_DATA \
        --target_data $VAL_DATA \
        --epochs 20 \
        --n_way 5 \
        --k_shot 5)

    # 將完整 log 寫入檔案
    echo "$TRAIN_LOG" >> $LOG_FILE
    
    # 🌟 自動從 Log 中抓取該模型最後表現最好的 Train / Val 準確率印在畫面上
    echo "$TRAIN_LOG" | grep "結算 -> Train Acc" | tail -n 1 | tee -a $LOG_FILE

    # 重新命名權重檔案以防止被下一個資料集實驗覆蓋
    if [ -f "$DEFAULT_CKPT" ]; then
        mv "$DEFAULT_CKPT" "$TARGET_CKPT"
    fi

    # --------------------------------------------------------
    # 2. 執行 600 局終極測試 (Test Phase)
    # --------------------------------------------------------
    if [ -f "$TARGET_CKPT" ]; then
        echo "✅ Checkpoint renamed & saved at $TARGET_CKPT" | tee -a $LOG_FILE
        echo "--- 🕵️ Step 2: Testing (600 Episodes) ---" | tee -a $LOG_FILE

        TEST_LOG=$(python test_fewshot.py \
            --test_data $TEST_DATA \
            --weights $TARGET_CKPT \
            --test_episodes 600 \
            --n_way 5 \
            --k_shot 5)

        # 將完整測試 log 寫入檔案
        echo "$TEST_LOG" >> $LOG_FILE
        
        # 🌟 自動從 Log 中抓取最終的 600 回合 Test Acc 與信賴區間印在畫面上
        echo "$TEST_LOG" | grep -A 3 "最終少樣本評估結果" | tee -a $LOG_FILE

    else
        echo "❌ [ERROR] Failed to save weights on $DATASET_NAME." | tee -a $LOG_FILE
    fi

    echo "----------------------------------------------------------" >> $LOG_FILE
done

echo -e "\n✨ All Dual-Teacher FSL Experiments completed! Results are saved in $LOG_FILE"