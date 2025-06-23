#!/usr/bin/env bash
#
# 檔名：run_train_loop.sh
# 功能：若 training/train.py 發生錯誤則自動重啟
# 用法：chmod +x run_train_loop.sh && ./run_train_loop.sh

# === 1. 你要執行的指令 =========================
CMD="python training/train.py \
      -c configs/abo/1_tt20_oom11_s_mode2_freeze_num_maskmem_1_before_neck_memory_position_0_no_mask_downsampler_pos_enc_at_attn_num_layers_1.yaml \
      --use-cluster 0 \
      --num-gpus 1"

# === 2. 無限循環，直到成功 ======================
while true; do
    echo "[$(date '+%F %T')] ⏩  開始執行訓練..."
    eval $CMD                                 # 執行指令
    EXIT_CODE=$?                              # 取得退出碼

    if [[ $EXIT_CODE -eq 0 ]]; then           # 成功就跳出
        echo "[$(date '+%F %T')] ✅  訓練順利結束！"
        break
    else                                      # 失敗就重試
        echo "[$(date '+%F %T')] ❌  訓練失敗 (exit code=$EXIT_CODE)，10 秒後重啟..."
        sleep 10
    fi
done