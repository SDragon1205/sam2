import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

from iopath.common.file_io import g_pathmgr

ckpt_path = "/home/si2/sdragon/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
# ckpt_path = "/home/si2/sdragon/sam2/sam2_logs/configs/sam2.1_training/yolo_sam2.1_hiera_b+_MOSE_finetune.yaml/checkpoints/checkpoint.pt"
with g_pathmgr.open(ckpt_path, "rb") as f:
    state_dict = torch.load(f, map_location="cpu")

state_dict_copy = state_dict.copy()

# 使用兩層迴圈插入 yolo_detection_head 的參數
for k, v in state_dict.items():  # 第一層
    print(k)
    # if isinstance(v, dict):  # 如果值是字典，進入第二層