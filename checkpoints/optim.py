import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

from iopath.common.file_io import g_pathmgr

# ckpt_path = "/home/si2/sdragon/sam2/sam2_logs/configs/sam2.1_training/yolom_s.yaml/checkpoints/checkpoint.pt"
ckpt_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s.pt"
with g_pathmgr.open(ckpt_path, "rb") as f:
    state_dict = torch.load(f, map_location="cpu")

for k, v in state_dict.items():  # 第一層
    print(k)