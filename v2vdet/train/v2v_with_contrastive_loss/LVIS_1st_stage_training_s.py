import os, sys
import logging

import wandb
# from wandb.integration.ultralytics import add_wandb_callback
from ultralytics.utils.callbacks.wb import on_train_epoch_end, on_fit_epoch_end, on_train_end

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld
import json

from datetime import datetime
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_Template_YOLO_Backbone_Model_Contrastive_Loss
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch, v2vTrainerFromScratch, v2vWorldTrainerFromScratch
import torch
from ultralytics import YOLO

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  # device = int(cuda_devices.split(',')[0])
  device = cuda_devices
else:
  device = 0

def main():
  data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis.yaml"],
    ),
    val=dict(yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis.yaml"]),
  )
  wandb_enable = True
  current_time = datetime.now().strftime("%Y%m%d_%H%M")

  project_name = 'v2vdet_A100_gay'
  ckpt_name = 'ckpt/yolov8s-world.pt'
  name = f"LVIS_Template_YOLO_s_Backbone_Share_Param"

  model = V2V_Template_YOLO_Backbone_Model_Contrastive_Loss("ultralytics/cfg/models/v8/yolov8s-world.yaml")
  model._load(ckpt_name, task='task')
  wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
  wandb.init(job_type="train", project=project_name, name=name, config=model)
  
  project_folder = f"training_result/{project_name}/{name}"

  result = model.train(
    project=project_folder,
    name=name,
    data=data,
    batch=64,
    epochs=10,
    device=device,
    workers=16,
    close_mosaic=0,
    save_period=1,
    cache=False,
    exist_ok=True,
    plots=True,
    )
  
if __name__ == '__main__':
  main()