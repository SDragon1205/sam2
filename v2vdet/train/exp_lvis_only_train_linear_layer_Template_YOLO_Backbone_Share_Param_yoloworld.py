import os, sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import wandb
# from wandb.integration.ultralytics import add_wandb_callback
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics.utils.callbacks.wb import on_train_epoch_end, on_fit_epoch_end, on_train_end

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from datetime import datetime
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_Template_YOLO_Backbone_Share_Param_Train_Linear_Layer

import torch
from ultralytics import YOLO

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  device = int(cuda_devices.split(',')[0])
else:
  device = 0

def main():
  data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"],
    ),
    val=dict(yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"]),
  )

  project_name = 'runs'
  ckpt_name = 'ckpt/yolov8m-world.pt'
  name = f"V2V_Template_YOLOv8m_Backbone_Share_Param_Train_Linear_Layer"
  
  model = V2V_Template_YOLO_Backbone_Share_Param_Train_Linear_Layer("ultralytics/cfg/models/v8/yolov8m-world.yaml")

  model._load(ckpt_name, task='task')
  
  result = model.train(
    project=f"training_result/{project_name}",
    name=name,
    data=data,
    batch=32,
    epochs=1,
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
