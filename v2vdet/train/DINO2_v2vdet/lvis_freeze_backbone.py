import os, sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import wandb

from datetime import datetime
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_Template_YOLO_Backbone
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch, v2vTrainerFromScratch, v2vWorldTrainerFromScratch
import torch
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_DINO

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  device = int(cuda_devices.split(',')[0])
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

  project_name = 'v2vdet'
  ckpt_name = 'ckpt/yolov8s-world.pt'
  name = f"lvis_v2vdet_DINO_base_freeze_backbone"

  argum_dict = {
            'rotation_range': (-30, 30),
            'scale_range': (0.8, 1.2),
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.8, 1.2),
            'prob': 0.5,
            'global_prob': 1
            }


  model = V2V_DINO("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
  model._load(ckpt_name, task='task')

  wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
  wandb.init(job_type="train", project=project_name, name=name, config=model)

  result = model.train(
    project=f"training_result/{project_name}",
    name=name,
    data=data,
    batch=32,
    epochs=10,
    device=device,
    workers=8,
    close_mosaic=0,
    save_period=10,
    cache=False,
    exist_ok=True,
    plots=True,
    freeze=[freeze_idx for freeze_idx in range(10)]
    )

  wandb.finish()

if __name__ == '__main__':
  main()
