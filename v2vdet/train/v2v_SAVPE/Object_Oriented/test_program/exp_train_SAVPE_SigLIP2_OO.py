import os, sys
import logging

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  device = cuda_devices
else:
  device = 0

from v2vdet.v2vdet_ultralytics.utils.wandb_callback import add_wandb_callback
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_With_MultiScale_SAVPE_ObjectOriented
import torch
from ultralytics import YOLO

def main():
  data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"],
    ),
    val=dict(yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"]),
  )
  wandb_enable = False

  ckpt_name = f'ckpt/yolo11s.pt'
  model = V2V_With_MultiScale_SAVPE_ObjectOriented("v2vdet_ultralytics/cfg/models/v2v/11/yolo11s-v2v-multiscale_1_3_5.yaml")
  model._load(weights=ckpt_name, task='task')

  result = model.train(
    project='training_result',
    name='test',
    data=data,
    batch=16,
    epochs=1,
    device=device,
    workers=0,
    close_mosaic=1,
    save_period=2,
    cache=False,
    exist_ok=True,
    plots=True,
    cos_lr=True,
    eval_batch_size = 2,
    gradient_accumulation_steps = 32,
    frozen_vision_encoder = False
    )

if __name__ == '__main__':
  main()