import os, sys
import logging

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  device = cuda_devices
else:
  device = 0

from v2vdet.v2vdet_ultralytics.utils.wandb_callback import add_wandb_callback
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_With_MultiScale_SAVPE_YOLOE_ObjectOriented as MODEL
import torch
from ultralytics import YOLO

def main():
  data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/ABO_100_testing_64.yaml"],
    ),
    val=dict(yolo_data=["v2vdet_ultralytics/cfg/datasets/ABO_100_testing_64.yaml"]),
  )
  wandb_enable = False

  ckpt_name = f'ckpt/offical_yoloe/yoloe-v8s-seg.pt'
  model = MODEL("ultralytics/cfg/models/11/yoloe-11s.yaml", task='detect')
  ckpt = torch.load(ckpt_name)
  model.load(weights=ckpt['model'])

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