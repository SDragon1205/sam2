import os, sys
import logging
import wandb

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  device = cuda_devices
else:
  device = 0

from v2vdet.v2vdet_ultralytics.utils.wandb_callback import add_wandb_callback
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_DINO_with_registers
import torch
from ultralytics import YOLO

def main():
  data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/coco8.yaml"],
    ),
    val=dict(yolo_data=["v2vdet_ultralytics/cfg/datasets/coco8.yaml"]),
  )
  wandb_enable = False
  
  

  ckpt_name = 'ckpt/yolov8s-world.pt'
  model = V2V_DINO_with_registers("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
  model._load(weights=ckpt_name, task='task')

  # add_wandb_callback(model, enable_model_checkpointing=True)

  result = model.train(
    project='test',
    name='test',
    data=data,
    batch=1,
    epochs=1,
    device=device,
    workers=4,
    close_mosaic=0,
    save_period=2,
    cache=False,
    exist_ok=True,
    plots=True,
    cos_lr=True,
    eval_batch_size = 2,
    gradient_accumulation_steps = 32,
    frozen_vision_encoder = False
    )
  
  # wandb.finish()

  # ckpt_name = f'{project_folder}/weights/best.pt'

  # # Eval COCO
  # COCO_PATH = os.path.join(project_folder, 'COCO')
  # if not os.path.exists(path=COCO_PATH):
  #   os.makedirs(COCO_PATH)
  
  # model._load(ckpt_name, task='task')
  # model.training=False
  # model=model.eval()

  # metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/coco.yaml",
  #                   batch=32,
  #                   save_json=True,
  #                   half=True,
  #                   plots = True,
  #                   project=COCO_PATH,
  #                   name=name
  #                   )
  # metrics_dict = {
  #   "DATASET": "COCO2017",
  #   "map50_95": metrics.box.map.tolist(),
  #   "map50": metrics.box.map50.tolist(),
  #   "map75": metrics.box.map75.tolist(),
  #   "map_all_class": metrics.box.maps.tolist()
  # }
  
  # data = "v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"
  # result = model.train(data=data,
  #                     batch=32, 
  #                     epochs=20,
  #                     plots=True,
  #                     resume=True)

if __name__ == '__main__':
  main()