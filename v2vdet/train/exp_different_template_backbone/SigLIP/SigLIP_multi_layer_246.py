import os, sys
import logging

import wandb
# from wandb.integration.ultralytics import add_wandb_callback
# from ultralytics.utils.callbacks.wb import on_train_epoch_end, on_fit_epoch_end, on_train_end

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld
import json

from datetime import datetime
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_template_SigLIP_multi_scale
import torch
from ultralytics import YOLO

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  # device = int(cuda_devices.split(',')[0])
  device = cuda_devices
else:
  device = 0

DATASET_YAML = "v2vdet_ultralytics/cfg/datasets/Objects365_t80k_v1024.yaml"
MODEL_YAML = "v2vdet_ultralytics/cfg/models/v2v/yolov8s-world_multiscale_2_4_6.yaml"
BATCH_SIZE = 32
EPOCHS = 5

'''
  If you have finetuned fundation model, batch size recommendation is 8. (on 4 3090 GPUs, 24*4=96G VRAM)  
'''
ARGUM_DICT={
          'rotation_range': (-30, 30),  
          'scale_range': (0.8, 1.2),    
          'brightness_range': (0.8, 1.2),
          'contrast_range': (0.8, 1.2),
          'prob': 0.75,              
          'global_prob': 0.7
        }
CKPT_NAME = 'ckpt/yolov8s-world.pt'
NAME = f"SigLIP_multi_layer_246"
FROZEN_VISION_ENCODER = True

def main():
  
  model = V2V_template_SigLIP_multi_scale(MODEL_YAML)
  
  data = dict(
    train=dict(
        yolo_data=[DATASET_YAML],
    ),
    val=dict(yolo_data=[DATASET_YAML]),
  )
  wandb_enable = True
  current_time = datetime.now().strftime("%Y%m%d_%H%M")

  ckpt_name = CKPT_NAME
  name = NAME
  
  model._load(ckpt_name, task='task')
  if wandb_enable:
    wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
    # wandb.init(job_type="train", project=name, name=name, config=model)
  
  project_folder_only = f"v2v_training_result"
  project_folder = os.path.join(project_folder_only, name)
  
  model.train(
    project=project_folder_only,
    name=name,
    data=data,
    batch=BATCH_SIZE,
    epochs=EPOCHS,
    device=device,
    workers=8,
    close_mosaic=0,
    save_period=1,
    cache=False,
    exist_ok=True,
    plots=True,
    cos_lr=True,
    eval_batch_size = 2,
    gradient_accumulation_steps = 256//BATCH_SIZE,
    frozen_vision_encoder = FROZEN_VISION_ENCODER,
    argum_dict=ARGUM_DICT,
    )
  
  ckpt_name = f'{project_folder}/weights/best.pt'

  # Eval COCO
  COCO_PATH = os.path.join(project_folder, 'COCO')
  if not os.path.exists(path=COCO_PATH):
    os.makedirs(COCO_PATH)
  
  model._load(ckpt_name, task='task')
  model.training=False
  model=model.eval()

  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/coco.yaml",
                    batch=32,
                    save_json=True,
                    half=True,
                    plots = True,
                    project=COCO_PATH,
                    name=name
                    )
  metrics_dict = {
    "DATASET": "COCO2017",
    "map50_95": metrics.box.map.tolist(),
    "map50": metrics.box.map50.tolist(),
    "map75": metrics.box.map75.tolist(),
    "map_all_class": metrics.box.maps.tolist()
  }
  
  wandb.log(metrics_dict)
  
  with open (f"{COCO_PATH}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
    
  # Eval VisDrone
  VisDrone_PATH = os.path.join(project_folder, 'VisDrone')
  if not os.path.exists(path=VisDrone_PATH):
    os.makedirs(VisDrone_PATH)
  
  # model._load(ckpt_name, task='task')
  model.training=False
  model=model.eval()

  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/VisDrone.yaml",
                    batch=32,
                    save_json=True,
                    half=True,
                    plots = True,
                    project=VisDrone_PATH,
                    name=name
                    )
  
  metrics_dict = {
    "DATASET": "VisDrone",
    "map50_95": metrics.box.map.tolist(),
    "map50": metrics.box.map50.tolist(),
    "map75": metrics.box.map75.tolist(),
    "map_all_class": metrics.box.maps.tolist()
  }
  
  wandb.log(metrics_dict)
  
  with open (f"{VisDrone_PATH}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
    
  # Eval PascalVOC
  VOC_PATH = os.path.join(project_folder, 'VOC')
  if not os.path.exists(path=VOC_PATH):
    os.makedirs(VOC_PATH)
    
  model.training=False
  model=model.eval()

  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/VOC.yaml",
                    batch=32,
                    save_json=True,
                    half=True,
                    plots = True,
                    project=VOC_PATH,
                    name=name
                    )
  
  metrics_dict = {
    "DATASET": "VOC",
    "map50_95": metrics.box.map.tolist(),
    "map50": metrics.box.map50.tolist(),
    "map75": metrics.box.map75.tolist(),
    "map_all_class": metrics.box.maps.tolist()
  }
  
  wandb.log(metrics_dict)
  
  with open (f"{VOC_PATH}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
  
  wandb.finish()
  
if __name__ == '__main__':
  main()