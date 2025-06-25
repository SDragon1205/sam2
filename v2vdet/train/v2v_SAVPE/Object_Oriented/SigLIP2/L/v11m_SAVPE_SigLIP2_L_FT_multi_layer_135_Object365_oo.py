import os, sys
import logging

import wandb
# from wandb.integration.ultralytics import add_wandb_callback
# from ultralytics.utils.callbacks.wb import on_train_epoch_end, on_fit_epoch_end, on_train_end

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld
import json

from datetime import datetime
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_With_MultiScale_SAVPE_SigLIP2_L_ObjectOriented
import torch
from ultralytics import YOLO

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  # device = int(cuda_devices.split(',')[0])
  device = cuda_devices
else:
  device = 0

scale='m'
DATASET_YAML = "v2vdet_ultralytics/cfg/datasets/ABO_256.yaml"
MODEL_YAML = f"v2vdet_ultralytics/cfg/models/v2v/11/yolo11{scale}-v2v-multiscale_1_3_5.yaml"
BATCH_SIZE = 128
EPOCHS = 5
NUM_WORKERS = 32
EVAL_BATCH_SIZE = 2

ARGUM_DICT={
          'rotation_range': (-30, 30),  
          'scale_range': (0.8, 1.2),    
          'brightness_range': (0.8, 1.2),
          'contrast_range': (0.8, 1.2),
          'prob': 0.75,              
          'global_prob': 0.7
        }
CKPT_NAME = f'ckpt/yolo11{scale}.pt'
NAME = f"OO_v11{scale}_SAVPE_SigLIP2_FT_multi_layer_135_Object365"
FROZEN_VISION_ENCODER = False

def main():
  
  model = V2V_With_MultiScale_SAVPE_SigLIP2_L_ObjectOriented(MODEL_YAML)
  
  data = dict(
    train=dict(
        yolo_data=[DATASET_YAML],
    ),
    val=dict(yolo_data=[DATASET_YAML]),
  )
  wandb_enable = True
  current_time = datetime.now().strftime("%Y%m%d_%H%M")

  name = NAME
  
  model._load(CKPT_NAME, task='task')
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
    workers=NUM_WORKERS,
    close_mosaic=0,
    save_period=1,
    cache=False,
    exist_ok=True,
    plots=True,
    cos_lr=True,
    eval_batch_size = EVAL_BATCH_SIZE,
    gradient_accumulation_steps = 256//BATCH_SIZE,
    frozen_vision_encoder = FROZEN_VISION_ENCODER,
    argum_dict=ARGUM_DICT,
    )

  # from v2vdet.v2vdet_ultralytics.utils.eval_function import eval_function

  # PROJECT_NAME='SAVPE_PE_v2v'

  # ckpt_name = f'v2v_training_result/{NAME}/weights/best.pt'
  
  # eval_function(MODEL=V2V_With_MultiScale_SAVPE_ObjectOriented,
  #               MODEL_YAML=MODEL_YAML,
  #               CKPT_NAME=ckpt_name,
  #               NAME=NAME,
  #               PROJECT_NAME=PROJECT_NAME,
  #               BATCH_SIZE=BATCH_SIZE)
    
if __name__ == '__main__':
  main()