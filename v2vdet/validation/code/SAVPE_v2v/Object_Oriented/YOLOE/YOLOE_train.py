import os, sys
import logging

import wandb
# from wandb.integration.ultralytics import add_wandb_callback
# from ultralytics.utils.callbacks.wb import on_train_epoch_end, on_fit_epoch_end, on_train_end

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld
import json

from datetime import datetime
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_With_MultiScale_SAVPE_YOLOE_ObjectOriented as MODEL
import torch
from ultralytics import YOLO
from v2vdet.v2vdet_ultralytics.utils.eval_function import eval_function

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  # device = int(cuda_devices.split(',')[0])
  device = cuda_devices
else:
  device = 0

scale = os.environ.get('SCALE','m')
MODEL_YAML = f"ultralytics/cfg/models/11/yoloe-11{scale}.yaml"
BATCH_SIZE = int(os.environ.get('BATCH_SIZE','32'))
EPOCHS = 5
NUM_WORKERS = 32
EVAL_BATCH_SIZE = 16

ARGUM_DICT={
          'rotation_range': (-30, 30),  
          'scale_range': (0.8, 1.2),    
          'brightness_range': (0.8, 1.2),
          'contrast_range': (0.8, 1.2),
          'prob': 0.75,              
          'global_prob': 0.7
        }

NAME = f"OO_v11{scale}_YOLOE_train"
# CKPT_NAME = f'ckpt/Object_Oriented_CKPT/{NAME}.pt'
# CKPT_NAME = f'ckpt/offical_yoloe/yoloe-11{scale}-seg.pt'
CKPT_NAME = f"v2v_training_result/from_H100/YOLOE/OO_v11{scale}_SAVPE_YOLOE/weights/best.pt"
# CKPT_NAME = f"v2v_training_result/from_H100/YOLOE/OO_v11l_SAVPE_YOLOE/weights/best.pt"

FROZEN_VISION_ENCODER = False

def main():
  
  PROJECT_NAME='OO_SAVPE_v2v'

  ckpt_name = CKPT_NAME
  
  # eval_function(MODEL=MODEL,
  #               MODEL_YAML=MODEL_YAML,
  #               CKPT_NAME=ckpt_name,
  #               NAME=NAME,
  #               PROJECT_NAME=PROJECT_NAME,
  #               BATCH_SIZE=BATCH_SIZE,
  #               DATASET_LIST=['ABO_100'],
  #               exist_ok=True)

  eval_function(MODEL=MODEL,
                MODEL_YAML=MODEL_YAML,
                CKPT_NAME=ckpt_name,
                NAME=NAME,
                PROJECT_NAME=PROJECT_NAME,
                BATCH_SIZE=BATCH_SIZE,
                DATASET_LIST=['ABO_199_HEADBOARD'],
                exist_ok=True)
    
if __name__ == '__main__':
  main()