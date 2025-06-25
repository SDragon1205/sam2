import os, sys
import logging

import wandb
# from wandb.integration.ultralytics import add_wandb_callback
# from ultralytics.utils.callbacks.wb import on_train_epoch_end, on_fit_epoch_end, on_train_end

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld
import json

from datetime import datetime
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_With_MultiScale_SAVPE
import torch
from ultralytics import YOLO
from v2vdet.v2vdet_ultralytics.utils.eval_function import eval_function

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  # device = int(cuda_devices.split(',')[0])
  device = cuda_devices
else:
  device = 0

scale='m'
DATASET_YAML = "v2vdet_ultralytics/cfg/datasets/Objects365.yaml"
MODEL_YAML = f"v2vdet_ultralytics/cfg/models/v2v/11/yolo11{scale}-v2v-multiscale_1_3_5.yaml"
BATCH_SIZE = 32
NUM_WORKERS = 0

CKPT_NAME = f'ckpt/v11m_SAVPE_SigLIP2_FT_multi_layer_135_Object365.pt'
NAME = f"v11{scale}_SAVPE_SigLIP2_FT_multi_layer_135_Object365"
FROZEN_VISION_ENCODER = False

def main():
  
  PROJECT_NAME='SAVPE_v2v'

  ckpt_name = CKPT_NAME
  
  eval_function(MODEL=V2V_With_MultiScale_SAVPE,
                MODEL_YAML=MODEL_YAML,
                CKPT_NAME=ckpt_name,
                NAME=NAME,
                PROJECT_NAME=PROJECT_NAME,
                BATCH_SIZE=BATCH_SIZE)
    
if __name__ == '__main__':
  main()