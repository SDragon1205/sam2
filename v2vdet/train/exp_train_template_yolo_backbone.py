import os, sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  device = int(cuda_devices.split(',')[0])
else:
  device = 0

import wandb
# from wandb.integration.ultralytics import add_wandb_callback
from v2vdet.v2vdet_ultralytics.utils.wandb_callback import add_wandb_callback

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from v2vdet.v2vdet_ultralytics.models.v2vdet.model import v2vdet_model, V2V_Template_YOLO_Backbone
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch, v2vTrainerFromScratch, v2vWorldTrainerFromScratch
import torch
from ultralytics import YOLO

def main():
  data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"],
        # yolo_data=["v2vdet_ultralytics/cfg/datasets/Objects365.yaml"],
        # grounding_data=[
        #     dict(
        #         img_path=f"{project_root}/DATASET/flickr30k/images",
        #         json_file=f"{project_root}/DATASET/flickr30k/final_flickr_separateGT_train.json",
        #     ),
            # dict(
            #     img_path="../datasets/GQA/images",
            #     json_file="../datasets/GQA/final_mixed_train_no_coco.json",
            # ),
        # ],
    ),
    val=dict(yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"]),
  )
  wandb_enable = False

  ckpt_name = 'ckpt/yolov8s-world.pt'
  model = V2V_Template_YOLO_Backbone("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
  model._load(ckpt_name, task='task')

  result = model.train(data=data,
                       batch=32,
                       epochs=1,
                       device=device,
                       project='v2vdet_world_train_project',
                       name=ckpt_name,
                       workers=8,
                       cache=False,
                       exist_ok=True)
  
  # data = "v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"
  # result = model.train(data=data,
  #                     batch=32, 
  #                     epochs=20,
  #                     plots=True,
  #                     resume=True)
  

if __name__ == '__main__':
  main()
