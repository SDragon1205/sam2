import os, sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import wandb
from wandb.integration.ultralytics import add_wandb_callback

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from v2vdet.v2vdet_ultralytics import YOLOWorld
from v2vdet.v2vdet_ultralytics import v2vdet_model
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch, v2vTrainerFromScratch
import torch

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
  model = v2vdet_model("v2vdet_ultralytics/cfg/models/v8/yolov8s-world_v2v.yaml")
  # model = YOLOWorld("yolov8s-world.pt")
  ckpt = torch.load(ckpt_name)
  # model.load_state_dict(ckpt['model'], strict=False)

  model._load(ckpt_name, task='task')
  result = model.train(data=data,
                       batch=16, 
                       epochs=20, 
                       device=2,
                       trainer=v2vTrainerFromScratch,
                       project='v2vdet_train_project_test',
                       name=ckpt_name,
                       workers=8,
                       lr0=0.0001,
                       warmup_epochs=5,
                       warmup_bias_lr = 0.1,
                       save_period=1, 
                       cache=False, 
                       exist_ok=True, 
                       plots=True)

if __name__ == '__main__':
  main()