import os, sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from sympy import plot
import wandb
from wandb.integration.ultralytics import add_wandb_callback

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from v2vdet.v2vdet_ultralytics.models.v2vdet import YOLOWorld, v2vYOLOWorld
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch, v2vTrainerFromScratch
import torch

def main():
  
  ckpt_name = 'ckpt/yolov8s-world.pt'
  model = v2vYOLOWorld("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
  # ckpt = torch.load(ckpt_name)
  # model.load_state_dict(ckpt['model'])
  model._load(ckpt_name, task='task')
  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/coco.yaml",
                      save_json=True,
                      plots=True)
  # breakpoint()
  # from ultralytics import YOLO

  # # Create a YOLO-World model
  # model = YOLO("yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes

  # # Conduct model validation on the COCO8 example dataset
  # metrics = model.val(data="coco8.yaml")

if __name__ == '__main__':
    main()