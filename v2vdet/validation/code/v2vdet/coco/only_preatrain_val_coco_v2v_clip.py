import os, sys
import logging
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

import wandb
from wandb.integration.ultralytics import add_wandb_callback

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from v2vdet.v2vdet_ultralytics.models.v2vdet import v2vYOLOWorld
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch, v2vTrainerFromScratch
import torch

def main():
  
  project = "validation/v2vdet/coco_val_result"
  name = "only_pretrain"
  
  ckpt_name = 'ckpt/yolov8s-world.pt'
  # ckpt_name = 'lvis_v2vdet_world_train_project/train_all17/weights/best.pt'
  model = v2vYOLOWorld("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")

  model._load(ckpt_name, task='task')
  model.training=False

  metrics = model.val(data="coco.yaml",
                    batch=32,
                    save_json=True,
                    half=True,
                    plots = True,
                    project=project,
                    name=name
                    )
  metrics_dict = {
    "map50_95": metrics.box.map.tolist(),
    "map50": metrics.box.map50.tolist(),
    "map75": metrics.box.map75.tolist(),
    "map_all_class": metrics.box.maps.tolist()
  }
  
  with open (f"{project}/{name}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)

if __name__ == '__main__':
    main()