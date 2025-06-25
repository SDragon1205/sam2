import os, sys
import logging
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)

import wandb
from wandb.integration.ultralytics import add_wandb_callback

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from v2vdet.v2vdet_ultralytics.models.v2vdet import v2vYOLOWorld
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch, v2vTrainerFromScratch
import torch

def main():
  
  scale = 's'
  project = "validation/result/YOLO_World"
  name = f"v8{scale}_world_v1"
  
  # ckpt_name = 'ckpt/yolov8s-world.pt'
  
  ckpt_name = f'ckpt/offical_YOLO_World/yolov8{scale}-world.pt'

  model = v2vYOLOWorld(f"v2vdet_ultralytics/cfg/models/v8/yolov8{scale}-world.yaml")
  
  wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
  wandb.init(project='YOLO_World_COCO_Val', name=name)
  # add_wandb_callback(model, enable_model_checkpointing=True)

  model._load(ckpt_name, task='task')
  model.training=False

  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/coco.yaml",
                    batch=16,
                    save_json=True,
                    half=True,
                    plots = True,
                    project=project,
                    name=name,
                    exist_ok=True
                    )
  metrics_dict = {
    "map50_95": metrics.box.map.tolist(),
    "map50": metrics.box.map50.tolist(),
    "map75": metrics.box.map75.tolist(),
    "map_all_class": metrics.box.maps.tolist()
  }
  wandb.log(metrics_dict)
  
  with open (f"{project}/{name}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
  
  wandb.finish()

if __name__ == '__main__':
    main()