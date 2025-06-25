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
  
  project = "validation/result/v2vdet/coco_val_result"
  name = "yolov8s-world_remove_i_pooling_attn"
  
  # ckpt_name = 'ckpt/yolov8s-world.pt'
  
  ckpt_name = 'training_result/lvis_v2vdet_remove_i_pooling_attn/lvis_v2vdet_remove_i_pooling_attn_20250113_0018/weights/best.pt'
  wandb_name = name
  model = v2vYOLOWorld("v2vdet_ultralytics/cfg/models/v8/yolov8s-world_remove_i_pooling_attn.yaml")
  

  # add_wandb_callback(model, enable_model_checkpointing=True)

  model._load(ckpt_name, task='task')
  model.training=False

  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/coco.yaml",
                    batch=64,
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
  
  # wandb.finish()

if __name__ == '__main__':
    main()