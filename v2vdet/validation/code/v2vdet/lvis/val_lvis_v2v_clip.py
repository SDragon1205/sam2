import os, sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

import wandb
from wandb.integration.ultralytics import add_wandb_callback

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from v2vdet.v2vdet_ultralytics.models.v2vdet import YOLOWorld, v2vYOLOWorld
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch, v2vTrainerFromScratch
import torch
import json
import shutil
import yaml

def main():
  project = "validation/v2vdet/lvis"
  name = "lvis_val"
  data_yaml = "v2vdet_ultralytics/cfg/datasets/lvis.yaml"
  
  if (os.path.exists(f"{project}/{name}")):
    shutil.rmtree(f"{project}/{name}")
  
  ckpt_name = 'ckpt/yolov8s-world.pt'
  model = v2vYOLOWorld("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
  # ckpt = torch.load(ckpt_name)
  model._load(ckpt_name, task='task')
  # model.load_state_dict(ckpt['model'])
  # with open("DATASET/lvis/lvis_categories.json") as f:
  #   lvis_class = json.load(f)
  
  with open(data_yaml, 'r', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)
  lvis_class = [yaml_data['names'][idx] for idx in yaml_data['names']]
  model.set_classes(lvis_class)
  model.training=False
  # metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/lvis.yaml",
  #                     plots = True,
  #                     project=project,
  #                     name=name
  #                     )
  
  metrics = model.val(data=data_yaml,
                      batch=32,
                      plots=True,
                      save_json=True,
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
  
  # from ultralytics import YOLO

  # # Create a YOLO-World model
  # model = YOLO("yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes

  # # Conduct model validation on the COCO8 example dataset
  # metrics = model.val(data="coco8.yaml")

if __name__ == '__main__':
    main()