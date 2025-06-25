import os, sys
import logging
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)

import wandb
from wandb.integration.ultralytics import add_wandb_callback

from v2vdet.v2vdet_ultralytics.models.v2vdet import V2V_Template_YOLO_Backbone
import torch

def main():
  
  project = "validation/result/v2vdet_template_yolov8s_backbone"
  name = "lvis_v2vdet_template_yolov8s_backbone_finetuning_template_backbone_layer9"
  
  ckpt_name = 'training_result/v2vdet/lvis_v2vdet_template_yolov8s_backbone_finetuning_template_backbone_layer9/weights/best.pt'
  model = V2V_Template_YOLO_Backbone("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
  
  wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
  wandb.init(project='v2vdet_validation', name=name, config=model)
  # add_wandb_callback(model, enable_model_checkpointing=True)

  model._load(ckpt_name, task='task')
  model.training=False

  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/coco.yaml",
                    batch=256,
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
  
  wandb.log(metrics_dict)
  
  with open (f"{project}/{name}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
  
  wandb.finish()

if __name__ == '__main__':
    main()