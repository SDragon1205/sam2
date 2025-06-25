import os, sys
import logging
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.append(project_root)

import wandb
from wandb.integration.ultralytics import add_wandb_callback

from v2vdet.v2vdet_ultralytics.models.v2vdet import V2V_Template_YOLO_Backbone_Share_Param
import torch

def main():

  project = "validation/result/v2vdet_template_yolov8s_backbone"
  name = "lvis_v2vdet_V2V_Template_YOLO_Backbone_Share_Param"

  ckpt_name = 'training_result/v2vdet/lvis_v2vdet_V2V_Template_YOLO_Backbone_Share_Param/weights/best.pt'
  model = V2V_Template_YOLO_Backbone_Share_Param("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")

  model._load(ckpt_name, task='task')
  model.training=False
  # model.requires_grad_(False)

  # with torch.inference_mode():
  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/coco.yaml",
                    batch=16,
                    save_json=False,
                    half=True,
                    plots = False,
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
