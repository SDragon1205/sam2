import os, sys
import logging
import json
import wandb

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

import wandb
from wandb.integration.ultralytics import add_wandb_callback

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from v2vdet.v2vdet_ultralytics.models.v2vdet import V2V_with_Patch_Attn_Pooling
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch, v2vTrainerFromScratch
import torch

def main():
  
  project = "validation/v2vdet/coco_val_result"
  name = "get_last_3rd_clip_layer"
  wandb_name = name
  
  # ckpt_name = 'ckpt/yolov8s-world.pt'
  
  ckpt_name = 'training_result/lvis_v2vdet_world_train_project/train_with_augment_template_with_attn_pooling/weights/best.pt'
  ckpt_name = "training_result/lvis_v2vdet_attn_pooling_finetune_from_cls_token/get_last_2nd_clip_layer_20241222_1802/weights/best.pt"
  ckpt_name = "training_result/lvis_v2vdet_attn_pooling_finetune_from_cls_token/get_last_3rd_clip_layer_20241222_1820/weights/best.pt"
  model = V2V_with_Patch_Attn_Pooling("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
  
  wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
  wandb.init(project='v2vdet_world_val', name=wandb_name)
  # add_wandb_callback(model, enable_model_checkpointing=True)

  model._load(weights=ckpt_name, task='task')
  model.training=False
  # wandb.watch(model)
  # ckpt = torch.load(ckpt_name)
  # model.model.attn_pooling.load_state_dict(ckpt['model'].attn_pooling.state_dict())
  
  metrics = model.val(data="coco.yaml",
                    batch=16,
                    workers=8,
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

if __name__ == '__main__':
    main()
