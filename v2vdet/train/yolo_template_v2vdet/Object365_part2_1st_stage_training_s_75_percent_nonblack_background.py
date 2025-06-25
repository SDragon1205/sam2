import os, sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import wandb
# from wandb.integration.ultralytics import add_wandb_callback
from datetime import datetime
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_Template_YOLO_Backbone_Share_Param

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  device = int(cuda_devices.split(',')[0])
else:
  device = 0

def main():
  data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/Objects365_2.yaml"],
    ),
    val=dict(yolo_data=["v2vdet_ultralytics/cfg/datasets/Objects365_2.yaml"]),
  )

  project_name = 'v2vdet'
  ckpt_name = 'training_result/v2vdet/Object365_part1_1st_stage_training_s_75_percent_nonblack_background/weights/best.pt'
  name = f"Object365_part2_1st_stage_training_s_75_percent_nonblack_background"

  model = V2V_Template_YOLO_Backbone_Share_Param("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
  model._load(ckpt_name, task='task')
  wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
  wandb.init(job_type="train", project=project_name, name=name, config=model)
  
  model.train(
    project=f"training_result/{project_name}",
    name=name,
    data=data,
    batch=32,
    epochs=1,
    device=device,
    workers=16,
    close_mosaic=20,
    save_period=1,
    cache=False,
    exist_ok=True,
    plots=True,
    )

  wandb.finish()

if __name__ == '__main__':
  main()
