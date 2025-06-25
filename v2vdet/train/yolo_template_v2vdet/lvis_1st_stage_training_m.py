import os, sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import wandb
# from wandb.integration.ultralytics import add_wandb_callback
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics.utils.callbacks.wb import on_train_epoch_end, on_fit_epoch_end, on_train_end

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from datetime import datetime
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_Template_YOLO_Backbone_Share_Param
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch, v2vTrainerFromScratch, v2vWorldTrainerFromScratch
import torch
from ultralytics import YOLO

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  device = int(cuda_devices.split(',')[0])
else:
  device = 0

def main():
  data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis.yaml"],
    ),
    val=dict(yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis.yaml"]),
  )

  project_name = 'v2vdet'
  ckpt_name = 'training_result/v2vdet_Only_Train_Linear_Layer/V2V_Template_YOLOv8m_Backbone_Share_Param_Train_Linear_Layer/weights/best.pt'
  name = f"bb_share_yolov8m_param_LVIS_training"
  
  model = V2V_Template_YOLO_Backbone_Share_Param("v2vdet_ultralytics/cfg/models/v8/yolov8m-world.yaml")
  model._load(ckpt_name, task='task')
  # model.save(f"exp_train_world_v2vdet_yolov8s-world.pt")

  # model = YOLO("yolo11n.pt")
  # model = v2vYOLOWorld("exp_train_world_v2vdet_yolov8s-world.pt")

  wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
  wandb.init(job_type="train", project=project_name, name=name, config=model)
  # add_wandb_callback(model=model, enable_model_checkpointing=False)
    # model.add_callback(on_fit_epoch_end)
    # model.add_callback(on_train_epoch_end)
    # model.add_callback(on_train_end)
  
  # wandb.log(argum_dict)
  
  result = model.train(
    project=f"training_result/{project_name}",
    name=name,
    data=data,
    batch=32,
    epochs=10,
    device=device,
    workers=16,
    close_mosaic=20,
    save_period=1,
    cache=False,
    exist_ok=True,
    plots=True,
    )
  # data = "v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"
  # result = model.train(data=data,
  #                     batch=32, 
  #                     epochs=20,
  #                     plots=True,
  #                     resume=True)

  wandb.finish()

if __name__ == '__main__':
  main()
