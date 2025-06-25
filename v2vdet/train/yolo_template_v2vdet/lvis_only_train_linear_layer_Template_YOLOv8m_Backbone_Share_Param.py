import os, sys
import logging

import wandb
import json
# from wandb.integration.ultralytics import add_wandb_callback
# from ultralytics.utils.callbacks.wb import on_train_epoch_end, on_fit_epoch_end, on_train_end

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from datetime import datetime
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_Template_YOLO_Backbone_Share_Param_Train_Linear_Layer

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
  wandb_enable = True
  
  current_time = datetime.now().strftime("%Y%m%d_%H%M")

  project_name = 'v2vdet_Only_Train_Linear_Layer'
  ckpt_name = 'ckpt/lvis_v2vdet_V2V_Template_YOLO_Backbone_Share_Param.pt'
  name = f"V2V_Template_YOLOv8m_Backbone_Share_Param_Train_Linear_Layer"
  
  argum_dict = {
            'rotation_range': (-30, 30),  
            'scale_range': (0.8, 1.2),    
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.8, 1.2),
            'prob': 0.5,                  
            'global_prob': 1            
            }
  
  model = V2V_Template_YOLO_Backbone_Share_Param_Train_Linear_Layer("ultralytics/cfg/models/v8/yolov8m-world.yaml")
  # model = YOLOWorld("yolov8s-world.pt")
  # ckpt = torch.load(ckpt_name)
  # model.load_state_dict(ckpt['model'])
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
  
  project_folder = f"training_result/{project_name}"
  
  result = model.train(
    project=project_folder,
    name=name,
    data=data,
    batch=32,
    epochs=10,
    device=device,
    workers=16,
    close_mosaic=0,
    save_period=1,
    cache=False,
    exist_ok=True,
    plots=True,
    eval_batch_size = 2,
    gradient_checkpointing = False,
    gradient_accumulation_steps = 8
    )

  ckpt_name = f'{project_folder}/weights/best.pt'

  # Eval COCO
  COCO_PATH = os.path.join(project_folder, 'COCO')
  if not os.path.exists(path=COCO_PATH):
    os.makedirs(COCO_PATH)
  
  model._load(ckpt_name, task='task')
  model.training=False
  model=model.eval()

  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/coco.yaml",
                    batch=32,
                    save_json=True,
                    half=True,
                    plots = True,
                    project=COCO_PATH,
                    name=name
                    )
  metrics_dict = {
    "DATASET": "COCO2017",
    "map50_95": metrics.box.map.tolist(),
    "map50": metrics.box.map50.tolist(),
    "map75": metrics.box.map75.tolist(),
    "map_all_class": metrics.box.maps.tolist()
  }
  
  wandb.log(metrics_dict)
  
  with open (f"{COCO_PATH}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
    
  # Eval VisDrone
  VisDrone_PATH = os.path.join(project_folder, 'VisDrone')
  if not os.path.exists(path=VisDrone_PATH):
    os.makedirs(VisDrone_PATH)
  
  # model._load(ckpt_name, task='task')
  model.training=False
  model=model.eval()

  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/VisDrone.yaml",
                    batch=32,
                    save_json=True,
                    half=True,
                    plots = True,
                    project=VisDrone_PATH,
                    name=name
                    )
  
  metrics_dict = {
    "DATASET": "VisDrone",
    "map50_95": metrics.box.map.tolist(),
    "map50": metrics.box.map50.tolist(),
    "map75": metrics.box.map75.tolist(),
    "map_all_class": metrics.box.maps.tolist()
  }
  
  wandb.log(metrics_dict)
  
  with open (f"{VisDrone_PATH}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
    
  # Eval PascalVOC
  VOC_PATH = os.path.join(project_folder, 'VOC')
  if not os.path.exists(path=VOC_PATH):
    os.makedirs(VOC_PATH)
    
  model.training=False
  model=model.eval()

  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/VOC.yaml",
                    batch=32,
                    save_json=True,
                    half=True,
                    plots = True,
                    project=VOC_PATH,
                    name=name
                    )
  
  metrics_dict = {
    "DATASET": "VOC",
    "map50_95": metrics.box.map.tolist(),
    "map50": metrics.box.map50.tolist(),
    "map75": metrics.box.map75.tolist(),
    "map_all_class": metrics.box.maps.tolist()
  }
  
  wandb.log(metrics_dict)
  
  with open (f"{VOC_PATH}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)


  wandb.finish()

if __name__ == '__main__':
  main()
