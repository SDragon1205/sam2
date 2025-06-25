import os, sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import wandb
from wandb.integration.ultralytics import add_wandb_callback
# from v2vdet.v2vdet_ultralytics.utils.wandb_callback import add_wandb_callback

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from v2vdet.v2vdet_ultralytics.models.v2vdet.model import v2vdet_model, v2vYOLOWorld
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch, v2vTrainerFromScratch, v2vWorldTrainerFromScratch
import torch
from ultralytics import YOLO

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
    device = int(cuda_devices.split(',')[0])  # 使用第一個可見的 GPU
else:
    device = 0  # 預設值

def main():
  data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/Objects365.yaml"],
        grounding_data=[
            dict(
                img_path=f"{project_root}/DATASET/flickr30k/images",
                json_file=f"{project_root}/DATASET/flickr30k/final_flickr_separateGT_train.json",
            ),]
            # dict(
            #     img_path="../datasets/GQA/images",
            #     json_file="../datasets/GQA/final_mixed_train_no_coco.json",
            # ),
        # ],
    ),
    val=dict(yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis.yaml"]),
  )
  wandb_enable = False

  ckpt_name = 'ckpt/yolov8s-world.pt'
  model = v2vYOLOWorld("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
  project_name = 'Objects365_flickr30k_v2vdet_world_train_project'
  # model = YOLOWorld("yolov8s-world.pt")
  # ckpt = torch.load(ckpt_name)
  # model.load_state_dict(ckpt['model'])
  model._load(ckpt_name, task='detect')
  # model.save(f"exp_train_world_v2vdet_yolov8s-world.pt")
  
  # model = YOLO("yolo11n.pt")
  # model = v2vYOLOWorld("exp_train_world_v2vdet_yolov8s-world.pt")

  if (wandb_enable):
    wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
    wandb.init(project=project_name, name=ckpt_name)
    add_wandb_callback(model, enable_model_checkpointing=True)
  
  result = model.train(data=data,
                       batch=32, 
                       epochs=5,
                       device=device,
                       project=project_name,
                       name=ckpt_name,
                       workers=0,
                       save_period=1, 
                       cache=False, 
                       exist_ok=False, 
                       plots=True)
  
  # data = "v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"
  # result = model.train(data=data,
  #                     batch=32, 
  #                     epochs=20,
  #                     plots=True,
  #                     resume=True)
  
  if (wandb_enable): wandb.finish()

if __name__ == '__main__':
  main()
