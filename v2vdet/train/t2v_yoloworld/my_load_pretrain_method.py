import os, sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import wandb
from wandb.integration.ultralytics import add_wandb_callback

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from datetime import datetime
from v2vdet.v2vdet_ultralytics import YOLOWorld
from v2vdet.v2vdet_ultralytics import v2vdet_model
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch, v2vTrainerFromScratch
import torch

# from enhanced_debug import debugger

# # 設置除錯環境
# debugger.setup()

# @debugger

def main():
  data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis.yaml"],
        # yolo_data=["v2vdet_ultralytics/cfg/datasets/Objects365.yaml"],
        # grounding_data=[
        #     dict(
        #         img_path=f"{project_root}/DATASET/flickr30k/images",
        #         json_file=f"{project_root}/DATASET/flickr30k/final_flickr_separateGT_train.json",
        #     ),
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
  YOLO_World = YOLOWorld("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
  # model = YOLOWorld("yolov8s-world.pt")
  # ckpt = torch.load(ckpt_name)
  # YOLO_World.load_state_dict(ckpt['model'])
  YOLO_World._load(ckpt_name)
  
  project_name = "lvis_yolo_world"
  name = datetime.now().strftime("%Y%m%d_%H_%M_%S")
  
  formatted_time = datetime.now().strftime("%Y%m%d_%H_%M_%S")
  wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
  wandb.init(job_type="train", project=project_name, name=name, config=YOLO_World)
  # add_wandb_callback(model=YOLO_World, enable_model_checkpointing=True)

  result = YOLO_World.train(data=data, 
                            batch=16, 
                            epochs=10, 
                            trainer=WorldTrainerFromScratch, 
                            project=project_name, 
                            name=name,
                            workers=8,
                            close_mosaic=0,
                            save_period=1,
                            cache=False,
                            exist_ok=True,
                            plots=True)
  
if __name__ == '__main__':
  main()
