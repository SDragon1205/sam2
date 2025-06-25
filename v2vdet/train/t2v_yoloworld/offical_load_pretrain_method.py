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
  YOLO_World = YOLOWorld("yolov8s-world.pt")
  # model = YOLOWorld("yolov8s-world.pt")
  # ckpt = torch.load(ckpt_name)
  # YOLO_World.load_state_dict(ckpt['model'])

  formatted_time = datetime.now().strftime("%Y%m%d_%H_%M_%S")
  result = YOLO_World.train(data=data, 
                            batch=16, 
                            epochs=20, 
                            trainer=WorldTrainerFromScratch, 
                            project='exp_t2v_yolo_world_train', 
                            name="resume_True_Load_With_Offical_Method",
                            workers=0,
                            save_period=1,
                            cache=False,
                            exist_ok=True,
                            resume=True,
                            plots=True)
  
if __name__ == '__main__':
  main()