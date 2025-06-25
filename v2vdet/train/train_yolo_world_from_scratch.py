import os, sys
import logging
import wandb
from wandb.integration.ultralytics import add_wandb_callback

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
# from ultralytics import YOLOWorld

from v2vdet.v2vdet_ultralytics import YOLOWorld
from v2vdet.v2vdet_ultralytics.models.v2vdet.train import WorldTrainerFromScratch
import torch

if __name__ == "__main__":
  data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"],
        # yolo_data=["v2vdet_ultralytics/cfg/datasets/Objects365.yaml"],
        #grounding_data=[
        #    dict(
        #        img_path=f"{project_root}/DATASET/flickr30k/images",
        #        json_file=f"{project_root}/DATASET/flickr30k/final_flickr_separateGT_train.json",
        #    ),
            # dict(
            #     img_path="../datasets/GQA/images",
            #     json_file="../datasets/GQA/final_mixed_train_no_coco.json",
            # ),
       # ],
    ),
    val=dict(yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis.yaml"]),
  )

  model = YOLOWorld("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
  # model = YOLOWorld("yolov8s-world.pt")
  ckpt = torch.load('yolov8s-world.pt')
  model.load_state_dict(ckpt['model'])

  result = model.train(data=data, workers=0, batch=4, epochs=20, trainer=WorldTrainerFromScratch)
