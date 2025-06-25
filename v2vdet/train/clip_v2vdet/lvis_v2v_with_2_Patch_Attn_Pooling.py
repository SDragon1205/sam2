import os, sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import wandb
# from wandb.integration.ultralytics import add_wandb_callback
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics.utils.callbacks.wb import on_train_epoch_end, on_fit_epoch_end, on_train_end

from datetime import datetime
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_with_2_Patch_Attn_Pooling
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
  
  current_time = datetime.now().strftime("%Y%m%d_%H%M")

  project_name = 'lvis_V2V_with_2_Patch_Attn_Pooling_load_pretrain_train_attn_pooling_finetune_yolo_model'
  ckpt_name = 'training_result/lvis_v2vdet_world_train_project_original/train_20250118_2307/weights/best.pt'
  name = f"train_{current_time}"
  
  argum_dict = {
            'rotation_range': (-30, 30),  
            'scale_range': (0.8, 1.2),    
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.8, 1.2),
            'prob': 0.5,                  
            'global_prob': 1            
            }
  
  model = V2V_with_2_Patch_Attn_Pooling("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
  # model = YOLOWorld("yolov8s-world.pt")
  # ckpt = torch.load(ckpt_name)
  # model.load_state_dict(ckpt['model'])
  model._load(ckpt_name, task='task')
  # model.save(f"exp_train_world_v2vdet_yolov8s-world.pt")

  # model = YOLO("yolo11n.pt")
  # model = v2vYOLOWorld("exp_train_world_v2vdet_yolov8s-world.pt")

  wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
  wandb.init(job_type="train", project=project_name, name=name, config=model)
  # add_wandb_callback(model=model, enable_model_checkpointing=True)
    # model.add_callback(on_fit_epoch_end)
    # model.add_callback(on_train_epoch_end)
    # model.add_callback(on_train_end)
  
  wandb.log(argum_dict)
  
  result = model.train(
    project=f"training_result/{project_name}",
    name=name,
    data=data,
    batch=32,
    epochs=10,
    device=device,
    workers=4,
    close_mosaic=0,
    save_period=5,
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
