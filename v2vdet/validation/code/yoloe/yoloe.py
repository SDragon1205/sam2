from v2vdet.v2vdet_ultralytics.models.yolo.model import YOLOE_v2v
from ultralytics import YOLOE
import torch

# Create a YOLOE model
model = YOLOE("yoloe-11s-seg.pt", task='detect')  # or select yoloe-m/l-seg.pt for different sizes
model = model.to('cuda')

# Conduct model validation on the COCO128-seg example dataset
with torch.inference_mode():
  # metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/coco128-seg.yaml", 
  #                     workers=0,
  #                     load_vp=True, 
  #                     refer_data="v2vdet_ultralytics/cfg/datasets/coco.yaml")
  
  # Batch size only can set to 1!!!

  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/brain-tumor.yaml",
                    batch = 1,
                    device='cuda',
                    load_vp=True, 
                    workers=8,
                    exist_ok = True)