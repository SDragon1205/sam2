from v2vdet.v2vdet_ultralytics.models.yolo.model import YOLOE_v2v
import torch

# Create a YOLOE model
model = YOLOE_v2v("v2vdet_ultralytics/cfg/models/11/yoloe-11m.yaml", task='detect')  # or select yoloe-m/l-seg.pt for different sizes

state = torch.load("ckpt/offical_yoloe/yoloe-11m-seg.pt")
model.load(state["model"])

model = model.to('cuda')

# Conduct model validation on the COCO128-seg example dataset
with torch.inference_mode():
  # metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/coco128-seg.yaml", 
  #                     workers=0,
  #                     load_vp=True, 
  #                     refer_data="v2vdet_ultralytics/cfg/datasets/coco.yaml")
  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/SA_V.yaml",
                      task='detect',
                      name = "SA_V_yoloe-11m-seg",
                      batch = 1,
                      device='cuda',
                      load_vp=True, 
                      workers=16,
                      exist_ok = True)
    