from v2vdet.v2vdet_ultralytics.models.yolo.model import YOLOE_v2v
import torch
import wandb
import os
import json

# Create a YOLOE model
# model = YOLOE_v2v("v2vdet_ultralytics/cfg/models/11/yoloe-11m.yaml", task='detect')  # or select yoloe-m/l-seg.pt for different sizes

# state = torch.load("ckpt/offical_yoloe/yoloe-11m-seg.pt")
# model.load(state["model"])

# model = model.to('cuda')

# Conduct model validation on the COCO128-seg example dataset
with torch.inference_mode():
  # metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/coco128-seg.yaml", 
  #                     workers=0,
  #                     load_vp=True, 
  #                     refer_data="v2vdet_ultralytics/cfg/datasets/coco.yaml")
  # metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/SA_V.yaml",
  #                     task='detect',
  #                     name = "SA_V_yoloe-11m-seg",
  #                     batch = 1,
  #                     device='cuda',
  #                     load_vp=True, 
  #                     workers=16,
  #                     exist_ok = True)

  SCALE = 'm'
  MODEL_YAML = f"v2vdet_ultralytics/cfg/models/v8/yoloe-v8{SCALE}.yaml"
  CKPT_NAME = f"ckpt/offical_yoloe/yoloe-v8{SCALE}-seg.pt"
  PROJECT_NAME = 'YOLOE'
  DATASET_LIST=['coco', 'VisDrone', 'VOC', 'brain-tumor', 'SA_V']
  NAME = f"YOLOE_v8{SCALE}"
  
  for dataset_name in DATASET_LIST:
    print(f"Evaluating on {dataset_name} dataset...")
    model = YOLOE_v2v(MODEL_YAML, task='detect')
    state = torch.load(CKPT_NAME)
    model.load(state["model"])
    name = f"{dataset_name}_{NAME}"
  
    wandb.init(project=PROJECT_NAME, name=name, config=model)
  
    project_folder_only = PROJECT_NAME
    project_folder = os.path.join(project_folder_only, name)

    # Eval
    DATASET_PATH = os.path.join(project_folder, dataset_name)
    if not os.path.exists(path=DATASET_PATH):
      os.makedirs(DATASET_PATH)

    metrics = model.val(data=f"v2vdet_ultralytics/cfg/datasets/{dataset_name}.yaml",
                      batch=1,
                      save_json=True,
                      half=True,
                      plots = True,
                      project=DATASET_PATH,
                      name=name,
                      exist_ok=True
                      )
    metrics_dict = {
      "DATASET": dataset_name,
      "map50_95": metrics.box.map.tolist(),
      "map50": metrics.box.map50.tolist(),
      "map75": metrics.box.map75.tolist(),
      "map_all_class": metrics.box.maps.tolist()
    }
  
    wandb.log(metrics_dict)
  
    with open (f"{DATASET_PATH}/metrics.json", 'w') as f:
      json.dump(metrics_dict, f, indent=2)
    
    wandb.finish()
    del model