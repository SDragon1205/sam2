import os, sys
import logging
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.append(project_root)

import wandb
from wandb.integration.ultralytics import add_wandb_callback

from v2vdet.v2vdet_ultralytics.models.v2vdet import V2V_With_MultiScale_SAVPE
import torch

MODEL_YAML = "v2vdet_ultralytics/cfg/models/v2v/v9/yolov9s-v2v-multiscale_135.yaml"

def main():

  project = "validation/result/SAVPE_v2v"
  name = "BrainTumor_v9s_SAVPE_SigLIP2_FT_multi_layer_123_LVIS"

  ckpt_name = 'v2v_training_result/v9s_SAVPE_SigLIP2_FT_multi_layer_123_LVIS/weights/best.pt'
  model = V2V_With_MultiScale_SAVPE(MODEL_YAML, task='detect')

  wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
  wandb.init(project='SAVPE_v2v', name=name, config=model)
  # add_wandb_callback(model, enable_model_checkpointing=True)

  model._load(ckpt_name, task='task')
  model.training=False
  model.eval()

  with torch.inference_mode():
    model.info()
    metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/brain-tumor.yaml",
                      batch=16,
                      save_json=True,
                      half=True,
                      plots=True,
                      project=project,
                      name=name,
                      exist_ok=True,
                      )
    
  metrics_dict = {
    "map50_95": metrics.box.map.tolist(),
    "map50": metrics.box.map50.tolist(),
    "map75": metrics.box.map75.tolist(),
    "map_all_class": metrics.box.maps.tolist()
  }

  wandb.log(metrics_dict)

  with open (f"{project}/{name}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)

  wandb.finish()

if __name__ == '__main__':
    main()
