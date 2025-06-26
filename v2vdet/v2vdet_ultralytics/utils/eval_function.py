import wandb
import os
import json
import torch

def eval_function(MODEL, MODEL_YAML, CKPT_NAME, NAME, PROJECT_NAME, BATCH_SIZE=32, DATASET_LIST=['coco', 'VisDrone', 'VOC', 'brain-tumor', 'SA_V'], exist_ok=False):
  """
  Evaluate the model on different datasets and log the results to wandb.
  """
  wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
  
  
  for dataset_name in DATASET_LIST:
    print(f"Evaluating on {dataset_name} dataset...")
    model = MODEL(MODEL_YAML)
    
    # wandb_enable = True

    ckpt_name = CKPT_NAME
    name = f"{dataset_name}_{NAME}"
  
    wandb.init(project=PROJECT_NAME, name=name, config=model)
  
    # weight = torch.load(ckpt_name)
    # model.load(weight['model'])
    model._load(ckpt_name, task='task')
    model.training = False  
  
    project_folder_only = PROJECT_NAME
    project_folder = os.path.join(project_folder_only, name)

    # Eval
    DATASET_PATH = os.path.join(project_folder, dataset_name)
    if not os.path.exists(path=DATASET_PATH):
      os.makedirs(DATASET_PATH)

    metrics = model.val(data=f"v2vdet_ultralytics/cfg/datasets/{dataset_name}.yaml",
                      batch=BATCH_SIZE,
                      save_json=True,
                      half=True,
                      plots = True,
                      project=DATASET_PATH,
                      name=name,
                      exist_ok=exist_ok
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
  
  return 
  
  name = f"VisDrone_{NAME}"
  
  # wandb.init(project=PROJECT_NAME, name=name, config=model)
    
  # Eval VisDrone
  VisDrone_PATH = os.path.join(project_folder, 'VisDrone')
  if not os.path.exists(path=VisDrone_PATH):
    os.makedirs(VisDrone_PATH)
    
  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/VisDrone.yaml",
                    batch=BATCH_SIZE,
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
  
  # wandb.log(metrics_dict)
  
  with open (f"{VisDrone_PATH}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
    
  # wandb.finish()
  
  name = f"VOC_{NAME}"
  
  # wandb.init(project=PROJECT_NAME, name=name, config=model)
    
  # Eval PascalVOC
  VOC_PATH = os.path.join(project_folder, 'VOC')
  if not os.path.exists(path=VOC_PATH):
    os.makedirs(VOC_PATH)

  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/VOC.yaml",
                    batch=BATCH_SIZE,
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
  
  # wandb.log(metrics_dict)
  
  with open (f"{VOC_PATH}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
    
  # Eval Brain-Tumor
  
  # wandb.finish()
  
  name = f"Brain_Tumor_{NAME}"
  
  # wandb.init(project=PROJECT_NAME, name=name, config=model)
  
  BRAIN_PATH = os.path.join(project_folder, 'Brain-Tumor')
  if not os.path.exists(path=BRAIN_PATH):
    os.makedirs(BRAIN_PATH)

  metrics = model.val(data="v2vdet_ultralytics/cfg/datasets/brain-tumor.yaml",
                    batch=BATCH_SIZE,
                    save_json=True,
                    half=True,
                    plots = True,
                    project=BRAIN_PATH,
                    name=name
                    )
  
  metrics_dict = {
    "DATASET": "Brain-Tumor",
    "map50_95": metrics.box.map.tolist(),
    "map50": metrics.box.map50.tolist(),
    "map75": metrics.box.map75.tolist(),
    "map_all_class": metrics.box.maps.tolist()
  }
  
  # wandb.log(metrics_dict)
  
  with open (f"{BRAIN_PATH}/metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
  
  # wandb.finish()