from os import name
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

if __name__ == "__main__":

  model = YOLO("yolo11n.pt")

  project="yolov11"
  name="official"

  wandb.login(key='1f589445bc1e9e7d5e83c40ed692adb9d87bc0a1')
  wandb.init(job_type="train", project=project, name=name, config=model)
  add_wandb_callback(model=model, enable_model_checkpointing=True)

  model.train(data="coco.yaml", batch=16, epochs=5, project=project, name=name, plots=True)

  wandb.finish()