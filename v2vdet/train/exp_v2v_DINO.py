import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  device = int(cuda_devices.split(',')[0])
else:
  device = 0

from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_DINO

def main():
  data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"],
        ),
    val=dict(yolo_data=["v2vdet_ultralytics/cfg/datasets/lvis_min.yaml"]),
  )
  ckpt_name = 'ckpt/yolov8s-world.pt'
  model = V2V_DINO("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")

  model._load(ckpt_name, task='task')

  result = model.train(data=data,
                       batch=16,
                       epochs=2,
                       device=device,
                       project='training_result',
                       name=ckpt_name,
                       workers=8,
                       save_period=1,
                       cache=False,
                       exist_ok=True,
                       plots=False)

if __name__ == '__main__':
  main()
