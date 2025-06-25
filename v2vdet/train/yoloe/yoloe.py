from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer, YOLOESegVPTrainer, YOLOEVPTrainer

model = YOLOE("yoloe-11s-seg.pt", task='detect')

data = dict(
    train=dict(
        yolo_data=["v2vdet_ultralytics/cfg/datasets/coco8.yaml"],
    ),
    val=dict(yolo_data=["v2vdet_ultralytics/cfg/datasets/coco8.yaml"]),
)

model.train(
    data="v2vdet_ultralytics/cfg/datasets/coco8.yaml",
    epochs=5,
    close_mosaic=10,
    batch=32,
)