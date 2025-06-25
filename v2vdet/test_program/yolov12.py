from ultralytics import YOLO

# Load a COCO-pretrained YOLOv12n model
model = YOLO("ultralytics/cfg/models/12/yolo12.yaml")
# model = YOLO('yolo12s.pt')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="v2vdet_ultralytics/cfg/datasets/coco8.yaml", epochs=10, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model("image/zebra_another.png", embed=[8])