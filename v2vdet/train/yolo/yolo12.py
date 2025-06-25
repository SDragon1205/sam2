from ultralytics import YOLO
import os, sys
import logging

cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_devices is not None:
  device = cuda_devices
else:
  device = 0

# Load a COCO-pretrained YOLO12n model
model = YOLO("yolo12s.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="v2vdet_ultralytics/cfg/datasets/coco8.yaml", 
                      epochs=100, 
                      imgsz=640,
                      device=device)

# Run inference with the YOLO12n model on the 'bus.jpg' image
results = model("path/to/bus.jpg")