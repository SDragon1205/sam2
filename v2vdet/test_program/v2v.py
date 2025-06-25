from v2vdet.v2vdet_ultralytics.models.v2vdet import V2V_Template_YOLO_Backbone_Share_Param
import time
import numpy as np
from PIL import Image
from v2vdet.v2vdet_ultralytics.utils.misc import resize_with_padding

ckpt_name = 'training_result/v2vdet/lvis_v2vdet_V2V_Template_YOLO_Backbone_Share_Param/weights/best.pt'
model = V2V_Template_YOLO_Backbone_Share_Param("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")

model._load(ckpt_name, task='task')
# model = YOLO('yolo12s.pt')

# Display model information (optional)
model.info()

# crop_img=['image/crop_cat.jpg']
crop_img = ['image/crop_zebra.jpg']
crop_img_pil = [resize_with_padding(Image.open(img).convert(mode="RGB")) for img in crop_img]

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="v2vdet_ultralytics/cfg/datasets/coco8.yaml", epochs=10, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image

model.predict(np.zeros((640, 640, 3), dtype=np.uint8))
model.set_classes(classes=[idx for idx in range(len(crop_img))], crop_img=crop_img_pil)
# results = model("image/zebra_another.png")
results = model.predict('image/zebra.jpg')
start = time.time()

for i in range(10):
  start = time.time()
  results = model.predict('image/zebra.jpg')

result_pic = results[0].save('gg123.png')
  
print(f"Time: {time.time()-start}, FPS: {1/(time.time()-start)}")