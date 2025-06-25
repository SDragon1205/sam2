from ultralytics import YOLOWorld
import torch
import json
import logging

from PIL import Image
from flask import Flask, render_template, request, jsonify,abort, send_file, g, Response, stream_with_context
import uuid
from pathlib import Path
from io import BytesIO
import base64
import json, requests
import numpy as np

app = Flask(__name__)

class t2vdet_class():
  def __init__ (self):
    FORMAT = '%(asctime)s %(filename)s %(levelname)s:%(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    
    self.model = YOLOWorld("yolov8m-world")
    
    # initialize
    self.initialize_model()
    logging.info(f'Model Initialized Done')
  
  @torch.inference_mode()
  def set_classes(self, pred_cls: list):
    self.pred_cls = pred_cls
    self.model.set_classes(self.pred_cls)
  
  @torch.inference_mode()
  def initialize_model(self):
    self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8))
  
  @torch.inference_mode()
  def predict(self, BytesIO_Obj: BytesIO, save_json = False, json_name = 'result.json', xywh = False, xywhn = False, xyxy = False, xyxyn = False):
    
    pil_image = Image.open(BytesIO_Obj)
    results = self.model.predict(pil_image)

    logging.info('Predicting Done')
    if (xywh):
      result_dict = {
        'cls': results[0].boxes.cls.to('cpu').numpy().tolist(),
        'orig_shape': [c for c in results[0].orig_shape],
        'bbox': results[0].boxes.xywh.to('cpu').numpy().tolist()
      }
      if (save_json):
        with open(json_name, 'w') as f:
          json.dump(result_dict, f, indent=2)
        
      return result_dict
    elif (xywhn):
      result_dict = {
        'cls': results[0].boxes.cls.to('cpu').numpy().tolist(),
        'orig_shape': [c for c in results[0].orig_shape],
        'bbox': results[0].boxes.xywhn.to('cpu').numpy().tolist()
      }
      if (save_json):
        with open(json_name, 'w') as f:
          json.dump(result_dict, f, indent=2)
          
      return result_dict
    elif (xyxy):
      result_dict = {
        'cls': results[0].boxes.cls.to('cpu').numpy().tolist(),
        'orig_shape': [c for c in results[0].orig_shape],
        'bbox': results[0].boxes.xyxy.to('cpu').numpy().tolist()
      }
      if (save_json):
        with open(json_name, 'w') as f:
          json.dump(result_dict, f, indent=2)
      return result_dict
    
    elif (xyxyn):
      result_dict = {
        'cls': results[0].boxes.cls.to('cpu').numpy().tolist(),
        'orig_shape': [c for c in results[0].orig_shape],
        'bbox': results[0].boxes.xyxyn.to('cpu').numpy().tolist()
      }
      if (save_json):
        with open(json_name, 'w') as f:
          json.dump(result_dict, f, indent=2)
      return result_dict
    
    else:
      logging.warning(f"You don't specify the return type, return xywhn format.")
      result_dict = {
        'cls': results[0].boxes.cls.to('cpu').numpy().tolist(),
        'orig_shape': [c for c in results[0].orig_shape],
        'bbox': results[0].boxes.xywhn.to('cpu').numpy().tolist()
      }
      if (save_json):
        with open(json_name, 'w') as f:
          json.dump(result_dict, f, indent=2)
      return result_dict

if __name__ == "__main__":
  t2vdet = t2vdet_class()
  t2vdet.set_classes(['person'])
  while True:
    with open('gg4997.jpg', 'rb') as f:
      bytes_io = BytesIO(f.read())
    result = t2vdet.predict(BytesIO_Obj = bytes_io, save_json=True)

# if __name__ == "__main__":
#     # Start Flask Server
#     app.run(host="0.0.0.0", port=8088, debug=False, use_reloader=False)