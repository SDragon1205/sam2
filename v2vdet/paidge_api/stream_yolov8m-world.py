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

    logging.warning("You didn't specify the return type, returning xywhn format.")
    bboxes = results[0].boxes.xyxyn.to('cpu').numpy().tolist()
    centers = [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in bboxes]
    
    result_dict = {
        'cls': results[0].boxes.cls.to('cpu').numpy().tolist(),
        'orig_shape': [c for c in results[0].orig_shape],
        'bbox': bboxes,
        'centers': centers
    }
    if (save_json):
      with open(json_name, 'w') as f:
        json.dump(result_dict, f, indent=2)
    return result_dict



t2vdet = t2vdet_class()
t2vdet.set_classes(['Keyboard', 'Person', 'Computer', 'Bed', 'Oven', "Bottle"])

@app.route('/detect_items', methods=['POST'])
def detect_items():
  if 'file' not in request.files:
      return jsonify({'error': 'No file part'}), 400

  file = request.files['file']
  if file:
    bytes_io = BytesIO()
    file.save(bytes_io)
    bytes_io.seek(0)
  
  result = t2vdet.predict(BytesIO_Obj = bytes_io, save_json=True)
  # print(result)
  return jsonify({'points': result['centers'], 'bbox': result['bbox']})
  # return result['centers']



if __name__ == "__main__":
    # Start Flask Server
    app.run(host="0.0.0.0", port=8089, debug=False, use_reloader=False)
