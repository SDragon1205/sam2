# from ultralytics import YOLOWorld

import os, sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import json
import logging
from PIL import Image
from flask import Flask, request, jsonify
import uuid
from io import BytesIO
import base64
import numpy as np
import threading
from collections import OrderedDict
import psycopg2
from psycopg2 import pool
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_Template_YOLO_Backbone, v2vYOLOWorld, V2V_multi_scale_clip
import cv2

app = Flask(__name__)

# Database configuration
DB_HOST = '127.0.0.1'
DB_PORT = '5432'
DB_NAME = 'paidge_db'
DB_USER = 'paidge'
DB_PASSWORD = 'si2si2'

# Initialize a connection pool
db_pool = psycopg2.pool.ThreadedConnectionPool(
    1, 20,  # min and max number of connections
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME
)

class v2vdet_class():
    def __init__(self):
        FORMAT = '%(asctime)s %(filename)s %(levelname)s:%(message)s'
        logging.basicConfig(level=logging.DEBUG, format=FORMAT)
        
        # self.model = V2V_Template_YOLO_Backbone("yolov8s-world.yaml") 
        self.model = V2V_multi_scale_clip("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
        self.ckpt_path = "training_result/v2vdet/lvis_V2V_multi_scale_clip_train_all_from_2_4_6_8_20250124_0034/weights/best.pt"
        
        self.model._load(weights=self.ckpt_path, task='detect')
        self.model.training=False
        self.iou = 0.6
        
        # Initialize
        self.initialize_model()
        # logging.info('Model Initialized Done')
      
    @torch.inference_mode()
    def set_classes(self, query_img: [str], query_class_name: [str]):
        self.query_img = query_img
        self.query_img_emb = [Image.open(img) for img in query_img]
        self.class_name = query_class_name
        self.model.set_classes(classes=self.class_name, crop_img=self.query_img_emb)
      
    @torch.inference_mode()
    def initialize_model(self):
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), iou=self.iou)
    
    @torch.inference_mode()
    def predict(self, BytesIO_Obj: BytesIO, save_json=False, json_name='result.json'):
        pil_image = Image.open(BytesIO_Obj)
        results = self.model.predict(pil_image, iou=self.iou)

        bboxes = results[0].boxes.xyxyn.to('cpu').numpy().tolist()
        centers = [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in bboxes]
        # print('cls', results[0].boxes.cls.to('cpu').numpy().tolist())
        # Extract class indices and convert to list
        class_indices = results[0].boxes.cls.to('cpu').numpy().tolist()
        
        # Map class indices to class names using self.names
        class_names = [self.model.names[int(c)] for c in class_indices]
        # print('class_names', class_names)
    
        result_dict = {
            # 'cls': results[0].boxes.cls.to('cpu').numpy().tolist(),
            'cls': class_names,
            'orig_shape': list(results[0].orig_shape),
            'bbox': bboxes,
            'centers': centers
        }
        if save_json:
            with open(json_name, 'w') as f:
                json.dump(result_dict, f, indent=2)
        return result_dict

# LRU Cache Implementation
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            try:
                value = self.cache.pop(key)
                self.cache[key] = value  # Move to the end to show that it was recently used
                return value
            except KeyError:
                return None

    def put(self, key, value):
        with self.lock:
            try:
                self.cache.pop(key)
            except KeyError:
                if len(self.cache) >= self.capacity:
                    # Remove first item: the least recently used
                    removed_key, removed_value = self.cache.popitem(last=False)
                    # Optional: Clean up resources associated with the removed model
                    del removed_value
            self.cache[key] = value

# Global model cache with capacity limit
model_cache = LRUCache(capacity=10)  # Adjust capacity based on your resource limits

# Function to fetch products from the database
def fetch_products(app_id):
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cursor:
            # Fetch product names and photos for the given app_id
            # SELECT name, photo FROM products WHERE app_id = %s
            cursor.execute("""
                SELECT product_id, photo FROM products WHERE app_id = %s
            """, (app_id,))
            products = cursor.fetchall()
            if not products:
                return None
            # Extract class names and image paths
            class_name_list = [product[0] for product in products]
            query_img_list = [product[1] for product in products]
            return class_name_list, query_img_list
    except Exception as e:
        logging.error(f"Database error: {e}")
        return None
    finally:
        db_pool.putconn(conn)

@app.route('/detect_items', methods=['POST'])
def detect_items():
    app_id = request.form.get('app_id')
    if not app_id:
        return jsonify({'error': 'No app_id provided'}), 400

    # Get or create the model for this app_id
    model_instance = model_cache.get(app_id)
    if not model_instance:
        # Create a new model instance
        model_instance = v2vdet_class()

        # Fetch query images and class names from the database
        data = fetch_products(app_id)
        if not data:
            return jsonify({'error': f'No products found for app_id {app_id}'}), 400
        class_name_list, query_img_list = data
        # Adjust image paths if necessary (e.g., prepend directory paths)
        query_img_list = [f"/DATA3/uploads/{img}" for img in query_img_list]

        # Check if image files exist
        for img_path in query_img_list:
            if not Path(img_path).is_file():
                return jsonify({'error': f'Image file not found: {img_path}'}), 400

        model_instance.set_classes(query_img_list, class_name_list)
        model_cache.put(app_id, model_instance)

    # Process the uploaded file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file:
        bytes_io = BytesIO()
        file.save(bytes_io)
        bytes_io.seek(0)
    else:
        return jsonify({'error': 'File is empty'}), 400

    result = model_instance.predict(BytesIO_Obj=bytes_io, save_json=True)
    return jsonify({'points': result['centers'], 'bbox': result['bbox'], 'ids': result['cls']})

# Update Classes for Cached app_id
@app.route('/update_classes', methods=['POST'])
def update_classes():
    app_id = request.form.get('app_id')
    if not app_id:
        return jsonify({'error': 'No app_id provided'}), 400

    # Check if the app_id is in the cache
    model_instance = model_cache.get(app_id)
    if not model_instance:
        return jsonify({'message': f'app_id {app_id} is not in the cache'}), 200

    # Fetch query images and class names from the database
    data = fetch_products(app_id)
    if not data:
        return jsonify({'error': f'No products found for app_id {app_id}'}), 400
    class_name_list, query_img_list = data

    # Adjust image paths if necessary (e.g., prepend directory paths)
    query_img_list = [f"/DATA3/uploads/{img}" for img in query_img_list]

    # Check if image files exist
    for img_path in query_img_list:
        if not Path(img_path).is_file():
            return jsonify({'error': f'Image file not found: {img_path}'}), 400

    try:
        model_instance.set_classes(query_img_list, class_name_list)
        return jsonify({'message': f'Classes updated successfully for app_id {app_id}'}), 200
    except Exception as e:
        logging.error(f"Error updating classes for app_id {app_id}: {e}")
        return jsonify({'error': 'Failed to update classes'}), 500


def convert_pil_to_opencv(pil_image):
    # Convert the PIL Image to a NumPy array
    np_image = np.array(pil_image)

    # Convert RGB to BGR (what OpenCV uses)
    opencv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return opencv_image

@app.route('/crop_image', methods=['POST'])
def crop_image():
    # Receive the image and bbox from the request
    image = request.files.get('image')
    if not image:
        return jsonify({'error': 'No image provided'}), 400

    bbox = request.form.get('bbox')
    if not bbox:
        return jsonify({'error': 'No bbox provided'}), 400

    bbox = json.loads(bbox)  # Expecting a dictionary with 'x_min', 'y_min', 'x_max', 'y_max'
    
    uuid_ = request.form.get('uuid')
    if not uuid_:
        return jsonify({'error': 'uuid is required'}), 400

    # Generate a secure, unique filename using uuid
    filename = f"{uuid_}_init.png"
    temp_path = os.path.join('/tmp', filename)
    image.save(temp_path)

    # Convert the PIL image to OpenCV format
    input_img = Image.open(temp_path).convert("RGB")
    img = convert_pil_to_opencv(input_img)  # Your custom conversion function
    height, width, _ = img.shape  # (H, W, C)

    # Cleanup original temp file as soon as possible
    os.remove(temp_path)

    # Extract bbox coordinates
    x_min = int(bbox['x_min'])
    y_min = int(bbox['y_min'])
    x_max = int(bbox['x_max'])
    y_max = int(bbox['y_max'])

    # Calculate the bbox width, height, and the side of the square
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    side = max(bbox_width, bbox_height)

    # Enforce a square by adjusting x_max or y_max accordingly
    # Starting assumption: keep x_min, y_min as the "top-left"
    x_max_new = x_min + side
    y_max_new = y_min + side

    # If the new x_max is out of image bounds, shift x_min left
    if x_max_new > width:
        x_max_new = width
        x_min = x_max_new - side
        # If we shift too far (x_min < 0), clamp to 0
        if x_min < 0:
            x_min = 0
            # Recompute x_max_new
            x_max_new = x_min + side
            # If that still exceeds the width, clamp it
            if x_max_new > width:
                x_max_new = width

    # Similarly, if the new y_max is out of image bounds, shift y_min up
    if y_max_new > height:
        y_max_new = height
        y_min = y_max_new - side
        # If we shift too far (y_min < 0), clamp to 0
        if y_min < 0:
            y_min = 0
            # Recompute y_max_new
            y_max_new = y_min + side
            # If that still exceeds the height, clamp it
            if y_max_new > height:
                y_max_new = height

    # Now we have a square region [x_min, x_max_new], [y_min, y_max_new]
    # Ensure they are ints within the image bounds
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max_new = min(width, int(x_max_new))
    y_max_new = min(height, int(y_max_new))

    # Finally, crop the square region from the OpenCV image
    crop_img = img[y_min:y_max_new, x_min:x_max_new]

    # Save the cropped image to the uuid directory
    output_filename = os.path.join('/DATA3/uploads/', f"{uuid_}_frame_0.png")
    cv2.imwrite(output_filename, crop_img)

    # Convert cropped image to base64 for returning in the response
    pil_image = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    img_io = BytesIO()
    pil_image.save(img_io, 'JPEG', quality=85)
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    # Return the final response
    return jsonify({
        'status': 'success',
        'image': img_base64,
        'uuid': uuid_,
        'filename': f'{uuid_}_frame_0.png'
    })

if __name__ == "__main__":
    # Start Flask Server
    app.run(host="0.0.0.0", port=8089, debug=False, use_reloader=False)
