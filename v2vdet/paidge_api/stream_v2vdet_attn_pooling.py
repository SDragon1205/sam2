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
from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_with_Patch_Attn_Pooling

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
        
        self.model = V2V_with_Patch_Attn_Pooling("v2vdet_ultralytics/cfg/models/v8/yolov8s-world.yaml")
        
        
        self.ckpt_path = 'training_result/lvis_v2vdet_world_train_project/train_with_augment_template_with_attn_pooling/weights/best.pt'
        

        # wandb.watch(model)
        # ckpt = torch.load(ckpt_name)
        # self.model.model.attn_pooling.load_state_dict(ckpt['model'].attn_pooling.state_dict())
        self.model._load(weights=self.ckpt_path, task='task')
        ckpt = torch.load(self.ckpt_path)
        self.model.model.attn_pooling.load_state_dict(ckpt['model'].attn_pooling.state_dict())
        self.model.training=False
        
        # Initialize
        self.initialize_model()
        logging.info('Model Initialized Done')
      
    @torch.inference_mode()
    def set_classes(self, query_img: [str], query_class_name: [str]):
        self.query_img = query_img
        self.query_img_emb = [Image.open(img) for img in query_img]
        self.class_name = query_class_name
        self.model.set_classes(crop_img=self.query_img_emb, classes=self.class_name)
      
    @torch.inference_mode()
    def initialize_model(self):
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8))
    
    @torch.inference_mode()
    def predict(self, BytesIO_Obj: BytesIO, save_json=False, json_name='result.json'):
        pil_image = Image.open(BytesIO_Obj)
        results = self.model.predict(pil_image)

        bboxes = results[0].boxes.xyxyn.to('cpu').numpy().tolist()
        centers = [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in bboxes]
        
        result_dict = {
            'cls': results[0].boxes.cls.to('cpu').numpy().tolist(),
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
            cursor.execute("""
                SELECT name, photo FROM products WHERE app_id = %s
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
    return jsonify({'points': result['centers'], 'bbox': result['bbox']})

# Update Classes for Cached app_id
@app.route('/update_classes', methods=['POST'])
def update_classes():
    app_id = request.form.get('app_id')
    if not app_id:
        return jsonify({'error': 'No app_id provided'}), 400

    # Check if the app_id is in the cache
    model_instance = model_cache.get(app_id)
    if not model_instance:
        return jsonify({'error': f'app_id {app_id} is not in the cache'}), 400

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

if __name__ == "__main__":
    # Start Flask Server
    app.run(host="0.0.0.0", port=8089, debug=False, use_reloader=False)
