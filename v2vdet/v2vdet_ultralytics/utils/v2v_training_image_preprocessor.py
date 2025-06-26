import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from functools import partial
from itertools import repeat
from typing import List, Dict, Any, Tuple
import torch

from v2vdet.v2vdet_ultralytics.utils import random_crop_img

class V2V_Training_Image_Preprocessor:
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or mp.cpu_count()
        
    def process_chunk(self, chunk_data: Dict[str, Any]) -> Tuple[int, List]:
        """
        處理一個數據塊，返回 (index, crops) 來保持順序
        """
        img_idx, img, batch_data, num_classes, crop_size = chunk_data.values()
        
        # 使用線程池處理I/O操作
        with ThreadPoolExecutor() as executor:
            # 準備隨機樣本的參數
            random_indices = np.random.randint(0, len(batch_data["im_file"]), num_classes)
            file_paths = [batch_data["im_file"][idx] for idx in random_indices]
            
            # 並行讀取圖片
            futures = [
                executor.submit(self._load_and_crop_image, path, crop_size)
                for path in file_paths
            ]
            class_crops = [future.result() for future in futures]
            
        # 處理正樣本
        # mask = (batch_data['batch_idx'] == img_idx)
        mask = (batch_data['batch_idx'] == torch.tensor(img_idx, device=batch_data['batch_idx'].device))
        if mask.any():
            batch_start = mask.nonzero()[0].item()
            batch_count = mask.sum().item()
            
            img_classes = batch_data['cls'][batch_start:batch_start + batch_count]
            img_boxes = batch_data['bboxes'][batch_start:batch_start + batch_count]
            
            # 獲取正樣本
            cropped_positives = self._crop_positives(
                img, img_boxes, img_classes, crop_size
            )
            
            # 更新正樣本
            for crop_data in cropped_positives:
                class_idx = int(crop_data['cls'])
                if class_idx < len(class_crops):
                    class_crops[class_idx] = crop_data['crop']
                    
        # 返回 (index, crops) 來保持順序
        return img_idx, class_crops

    @staticmethod
    def _load_and_crop_image(file_path: str, crop_size: int):
        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                return random_crop_img(img, crop_size)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
            
    @staticmethod
    def _crop_positives(img, boxes, classes, size):
        class_crops = []
        unique_classes = np.unique(classes)
        
        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = boxes[cls_mask]
            
            areas = (cls_boxes[:, 2] - cls_boxes[:, 0]) * (cls_boxes[:, 3] - cls_boxes[:, 1])
            max_idx = np.argmax(areas)
            box = cls_boxes[max_idx]
            
            crop = img.crop((box[0], box[1], box[2], box[3]))
            crop = crop.resize((size, size))
            
            class_crops.append({
                'cls': cls,
                'crop': crop
            })
        
        return class_crops

    def process_batch(self, batch: Dict[str, Any], num_classes: int, crop_size: int) -> List:
        """
        處理整個批次並保持順序
        """
        # 準備數據塊
        chunks = [
            {
                'img_idx': idx,
                'img': img,
                'batch_data': batch,
                'num_classes': num_classes,
                'crop_size': crop_size
            }
            for idx, img in enumerate(batch['img'].to('cpu'))
        ]
        pass
        # 使用進程池進行並行處理
        with mp.Pool(processes=self.num_workers) as pool:
            # 獲取結果並保持順序
            results = pool.map(self.process_chunk, chunks)
            
        # 按原始順序排序結果
        sorted_results = sorted(results, key=lambda x: x[0])
        
        # 整合結果，保持順序
        crop_img_list = []
        for _, crops in sorted_results:
            if crops:
                crop_img_list.extend(crops)
                
        return crop_img_list

def move_to_cpu(data):
    """遞迴地將所有 tensor 移到 CPU"""
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {key: move_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_cpu(item) for item in data]
    return data

def process_images_parallel(batch: Dict[str, Any], num_classes: int, crop_size: int, 
                          num_workers: int = None) -> List:
    """
    主處理函數
    Args:
        batch: 輸入批次數據
        num_classes: 類別數量
        crop_size: 裁剪大小
        num_workers: CPU工作進程數
    Returns:
        處理後的圖片列表，保持與原始程式相同的順序
    """
    batch_on_cpu = move_to_cpu(batch)
    processor = V2V_Training_Image_Preprocessor(num_workers)
    return processor.process_batch(batch_on_cpu, num_classes, crop_size)