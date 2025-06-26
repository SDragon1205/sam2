from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
import cv2
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoImageProcessor, Dinov2Model, BatchFeature
import random
import torchvision.transforms as T
import torch
import torchvision.transforms.functional as F
import supervision as sv
# from supervision.utils.conversion import pillow_to_cv2
import matplotlib.pyplot as plt
from typing import TypeVar
import torchvision.utils as vutils

import copy
from datetime import datetime
from typing import Callable, Dict, Optional, Union

from packaging import version

try:
    import dill as pickle
except ImportError:
    import pickle

from ultralytics.models import YOLO
from torchvision import transforms
import random
from copy import deepcopy
from pycocotools import mask as mask_utils

ImageType = TypeVar("ImageType", np.ndarray, Image.Image)


def load_images(imgs, format='PIL'):
    """
    載入並處理不同格式的查詢圖片，統一輸出指定格式

    參數:
        imgs: 可以是以下格式或其list:
            - 字串/Path路徑
            - numpy array
            - PIL.Image
            - torch.Tensor
        format: 輸出格式，目前支援 'PIL'

    回傳:
        list of PIL.Image: 圖片列表
    """

    def _convert_to_pil(img):
        # 處理字串或Path路徑
        if isinstance(img, (str, Path)):
            return Image.open(str(img)).convert('RGB')

        # 處理numpy array
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            return Image.fromarray(img)

        # 處理PIL Image
        if isinstance(img, Image.Image):
            return img

        # 處理torch.Tensor
        if isinstance(img, torch.Tensor):
            if img.ndim == 3:
                if img.shape[0] == 3:  # CHW -> HWC
                    img = img.permute(1, 2, 0)
                img = (img.numpy() * 255).astype(np.uint8)
                return Image.fromarray(img)
            raise ValueError(f"Error Tensor Shape {img.shape}")

        raise ValueError(f"Not accepted type: {type(img)}")

    # 驗證format參數
    if format.upper() != 'PIL':
        raise ValueError(f"Only PIL format output accepted now, you format: {format}")

    # 處理單一輸入或list輸入
    if isinstance(imgs, (list, tuple)):
        return [_convert_to_pil(img) for img in imgs]
    else:
        return [_convert_to_pil(imgs)]



def crop_normalized_bbox(image_path, bbox_xywh):
  """
  Crop an image region based on normalized bounding box coordinates using PIL

  Parameters:
  image_path (str): Path to the image file
  bbox_xywh (tuple/list): Normalized bbox coordinates (x, y, width, height), values between 0 and 1

  Returns:
  PIL.Image: Cropped image region, or None if error occurs

  Example:
  >>> bbox = [0.2, 0.3, 0.4, 0.5]  # normalized xywh coordinates
  >>> cropped_img = crop_normalized_bbox("image.jpg", bbox)
  >>> if cropped_img:
  >>>     cropped_img.save("cropped.jpg")
  """
  try:
    # Input validation for bbox format
    if len(bbox_xywh) != 4:
      raise ValueError("bbox_xywh must contain exactly 4 values (x, y, width, height)")

    # Check if all bbox values are within valid range (0-1)
    if not all(0 <= v <= 1 for v in bbox_xywh):
      raise ValueError("All bbox coordinates must be between 0 and 1")

    # Load the image
    image = Image.open(image_path)
    width, height = image.size

    # Convert normalized coordinates to pixel coordinates
    x = int(bbox_xywh[0] * width)
    y = int(bbox_xywh[1] * height)
    w = int(bbox_xywh[2] * width)
    h = int(bbox_xywh[3] * height)

    # Ensure coordinates don't exceed image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)

    # Crop the image region
    # PIL's crop method takes coordinates as (left, top, right, bottom)
    cropped_image = image.crop((x, y, x+w, y+h))

    return cropped_image

  except Exception as e:
    print(f"Error processing image: {str(e)}")
    return None

def extract_class_crops(data_list: List[Dict], size: Tuple[int, int] = None) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Extract cropped images with two return lists:
    1. One random cropped image for each class
    2. All cropped images sorted by class

    Parameters:
    data_list (List[dict]): List of dictionaries containing image information
        Expected dictionary format:
        {
            'image_path': str,  # Path to the image
            'boxes': List[List[float]],  # List of normalized xywh coordinates
            'classes': List[int]  # List of class ids corresponding to boxes
        }
    size (tuple, optional): Size to resize the cropped images (width, height)

    Returns:
    Tuple[List[PIL.Image.Image], List[PIL.Image.Image]]:
        - First list: One random crop per class, sorted by class_id
        - Second list: All crops sorted by class_id
    """
    # Dictionary to store crops for each class
    class_crops = defaultdict(list)

    def crop_bbox(image: Image.Image, bbox: List[float]) -> Image.Image:
        """Helper function to crop image using normalized coordinates"""
        width, height = image.size
        x = int(bbox[0] * width)
        y = int(bbox[1] * height)
        w = int(bbox[2] * width)
        h = int(bbox[3] * height)

        # Ensure coordinates are within image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        cropped = image.crop((x, y, x+w, y+h))

        # Convert grayscale to RGB if necessary
        if cropped.mode != 'RGB':
            cropped = cropped.convert('RGB')

        return cropped

    # Process each item in the list
    for idx, item in enumerate(data_list):
        try:
            # Get data from dictionary
            img_path = item['im_file']
            boxes = item['bboxes']
            classes = [int(cls[0]) for cls in item['cls']]

            # Load image
            image = Image.open(img_path)

            # Process each bbox and its corresponding class
            for box_idx, (box, class_id) in enumerate(zip(boxes, classes)):
                try:
                    # Crop the image
                    cropped = crop_bbox(image, box)

                    # Resize if specified
                    if size is not None:
                        cropped = cropped.resize(size, Image.Resampling.LANCZOS)

                    # Add to class dictionary
                    class_crops[class_id].append(cropped)

                except Exception as e:
                    print(f"Error processing box {box_idx} in image {img_path}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error processing item {idx} ({img_path if 'img_path' in locals() else 'unknown'}): {str(e)}")
            continue

    # Prepare both random and full lists
    random_crops = []  # One random crop per class
    all_crops = []    # All crops

    # Sort by class ID and process
    for class_id in sorted(class_crops.keys()):
        if class_crops[class_id]:  # Check if class has any crops
            # Add random crop for this class
            random_crops.append([class_id, random.choice(class_crops[class_id])])
            # Add all crops for this class
            # all_crops.extend([class_id, class_crops[class_id]])
        # else:
        #   random_crops.append([class_id, []])

    for cls_idx in sorted(class_crops):
      if len(class_crops[cls_idx])>0:
        for img_idx in class_crops[cls_idx]:
          all_crops.append([cls_idx, img_idx])
      else:
        all_crops.append([cls_idx, []])

    return random_crops, all_crops

def count_trainable_parameters(model):
    """
    計算PyTorch模型中可訓練的參數總數

    Args:
        model: PyTorch 模型實例

    Returns:
        total_params: 可訓練的參數總數
        total_size_mb: 參數占用的記憶體大小(MB)
    """
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()

    # 計算參數占用的記憶體大小(假設使用32位元浮點數)
    total_size_mb = total_params * 4 / (1024 * 1024)

    return total_params, total_size_mb

def apply_augmentation(
  img,
  aug_params={
      'rotation_range': (-30, 30),     # Range of rotation angles
      'scale_range': (0.8, 1.2),       # Range of scaling factors
      'brightness_range': (0.8, 1.2),   # Range of brightness adjustment
      'contrast_range': (0.8, 1.2),     # Range of contrast adjustment
      'prob': 0.5,                       # Probability of applying each augmentation
      'global_prob': 0.7                 # Probability of applying augmentation
  }):
  # """Internal function for applying data augmentation"""
  # if not augment:
  #     return img

  # Create a copy to avoid modifying the original image
  img = img.copy()

  if random.random() > aug_params['global_prob']:
    return img

  # 1. Random rotation
  if random.random() < aug_params['prob']:
      angle = random.uniform(*aug_params['rotation_range'])
      img = img.rotate(angle, Image.BILINEAR, expand=True)

  # 2. Random scaling
  if random.random() < aug_params['prob']:
      scale = random.uniform(*aug_params['scale_range'])
      new_size = tuple(int(dim * scale) for dim in img.size)
      try:
        img = img.resize(new_size, Image.LANCZOS)
      except:
        pass

  # 3. Brightness adjustment
  if random.random() < aug_params['prob']:
      brightness_factor = random.uniform(*aug_params['brightness_range'])
      enhancer = ImageEnhance.Brightness(img)
      img = enhancer.enhance(brightness_factor)

  # 4. Contrast adjustment
  if random.random() < aug_params['prob']:
      contrast_factor = random.uniform(*aug_params['contrast_range'])
      enhancer = ImageEnhance.Contrast(img)
      img = enhancer.enhance(contrast_factor)

  return img

def crop_and_resize_largest_bbox_per_class(
    image,
    boxes,
    classes,
    size=(224, 224),
    augment=False,
    aug_params={
        'rotation_range': (-30, 30),     # Range of rotation angles
        'scale_range': (0.8, 1.2),       # Range of scaling factors
        'brightness_range': (0.8, 1.2),   # Range of brightness adjustment
        'contrast_range': (0.8, 1.2),     # Range of contrast adjustment
        'prob': 0.5,                       # Probability of applying each augmentation
        'global_prob': 0.7                 # Probability of applying augmentation
    }
):
    """
    Crop and resize the largest bounding box for each class, with optional data augmentation.

    Args:
        image (torch.Tensor): Input image tensor with shape (C,H,W)
        boxes (torch.Tensor): Bounding boxes tensor with shape (N,4) in format [x,y,w,h]
        classes (torch.Tensor): Class labels tensor with shape (N,)
        size (tuple): Output image size, default is (224,224) to comply with CLIP
        augment (bool): Whether to apply data augmentation, default is False
        aug_params (dict): Dictionary of augmentation parameters including:
            - rotation_range: Range of rotation angles
            - scale_range: Range of scaling factors
            - brightness_range: Range of brightness adjustment
            - contrast_range: Range of contrast adjustment
            - prob: Probability of applying each augmentation

    Returns:
        List[Dict]: List of dictionaries containing cropped images and metadata for each class
                   Each dict contains:
                   - 'cls': class label (int)
                   - 'crop_img': cropped and resized PIL Image
                   - 'bbox_area': area of the bounding box (float)
                   - 'crop_tensor_img': tensor representation of the cropped image
    """
    boxes = boxes.clone().to("cpu")

    # Basic type checking
    if not isinstance(image, torch.Tensor) or not isinstance(boxes, torch.Tensor):
        raise TypeError("image and boxes must be torch.Tensor")

    H, W = image.shape[1:]

    # Convert bounding boxes to coordinates
    areas = (boxes[:, 2] * boxes[:, 3])
    coords = torch.zeros_like(boxes)
    coords[:, 0] = ((boxes[:, 0] - boxes[:, 2]/2) * W)  # x
    coords[:, 1] = ((boxes[:, 1] - boxes[:, 3]/2) * H)  # y
    coords[:, 2] = (boxes[:, 2] * W)  # w
    coords[:, 3] = (boxes[:, 3] * H)  # h

    # Clamp coordinates to image boundaries
    coords[:, 0].clamp_(torch.tensor(0), W-1)
    coords[:, 1].clamp_(torch.tensor(0), H-1)
    coords[:, 2].clamp_(torch.tensor(0), W-coords[:, 0])
    coords[:, 3].clamp_(torch.tensor(0), H-coords[:, 1])

    # Use a dictionary to store the largest bounding box for each class
    cls_dict = {}
    for idx, (cls, area, coord) in enumerate(zip(classes, areas, coords)):
        cls = cls.item()
        if cls not in cls_dict or area > cls_dict[cls]['area']:
            cls_dict[cls] = {'area': area.item(), 'coord': coord}

    image_pil = T.ToPILImage()(image)

    crop_img_list = []
    for cls, data in cls_dict.items():
        x, y, w, h = data['coord'].int().tolist()
        if w == 0 or h == 0:
          continue
        crop = image_pil.crop((x, y, x + w, y + h))

        bg = Image.new('RGB', size, (0, 0, 0))
        
        if augment:
            crop = apply_augmentation(crop, aug_params)
            bg = Image.new('RGB', size, (0, 0, 0))

        scale = min(size[0]/w, size[1]/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        crop = crop.resize((new_w, new_h), Image.LANCZOS)

        paste_x = (size[0] - new_w) // 2
        paste_y = (size[1] - new_h) // 2
        bg.paste(crop, (paste_x, paste_y))
        crop = bg
        tensor_crop = T.ToTensor()(crop)

        crop_img_list.append({
            'cls': cls,
            'crop_img': crop,
            'bbox_area': data['area'],
            'crop_tensor_img': tensor_crop
        })

    return crop_img_list

def random_crop_img(image:Image, size=(64, 64)):
  """
  Randomly crop a region from a PIL Image.

  Args:
      image (PIL.Image): Input image in PIL Image format
      size (tuple): Target size for cropping in (width, height) format, default is (64, 64)

  Returns:
      PIL.Image: Cropped image patch with specified size

  Example:
      >>> from PIL import Image
      >>> img = Image.open('sample.jpg')
      >>> cropped = random_crop_img(img, size=(128, 128))
  """

  H, W = image.size
  x1 = random.randint(0, max(0, W - size[0]))
  y1 = random.randint(0, max(0, H - size[1]))

  crop_box = (x1, y1, x1 + size[0], y1 + size[1])

  return image.crop(crop_box)

def prepare_v2v_crop_image(batch,
                           nc,
                           device='cpu',
                           augment=False,
                           crop_size = (224, 224),
                           aug_params={
                            'rotation_range': (-30, 30),     # Range of rotation angles
                            'scale_range': (0.8, 1.2),       # Range of scaling factors
                            'brightness_range': (0.8, 1.2),   # Range of brightness adjustment
                            'contrast_range': (0.8, 1.2),     # Range of contrast adjustment
                            'prob': 0.5,                       # Probability of applying each augmentation
                            'global_prob': 0.7                 # Probability of applying augmentation
                            }
    ):

    num_classes = nc  # Usually 80
    batch_size = len(batch['img'])

    batch_img = batch['img'].clone().to(device)

    # Prepare storage for batch embeddings
    crop_img_list = []
    random_crop = transforms.RandomCrop(size=crop_size)
    for img_idx, img in enumerate(batch_img):
        # create a list of random cropped images for each class
        single_sample_crop_img = [random_crop(batch_img[random.randint(0, batch_size-1)]) for _ in range(num_classes)]
        # find the matching boxes and classes for the current image
        matches = (batch['batch_idx'] == img_idx).nonzero()
        if len(matches) > 0:
            batch_start = (batch['batch_idx'] == img_idx).nonzero()[0].item()
            batch_count = (batch['batch_idx'] == img_idx).sum().item()
            img_classes = batch['cls'][batch_start:batch_start + batch_count]
            img_boxes = batch['bboxes'][batch_start:batch_start + batch_count]
            # crop the largest bounding box for each class
            cropped_positives = crop_and_resize_largest_bbox_per_class(
                img, img_boxes, classes=img_classes, augment=augment, size=crop_size, aug_params=aug_params
            )
            # cropped_positives = optimized_crop_and_resize(
            #     img, img_boxes, classes=img_classes, augment=augment, size=crop_size, aug_params=aug_params
            # )
            for crop_data in cropped_positives:
                single_sample_crop_img[int(crop_data['cls'])] = crop_data['crop_tensor_img']

        crop_img_list.extend(single_sample_crop_img)

    crop_img_tensor = torch.stack(crop_img_list)
    return crop_img_tensor

def crop_and_resize(
    image,
    boxes,
    classes,
    size=(224, 224),
    augment=False,
    aug_params=None
):
    """
    裁剪和調整每個類別最大邊界框函數，保持固定輸出尺寸
    Args:
        image (torch.Tensor): 輸入圖像張量，形狀為 (C,H,W)
        boxes (torch.Tensor): 邊界框張量，形狀為 (N,4)，格式為 [x,y,w,h]
        classes (torch.Tensor): 類別標籤張量，形狀為 (N,)
        size (tuple): 輸出圖像尺寸，預設為 (224,224)
        augment (bool): 是否應用資料增強，預設為 False
        aug_params (dict): 資料增強參數字典
    Returns:
        List[Dict]: 包含每個類別裁剪圖像和元數據的字典列表
    """
    # 基本類型檢查
    if not isinstance(image, torch.Tensor) or not isinstance(boxes, torch.Tensor):
        raise TypeError("image and boxes must be torch.Tensor")
    
    boxes = boxes.clone().detach().cpu()
    classes = classes.clone().detach().cpu()

    if augment:
        aug_params = aug_params if aug_params is not None else {
            'rotation_range': (-30, 30),
            'scale_range': (0.8, 1.2),
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.8, 1.2),
            'prob': 0.5,
            'global_prob': 0.7
        }
    else:
        aug_params = None
    
    # 獲取圖像尺寸
    H, W = image.shape[1:]
    
    # 計算所有邊界框的面積 (向量化操作)
    areas = boxes[:, 2] * boxes[:, 3]
    
    # 轉換所有邊界框到像素坐標 (批量操作)
    coords = torch.zeros_like(boxes)
    coords[:, 0] = (boxes[:, 0] - boxes[:, 2]/2) * W  # x_min
    coords[:, 1] = (boxes[:, 1] - boxes[:, 3]/2) * H  # y_min
    coords[:, 2] = boxes[:, 2] * W  # width
    coords[:, 3] = boxes[:, 3] * H  # height
    
    # 獲取唯一類別 (向量化操作)
    unique_classes = torch.unique(classes)
    
    # 預分配結果列表
    crop_img_list = []
    
    # 只進行一次圖像轉換
    image_pil = transforms.ToPILImage()(image)
    
    # 針對每個唯一類別，找出最大的邊界框
    for cls in unique_classes:
        # 找到當前類別的所有邊界框索引 (向量化操作)
        class_indices = (classes == cls).nonzero(as_tuple=True)[0]
        
        # 找到此類別中面積最大的邊界框索引
        max_area_idx = torch.argmax(areas[class_indices])
        bbox_idx = class_indices[max_area_idx]
        
        # 獲取最大邊界框的坐標和面積
        coord = coords[bbox_idx]
        area = areas[bbox_idx].item()
        
        # 獲取邊界框的坐標
        x, y, w, h = coord.int().tolist()
        
        # 檢查寬度和高度是否有效
        if w <= 0 or h <= 0:
            continue
            
        # 計算裁剪區域的中心點
        center_x = x + w / 2
        center_y = y + h / 2
        
        # 直接計算 224x224 的裁剪區域（以物體中心為中心點）
        crop_width, crop_height = size
        
        # 計算裁剪區域的左上角和右下角坐標
        crop_x1 = max(0, int(center_x - crop_width / 2))
        crop_y1 = max(0, int(center_y - crop_height / 2))
        
        # 處理圖像邊界情況
        if crop_x1 + crop_width > W:
            crop_x1 = max(0, W - crop_width)
        if crop_y1 + crop_height > H:
            crop_y1 = max(0, H - crop_height)
            
        crop_x2 = min(W, crop_x1 + crop_width)
        crop_y2 = min(H, crop_y1 + crop_height)
        
        # 確保我們有足夠的像素來填充目標尺寸
        # 如果圖像太小，可能需要進行填充
        actual_width = crop_x2 - crop_x1
        actual_height = crop_y2 - crop_y1
        
        # 裁剪區域
        crop = image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        # 應用資料增強（如果啟用）
        if augment:
            crop = optimized_apply_augmentation(crop, aug_params)
        
        # 創建輸出圖像
        bg = Image.new('RGB', size, (0, 0, 0))
        
        # 隨機背景（如果啟用）
        if augment:
            if  (random.random() < aug_params['prob']):
            # Crop random area from pic as background
                random_crop_image_pil = random_crop_pil_image(pil_image=image_pil)
                bg = optimized_apply_augmentation(
                    random_crop_image_pil,
                    aug_params,
                    apply_rotation=False,
                    apply_scaling=False
                )
        
        # 如果裁剪區域小於目標尺寸，需要調整大小
        if actual_width < crop_width or actual_height < crop_height:
            # 將裁剪區域調整為目標尺寸
            crop = crop.resize(size, Image.LANCZOS)
            # 直接將調整後的圖像粘貼到背景上
            bg.paste(crop, (0, 0))
        else:
            # 裁剪區域已經是目標尺寸，直接粘貼
            bg.paste(crop, (0, 0))
        
        # 轉換回張量
        tensor_crop = transforms.ToTensor()(bg)
        
        # 添加到結果列表
        crop_img_list.append({
            'cls': cls.item(),
            'crop_img': bg,
            'bbox_area': area,
            'crop_tensor_img': tensor_crop
        })
    
    return crop_img_list

def origin_crop_and_resize(
    image,
    boxes,
    classes,
    size=(224, 224),
    augment=False,
    aug_params={
        'rotation_range': (-30, 30),
        'scale_range': (0.8, 1.2),
        'brightness_range': (0.8, 1.2),
        'contrast_range': (0.8, 1.2),
        'prob': 0.5,
        'global_prob': 0.7
    }
):
    """
    優化版本的裁剪和調整每個類別最大邊界框函數

    Args:
        image (torch.Tensor): 輸入圖像張量，形狀為 (C,H,W)
        boxes (torch.Tensor): 邊界框張量，形狀為 (N,4)，格式為 [x,y,w,h]
        classes (torch.Tensor): 類別標籤張量，形狀為 (N,)
        size (tuple): 輸出圖像尺寸，預設為 (224,224)
        augment (bool): 是否應用資料增強，預設為 False
        aug_params (dict): 資料增強參數字典

    Returns:
        List[Dict]: 包含每個類別裁剪圖像和元數據的字典列表
    """
    # 基本類型檢查
    if not isinstance(image, torch.Tensor) or not isinstance(boxes, torch.Tensor):
        raise TypeError("image and boxes must be torch.Tensor")

    # 克隆張量並移至CPU，避免修改原始數據和確保安全操作
    boxes = boxes.clone().detach().cpu()
    classes = classes.clone().detach().cpu()

    # 獲取圖像尺寸
    H, W = image.shape[1:]

    # 計算所有邊界框的面積 (向量化操作)
    areas = boxes[:, 2] * boxes[:, 3]

    # 轉換所有邊界框到像素坐標 (批量操作)
    coords = torch.zeros_like(boxes)
    coords[:, 0] = (boxes[:, 0] - boxes[:, 2]/2) * W  # x_min
    coords[:, 1] = (boxes[:, 1] - boxes[:, 3]/2) * H  # y_min
    coords[:, 2] = boxes[:, 2] * W  # width
    coords[:, 3] = boxes[:, 3] * H  # height

    # 限制坐標範圍到圖像邊界
    coords[:, 0].clamp_(0, W-1)
    coords[:, 1].clamp_(0, H-1)
    tensor_W = torch.tensor(W).repeat(len(coords))
    tensor_H = torch.tensor(H).repeat(len(coords))

    coords[:, 2].clamp_(torch.tensor(10), tensor_W-coords[:, 0])  # 確保寬度至少為1
    coords[:, 3].clamp_(torch.tensor(10), tensor_H-coords[:, 1])  # 確保高度至少為1

    # 獲取唯一類別 (向量化操作)
    unique_classes = torch.unique(classes)

    # 預分配結果列表
    crop_img_list = []

    # 只進行一次圖像轉換
    image_pil = T.ToPILImage()(image)

    # 針對每個唯一類別，找出最大的邊界框
    for cls in unique_classes:
        # 找到當前類別的所有邊界框索引 (向量化操作)
        class_indices = (classes == cls).nonzero(as_tuple=True)[0]

        # 找到此類別中面積最大的邊界框索引
        max_area_idx = torch.argmax(areas[class_indices])
        bbox_idx = class_indices[max_area_idx]

        # 獲取最大邊界框的坐標和面積
        coord = coords[bbox_idx]
        area = areas[bbox_idx].item()

        # 檢查寬度和高度是否有效
        x, y, w, h = coord.int().tolist()
        if w <= 0 or h <= 0:
            continue

        # 裁剪最大邊界框區域
        crop = image_pil.crop((x, y, x + w, y + h))
        
        # 建立黑色背景並粘貼縮放後的裁剪圖像
        bg = Image.new('RGB', size, (0, 0, 0))

        # 應用資料增強（如果啟用）
        if augment:
            crop = optimized_apply_augmentation(crop, aug_params)
            if random.random() < aug_params['prob']:
                # Crop random area from pic as background
                random_crop_image_pil = random_crop_pil_image(image_pil)
                bg = optimized_apply_augmentation(random_crop_image_pil, 
                    aug_params,
                    apply_rotation = False,
                    apply_scaling = False
                )
                
        # 計算保持長寬比的縮放因子
        scale = min(size[0]/w, size[1]/h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # 縮放裁剪圖像
        crop = crop.resize((new_w, new_h), Image.LANCZOS)

        # 計算居中位置
        paste_x = (size[0] - new_w) // 2
        paste_y = (size[1] - new_h) // 2

        # 將縮放後的圖像粘貼到黑色背景上
        bg.paste(crop, (paste_x, paste_y))

        # 轉換回張量
        tensor_crop = T.ToTensor()(bg)

        # 添加到結果列表
        crop_img_list.append({
            'cls': cls.item(),
            'crop_img': bg,
            'bbox_area': area,
            'crop_tensor_img': tensor_crop
        })

    return crop_img_list

def optimized_apply_augmentation(
    img,
    aug_params={
        'rotation_range': (-30, 30),
        'scale_range': (0.8, 1.2),
        'brightness_range': (0.8, 1.2),
        'contrast_range': (0.8, 1.2),
        'prob': 0.5,
        'global_prob': 0.7
    },
    apply_rotation=True,
    apply_scaling=True,
    apply_brightness=True,
    apply_contrast=True
):
    """優化版本的資料增強應用函數"""
    # 全局概率檢查 - 提前退出以提高效率
    if random.random() > aug_params['global_prob']:
        return img

    # 創建副本以避免修改原始圖像
    img = img.copy()

    # 預先決定應用哪些增強，減少隨機數生成次數
    apply_rotation = apply_rotation & (random.random() < aug_params['prob'])
    apply_scaling = apply_scaling & (random.random() < aug_params['prob'])
    apply_brightness = apply_brightness & (random.random() < aug_params['prob'])
    apply_contrast = apply_contrast & (random.random() < aug_params['prob'])

    # 1. 隨機旋轉
    if apply_rotation:
        angle = random.uniform(*aug_params['rotation_range'])
        img = img.rotate(angle, Image.BILINEAR, expand=True)

    # 2. 隨機縮放
    if apply_scaling:
        scale = random.uniform(*aug_params['scale_range'])
        new_size = tuple(max(1, int(dim * scale)) for dim in img.size)
        img = img.resize(new_size, Image.LANCZOS)

    # 3. 亮度調整
    if apply_brightness:
        brightness_factor = random.uniform(*aug_params['brightness_range'])
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)

    # 4. 對比度調整
    if apply_contrast:
        contrast_factor = random.uniform(*aug_params['contrast_range'])
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

    return img

def random_crop_pil_image(pil_image, crop_size=(224, 224)):
    """
    從PIL圖像中隨機裁剪指定大小的區域
    
    參數:
        pil_image (PIL.Image): 輸入的PIL圖像
        crop_size (tuple): 裁剪尺寸，默認(224, 224)
        
    返回:
        PIL.Image: 裁剪後的PIL圖像
    """
    # 獲取原始圖像尺寸
    width, height = pil_image.size
    crop_width, crop_height = crop_size
    
    # 確保原始圖像大於需要裁剪的尺寸
    if height < crop_height or width < crop_width:
        bg = Image.new('RGB', crop_size, (0, 0, 0))
        paste_x = (crop_size[0] - width) // 2
        paste_y = (crop_size[1] - height) // 2
        bg.paste(pil_image, (paste_x, paste_y))
        # raise ValueError(f"原始圖像大小{(width, height)}小於裁剪大小{crop_size}")
    
    # 隨機選擇裁剪的起始點
    left = random.randint(0, max(0, width - crop_width))
    top = random.randint(0, max(0, height - crop_height))
    right = left + crop_width
    bottom = top + crop_height
    
    # 執行裁剪
    cropped_image = pil_image.crop((left, top, right, bottom))
    
    return cropped_image

def random_tensor_crop_to_pil(tensor_image, crop_size=(224, 224)):
    """
    從tensor image中隨機裁剪指定大小的區域，並轉換成PIL圖像
    
    參數:
        tensor_image (torch.Tensor): 形狀為[C, H, W]的圖像tensor
        crop_size (tuple): 裁剪尺寸，默認(224, 224)
        
    返回:
        PIL.Image: 裁剪後的PIL圖像
    """
    # 確保輸入是正確的形狀
    if len(tensor_image.shape) != 3:
        raise ValueError(f"輸入tensor需為[C, H, W]形狀，但收到{tensor_image.shape}")
    
    c, h, w = tensor_image.shape
    crop_height, crop_width = crop_size
    
    # 確保原始圖像大於需要裁剪的尺寸
    if h < crop_height or w < crop_width:
        raise ValueError(f"原始圖像大小{(h, w)}小於裁剪大小{crop_size}")
    
    # 隨機選擇裁剪的起始點
    top = random.randint(0, h - crop_height)
    left = random.randint(0, w - crop_width)
    
    # 執行裁剪
    cropped_tensor = tensor_image[:, top:top+crop_height, left:left+crop_width]
    
    # 轉換為PIL圖像
    # 首先確保tensor在正確的範圍內 (0-1 或 0-255)
    if cropped_tensor.max() <= 1.0:
        cropped_tensor = cropped_tensor * 255
    
    # 轉換為uint8並轉置為[H, W, C]格式
    cropped_array = cropped_tensor.permute(1, 2, 0).cpu().numpy().astype('uint8')
    
    # 創建PIL圖像
    pil_image = Image.fromarray(cropped_array)
    
    return pil_image

def train_prepare_template(vision_encoder, num_classes, batch_size, crop_size, batch, device, aug_params):
    """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""

    # Prepare storage for batch embeddings
    crop_img_list = []
    random_crop = transforms.RandomCrop(size=crop_size)
    for img_idx, img in enumerate(batch['img']):
      single_sample_crop_img = [random_crop(
          batch['img'][random.randint(0, batch_size-1)]) for _ in range(num_classes)]
      matches = (batch['batch_idx'] == img_idx).nonzero()
      if len(matches) > 0:
        batch_start = (batch['batch_idx'] == img_idx).nonzero()[0].item()
        batch_count = (batch['batch_idx'] == img_idx).sum().item()
        img_classes = batch['cls'][batch_start:batch_start + batch_count]
        img_boxes = batch['bboxes'][batch_start:batch_start + batch_count]

        cropped_positives = crop_and_resize_largest_bbox_per_class(
            img, img_boxes, img_classes, size=crop_size, augment=True, aug_params=aug_params
        )
        for crop_data in cropped_positives:
          single_sample_crop_img[int(
              crop_data['cls'])] = crop_data['crop_tensor_img'].to(device)

      crop_img_list.extend(single_sample_crop_img)

      # Extract features using CLIP
    with torch.inference_mode():
      vision_encoder = vision_encoder.to(device)
      vision_encoder.eval()
      crop_img_tensor = torch.stack(crop_img_list)

      clip_input = BatchFeature(
            {'pixel_values': crop_img_tensor}, tensor_type='pt')
      vision_model_output = vision_encoder(**clip_input, return_dict=True)


    return vision_model_output


def process_segmentation_mask_np(rle_data, original_image=None):
    """
    處理 RLE 格式的分割遮罩，並將分割區域剪下貼到黑色背景上

    參數:
    rle_data (dict): 包含 'size' 和 'counts' 的 RLE 資料
    original_image (numpy.ndarray, optional): 原始圖像，如果提供則會將分割區域從原圖中剪下

    返回:
    tuple: (mask_image, extracted_segment) 遮罩圖像和分割區域
    """
    mask_width, mask_height = rle_data['size'][0], rle_data['size'][1]
    resize_scale = 1

    # 使用 pycocotools 解碼 RLE
    binary_mask = mask_utils.decode(rle_data)

    # 如果提供了目標尺寸，則對遮罩進行縮放
    if original_image is not None:
      target_width, target_height  = original_image.shape[:2]
      # 使用 OpenCV 的 resize 函數對遮罩進行縮放
      # 使用 INTER_NEAREST 以保持二值特性
      binary_mask = cv2.resize(binary_mask, (target_width//resize_scale, target_height//resize_scale), interpolation=cv2.INTER_NEAREST)
      binary_mask = crop_center(binary_mask,
                                scale=resize_scale,
                  target_height=target_height//resize_scale,
                  target_width=target_width//resize_scale)
      # breakpoint()
    #   print(f"遮罩已從 {mask_width}x{mask_height} 縮放到 {target_width}x{target_height}")
    else:
      target_width, target_height = mask_width, mask_height

    # 如果提供了原始圖像，則從原圖中提取分割區域
    if original_image is not None:
      # 確保原圖尺寸正確
      img_height, img_width = original_image.shape[:2]

      # 如果原圖尺寸與目標尺寸不匹配，則調整原圖尺寸
    #   if img_height != target_height or img_width != target_width:
      original_image = cv2.resize(original_image, (target_width//resize_scale, target_height//resize_scale))
        #   print(f"原圖已從 {img_width}x{img_height} 調整到 {target_width}x{target_height}")

      # 提取分割區域
      extracted_segment = np.zeros_like(original_image)
      for c in range(3):  # 對 RGB 三個通道分別處理
          extracted_segment[:,:,c] = original_image[:,:,c] * binary_mask

      # 將分割區域放到黑色背景上
      segment_on_black = extracted_segment.copy()
    else:
      # 如果沒有原始圖像，只顯示白色的遮罩區域
      extracted_segment = np.stack([binary_mask * 255] * 3, axis=2).astype(np.uint8)
      segment_on_black = extracted_segment.copy()

    segment_on_black = cv2.resize(segment_on_black, (target_width, target_height))
    binary_mask = cv2.resize(binary_mask, (target_width, target_height))
    return binary_mask, segment_on_black


def process_segmentation_mask(rle_data, original_image=None):
    """
    處理 RLE 格式的分割遮罩，並將分割區域剪下貼到黑色背景上

    參數:
    rle_data (dict): 包含 'size' 和 'counts' 的 RLE 資料
    original_image (torch.Tensor, optional): 原始圖像的 PyTorch tensor，
                                           格式應為 [C, H, W] 或 [B, C, H, W]

    返回:
    tuple: (mask_image, extracted_segment) 遮罩圖像和分割區域
    """
    # 從 RLE 資料解碼分割遮罩
    h, w = rle_data['size'][1], rle_data['size'][0]

    # 使用 pycocotools 解碼 RLE
    binary_mask = mask_utils.decode(rle_data)

    # 將二進制遮罩轉換為 PyTorch tensor
    binary_mask_tensor = torch.from_numpy(binary_mask).float()

    # 如果提供了原始圖像 (PyTorch tensor)，則從原圖中提取分割區域
    if original_image is not None:
        # 處理 tensor 維度
        if original_image.dim() == 4:  # [B, C, H, W]
            # 取第一個樣本，假設批次大小為 1
            original_image = original_image[0]

        # 確保原圖尺寸正確 (PyTorch tensor 格式為 [C, H, W])
        assert original_image.shape[1:] == (h, w), f"原始圖像尺寸不匹配: {original_image.shape[1:]} vs 預期的 {(h, w)}"

        # 創建適合遮罩的形狀
        # 將 2D 遮罩 [H, W] 擴展為 3D [1, H, W]，便於廣播
        mask_expanded = binary_mask_tensor.unsqueeze(0)

        # 提取分割區域: 將遮罩應用於每個通道
        # 使用 PyTorch 的廣播機制
        extracted_segment = original_image * mask_expanded

        # 將 PyTorch tensor 轉換回 numpy 以便繪圖
        # 轉置為 [H, W, C] 格式
        # extracted_segment_np = extracted_segment.permute(1, 2, 0).cpu().numpy()

        # 將分割區域放到黑色背景上 (已經是黑色背景，因為未遮罩的部分為 0)
        # segment_on_black = extracted_segment_np
        segment_on_black = extracted_segment

    else:
        # 如果沒有原始圖像，只顯示白色的遮罩區域
        extracted_segment = torch.stack([binary_mask * 255] * 3, axis=2).astype(np.uint8)
        segment_on_black = deepcopy(extracted_segment)

    return binary_mask, segment_on_black

def process_numpy_to_cropped_tensors(numpy_images, crop_size)->torch.Tensor:
    """
    將多個numpy陣列圖像轉換為PyTorch張量，進行中心裁剪，並存儲到列表中
    
    參數:
        numpy_images: 包含多個numpy陣列圖像的列表
        crop_size: 裁剪後的尺寸，可以是單一整數或(height, width)元組
    
    返回:
        裁剪後的PyTorch張量列表
    """
    # 定義轉換流程
    if isinstance(crop_size, int):
        crop_transform = transforms.CenterCrop(crop_size)
    else:
        crop_transform = transforms.CenterCrop(crop_size)
    
    cropped_tensors = []
    
    for np_img in numpy_images:
        # 將numpy陣列轉換為PyTorch張量
        if len(np_img.shape) == 3:  # 彩色圖像 (H, W, C)
            # 將 numpy 的 (H, W, C) 轉換為 PyTorch 的 (C, H, W)
            tensor_img = torch.from_numpy(np_img.transpose(2, 0, 1)).float()
        else:  # 灰度圖像 (H, W)
            tensor_img = torch.from_numpy(np_img).float().unsqueeze(0)
        
        # 進行中心裁剪
        cropped_img = crop_transform(tensor_img)
        
        # 添加到結果列表
        cropped_tensors.append(cropped_img)
    
    return torch.cat(cropped_tensors, dim=0)

def extract_and_sample_images(data_list, sample_size=80, crop_size=(224, 224))->torch.Tensor:
    """
    從包含字典的列表中提取'img'字段，並隨機採樣指定數量的圖像
    
    參數:
        data_list: 包含字典的列表，每個字典中有'img'鍵對應numpy圖像
        sample_size: 想要採樣的圖像數量（允許重複）
        crop_size: 中心裁剪的尺寸
        
    返回:
        採樣並裁剪後的PyTorch張量列表
    """
    # 定義中心裁剪轉換
    # crop_transform = transforms.CenterCrop(crop_size)
    
    # 從列表中提取所有圖像
    all_images = [item['img'] for item in data_list]
    
    # 隨機採樣（允許重複）
    sampled_indices = random.choices(range(len(all_images)), k=sample_size)
    sampled_images = [all_images[i] for i in sampled_indices]

    return process_numpy_to_cropped_tensors(sampled_images, crop_size)
    
    # # 轉換為PyTorch張量並裁剪
    # cropped_tensors = []
    # for np_img in sampled_images:
    #     if len(np_img.shape) == 3:  # 彩色圖像 (H, W, C)
    #         tensor_img = torch.from_numpy(np_img.transpose(2, 0, 1)).float()
    #     else:  # 灰度圖像 (H, W)
    #         tensor_img = torch.from_numpy(np_img).float().unsqueeze(0)
        
    #     # 進行中心裁剪
    #     cropped_img = crop_transform(tensor_img)
        
    #     # 添加到結果列表
    #     cropped_tensors.append(cropped_img)
    
    # return cropped_tensors

def random_sample_picture(data_list, sample_size=80)->list:
    """
    從包含字典的列表中提取'img'字段，並隨機採樣指定數量的圖像
    
    參數:
        data_list: 包含字典的列表，每個字典中有'img'鍵對應numpy圖像
        sample_size: 想要採樣的圖像數量（允許重複）
        crop_size: 中心裁剪的尺寸
        
    返回:
        採樣並裁剪後的PyTorch張量列表
    """
    # 定義中心裁剪轉換
    # crop_transform = transforms.CenterCrop(crop_size)
    
    # 從列表中提取所有圖像
    all_images = [item['img'] for item in data_list]
    
    # 隨機採樣（允許重複）
    sampled_indices = random.choices(range(len(all_images)), k=sample_size)
    sampled_images = [all_images[i] for i in sampled_indices]

    img_torch_list = [torch.from_numpy(img).permute(2, 0, 1) for img in sampled_images]

    return img_torch_list


def crop_center(image, scale=2, target_height=None, target_width=None):
  """
  從圖片中心截取指定大小的區域。
  如果未指定目標尺寸，則默認截取長寬為原圖1/2的中心區域。

  參數:
  image (numpy.ndarray): 輸入圖片的 numpy 陣列
  target_height (int, optional): 目標高度，默認為原圖高度的一半
  target_width (int, optional): 目標寬度，默認為原圖寬度的一半

  返回:
  numpy.ndarray: 截取的中心區域
  """
  # 獲取原始圖片尺寸
  height, width = image.shape[:2]

  # 如果未指定目標尺寸，設為原圖的一半
  if target_height is None:
    target_height = height // scale
  if target_width is None:
    target_width = width // scale

  # 計算起始坐標 (確保中心點對齊)
  start_y = (height - target_height) // scale
  start_x = (width - target_width) // scale

  # 截取中心區域
  cropped_image = image[start_y:start_y+target_height, start_x:start_x+target_width]

  return cropped_image

def pillow_to_cv2(image: Image.Image) -> np.ndarray:
    """
    Converts Pillow image into OpenCV image, handling RGB -> BGR
    conversion.

    Args:
        image (Image.Image): Pillow image (in RGB format).

    Returns:
        (np.ndarray): Input image converted to OpenCV format.
    """
    scene = np.array(image)
    scene = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
    return scene

def resize_with_padding(img, target_size=(224, 224)):
    width, height = img.size
    ratio = min(target_size[0] / width, target_size[1] / height)
    new_size = (int(width * ratio), int(height * ratio))
    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    new_img = Image.new("RGB", target_size, (0, 0, 0))  # 黑色背景
    paste_position = ((target_size[0] - new_size[0]) // 2, 
                     (target_size[1] - new_size[1]) // 2)
    new_img.paste(resized_img, paste_position)
    return new_img

def save_image(
    image: ImageType,
    size: Tuple[int, int] = (12, 12),
    cmap: Optional[str] = "gray",
    save_path_name = "image.png"
) -> None:
    """
    save Plots image using matplotlib.

    Args:
        image (ImageType): The frame to be displayed ImageType
             is a flexible type, accepting either `numpy.ndarray` or `PIL.Image.Image`.
        size (Tuple[int, int]): The size of the plot in inches.
        cmap (str): the colormap to use for single channel images.

    Examples:
        ```python
        import cv2
        import supervision as sv

        image = cv2.imread("path/to/image.jpg")

        %matplotlib inline
        sv.plot_image(image=image, size=(16, 16))
        ```
    """
    if isinstance(image, Image.Image):
        image = pillow_to_cv2(image)

    plt.figure(figsize=size)

    if image.ndim == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.savefig(save_path_name, dpi=300, bbox_inches='tight')
    plt.close()

def save_images_grid(
    images: Union[List[ImageType], torch.Tensor],
    grid_size: Tuple[int, int],
    titles: Optional[List[str]] = None,
    size: Tuple[int, int] = (12, 12),
    cmap: Optional[str] = "gray",
    save_path_name = "image_grid.png"
) -> None:
    """
    Saving plots images in a grid using matplotlib.

    Args:
       images (List[ImageType]): A list of images as ImageType
             is a flexible type, accepting either `numpy.ndarray` or `PIL.Image.Image`.
       grid_size (Tuple[int, int]): A tuple specifying the number
            of rows and columns for the grid.
       titles (Optional[List[str]]): A list of titles for each image.
            Defaults to None.
       size (Tuple[int, int]): A tuple specifying the width and
            height of the entire plot in inches.
       cmap (str): the colormap to use for single channel images.

    Raises:
       ValueError: If the number of images exceeds the grid size.

    Examples:
        ```python
        import cv2
        import supervision as sv
        from PIL import Image

        image1 = cv2.imread("path/to/image1.jpg")
        image2 = Image.open("path/to/image2.jpg")
        image3 = cv2.imread("path/to/image3.jpg")

        images = [image1, image2, image3]
        titles = ["Image 1", "Image 2", "Image 3"]

        %matplotlib inline
        plot_images_grid(images, grid_size=(2, 2), titles=titles, size=(16, 16))
        ```
    """
    nrows, ncols = grid_size

    if isinstance(images, torch.Tensor):
        images = deepcopy(images).to('cpu').permute(0, 2, 3, 1).numpy()

    for idx, img in enumerate(images):
        if isinstance(img, Image.Image):
            images[idx] = pillow_to_cv2(img)

    if len(images) > nrows * ncols:
        raise ValueError(
            "The number of images exceeds the grid size. Please increase the grid size"
            " or reduce the number of images."
        )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)

    for idx, ax in enumerate(axes.flat):
        # print(images)
        if idx < len(images):
            if images[idx].ndim == 2:
                ax.imshow(images[idx], cmap=cmap)
            else:
                ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))

            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx])

        ax.axis("off")

    plt.savefig(save_path_name, dpi=300, bbox_inches='tight')
    plt.close()


def save_bbox(images: Union[List[ImageType], torch.Tensor], xywh, class_id=None) -> None:
  # image = cv2.imread("image.jpeg")
  if isinstance(images, torch.Tensor):
    image = deepcopy(images).to('cpu').permute(1, 2, 0).numpy()

  W, H = image.shape[0], image.shape[1]
  xyxy = [[float(x-w/2)*W, float(y-h/2)*H, float(x+w/2)*W, float(y+h/2)*H] for x, y, w, h in xywh]

  if class_id is None:
    class_id = [0 for i in range(len(xyxy))]

  if isinstance(class_id, torch.Tensor):
    class_id = sum(class_id.tolist(), [])

  detections = sv.Detections(
  xyxy=np.array(xyxy),
  class_id=np.array(class_id),
  confidence=np.array([0.94 for i in range(len(xyxy))])
  )

  bounding_box_annotator = sv.BoundingBoxAnnotator()
  annotated_frame = bounding_box_annotator.annotate(
      scene=image.copy(),
      detections=detections
  )

  save_image(annotated_frame)

def draw_bbox(image:Union[np.array, torch.tensor], xyxy, class_id) -> None:
  # image = cv2.imread("image.jpeg")
  if isinstance(image, torch.Tensor):
    image = image.to('cpu').permute(1, 2, 0).numpy()
    image = np.ascontiguousarray(image)

  detections = sv.Detections(
  xyxy=np.array(xyxy),
  class_id=np.array([0]),
  confidence=np.array([0.94 for i in range(len(xyxy))])
  )

  bounding_box_annotator = sv.BoundingBoxAnnotator()
  annotated_frame = bounding_box_annotator.annotate(
      scene=image.copy(),
      detections=detections
  )

  return annotated_frame

def save_tensor_image(img_tensor, save_name='batch_grid.png', normalize=True, padding=2):
    vutils.save_image(img_tensor, save_name, normalize=True, padding=2)
    
def batch_visualize_batch_annotations(batch, class_names=None, max_images=16):
    visualize_batch_annotations(batch['img'], batch['batch_idx'], batch['cls'], batch['bboxes'], class_names=class_names, max_images=max_images) 

def visualize_batch_annotations(images, batch_id, cls, bboxes, class_names=None, max_images=16):
    """
    可視化批次中的圖像和對應的標註框
    
    Parameters:
    - images: 批次圖像，形狀為 [B, C, H, W]，B 是批次大小
    - batch_id: 標註對應的圖片編號，形狀為 [N, 1] 或 [N]
    - cls: 標註的類別編號，形狀為 [N, 1] 或 [N]
    - bboxes: 標註的邊界框，標準化 xywh 格式，形狀為 [N, 4]
    - class_names: 類別編號對應的名稱字典，若為 None 則使用類別編號
    - max_images: 最多顯示的圖像數量
    
    Returns:
    - grid: 包含所有可視化結果的網格圖像
    """
    batch_size = len(images)
    H, W = images.shape[2], images.shape[3]
    
    # 處理輸入以確保形狀一致
    if len(batch_id.shape) > 1:
        batch_id = batch_id.squeeze()
    if len(cls.shape) > 1:
        cls = cls.squeeze()
        
    # 確保所有輸入都是張量
    if not isinstance(batch_id, torch.Tensor):
        batch_id = torch.tensor(batch_id)
    if not isinstance(cls, torch.Tensor):
        cls = torch.tensor(cls)
    if not isinstance(bboxes, torch.Tensor):
        bboxes = torch.tensor(bboxes)
    
    # 準備可視化圖像列表
    viz_images = []
    
    # 對批次中的每張圖像進行處理
    for i in range(min(batch_size, max_images)):
        # 獲取當前圖像
        img = images[i].clone()
        
        # 將圖像轉換為 uint8 類型
        if img.dtype != torch.uint8:
            img = (img * 255).to(torch.uint8)
        
        # 找出對應當前圖像的所有標註
        mask = (batch_id == i)
        if not torch.any(mask):
            # 如果沒有對應的標註，直接添加原圖
            viz_images.append(img)
            continue
        
        # 獲取對應當前圖像的標註
        img_cls = cls[mask]
        img_bboxes = bboxes[mask]
        
        # 將標準化的 xywh 格式轉換為絕對像素的 xyxy 格式
        # xywh (中心點 x, 中心點 y, 寬度, 高度) -> xyxy (左上角 x, 左上角 y, 右下角 x, 右下角 y)
        xyxy_boxes = torch.zeros_like(img_bboxes)
        xyxy_boxes[:, 0] = (img_bboxes[:, 0] - img_bboxes[:, 2] / 2) * W  # x1 = x - w/2
        xyxy_boxes[:, 1] = (img_bboxes[:, 1] - img_bboxes[:, 3] / 2) * H  # y1 = y - h/2
        xyxy_boxes[:, 2] = (img_bboxes[:, 0] + img_bboxes[:, 2] / 2) * W  # x2 = x + w/2
        xyxy_boxes[:, 3] = (img_bboxes[:, 1] + img_bboxes[:, 3] / 2) * H  # y2 = y + h/2
        xyxy_boxes = xyxy_boxes.to(torch.int64)  # 轉換為整數
        
        # 準備標籤文字
        if class_names is not None:
            labels = [class_names[c.item()] for c in img_cls]
        else:
            labels = [f"Class {c.item()}" for c in img_cls]
        
        # 繪製邊界框
        img_with_boxes = vutils.draw_bounding_boxes(
            img,
            xyxy_boxes,
            labels=labels,
            width=2,
            font_size=12
        )
        
        viz_images.append(img_with_boxes)
    
    viz_images_tensor = torch.stack(viz_images, dim=0)
    
    viz_images_tensor = viz_images_tensor.float() / 255.0
    
    vutils.save_image(viz_images_tensor, 'visualize_batch_ann.png', normalize=True, padding=2)


def create_heatmap(encoder_output):
    import seaborn as sns
    # 如果輸出是 4D 張量，需要選擇一個通道或計算平均值/最大值
    if len(encoder_output.shape) == 4:
        # 方法1：選擇特定通道
        heatmap_data = encoder_output[0, :, :, 0]  # 第一個樣本的第一個通道
        
        # 方法2：計算所有通道的平均值
        # heatmap_data = np.mean(encoder_output[0], axis=-1)  # 對通道維度取平均
        
        # 方法3：計算所有通道的最大值
        # heatmap_data = np.max(encoder_output[0], axis=-1)  # 對通道維度取最大值
    else:
        heatmap_data = encoder_output
    
    # 繪製熱圖
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='viridis', annot=False)
    plt.title('Vision Encoder Heatmap')
    plt.savefig('encoder_heatmap.png')
    plt.show()
    
    return heatmap_data

def overlay_heatmap(image, heatmap, alpha=0.5):
    # 調整熱圖大小以匹配圖像尺寸
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # 將熱圖轉換為RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 疊加熱圖到原始圖像
    overlaid = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    
    return overlaid