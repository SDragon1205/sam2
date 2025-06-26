import os
import supervision as sv
import numpy as np
from tqdm import tqdm
from PIL import Image
import requests
from custom_supervision import LVIS_DetectionDataset


IMAGE_FOLDER_NAME = "val2017"
COCO_DATASET_PATH = "/home/user/erictsai/v2vdet/DATASET/lvis"
TEMPLATE_IMAGE_PATH = f"{COCO_DATASET_PATH}/template_images/{IMAGE_FOLDER_NAME}"

if __name__ == "__main__":
  ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"/home/user/erictsai/v2vdet/DATASET/lvis/images/val2017",
    annotations_path=f"/home/user/erictsai/v2vdet/DATASET/lvis/lvis_v1_val.json",
  )

  os.makedirs(TEMPLATE_IMAGE_PATH, exist_ok=True)
  for idx in range(len(ds.classes)):
    os.makedirs(f"{TEMPLATE_IMAGE_PATH}/{idx}", exist_ok=True)
  
  for ann in tqdm(ds):
    # x1, y1, x2, y2 = int(ann[2].xyxy)
    img_url = requests.get(ann[0], stream=True).raw
    image_rgb = Image.open(img_url).convert("RGB")
    image_np = np.array(image_rgb)
    h, w = image_np.shape[:2]
    for bbox_idx, bbox in enumerate(ann[2].xyxy):
      round_bbox = np.round(bbox).astype(np.int32)
      x1, y1, x2, y2 = int(round_bbox[0]), int(round_bbox[1]), int(round_bbox[2]), int(round_bbox[3])
      if (x1==x2): x2+=1
      if (y1==y2): y2+=1
      x1 = max(0, x1)
      y1 = max(0, y1)
      x2 = min(w, x2)
      y2 = min(h, y2)
      
      cropped_image = image_np[y1:y2, x1:x2]
      origin_img_name = ann[0].split("/")[-1].split(".")[0]
      class_id = ann[2].class_id[bbox_idx]
      pil_image = Image.fromarray(cropped_image)
    # x1, y1, x2, y2 = map(int, ann[2].xyxy)
      save_path = f"{TEMPLATE_IMAGE_PATH}/{class_id}/{origin_img_name}_{bbox_idx}.jpg"
      pil_image.save(save_path, 'JPEG')
  
  with open(f"{TEMPLATE_IMAGE_PATH}/classes.txt", "w") as f:
    for idx, c in enumerate(ds.classes):
      f.write(f"{idx}: {c}\n")