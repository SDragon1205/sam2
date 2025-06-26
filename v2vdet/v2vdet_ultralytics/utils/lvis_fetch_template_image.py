import os, sys
import supervision as sv
import numpy as np
from tqdm import tqdm
from PIL import Image
import requests
from custom_supervision import LVIS_DetectionDataset
from concurrent.futures import ThreadPoolExecutor, as_completed


IMAGE_FOLDER_NAME = "train2017"
COCO_DATASET_PATH = "./DATASET/lvis"
TEMPLATE_IMAGE_PATH = f"{COCO_DATASET_PATH}/template_images/{IMAGE_FOLDER_NAME}"

def process_annotation(ann):
  """process_annotation

  Args:
      ann (_type_): _description_

  Returns:
      _type_: _description_
  """
  try:
    img_url = requests.get(url=ann[0], stream=True).raw
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
      if not os.path.exists(f"{TEMPLATE_IMAGE_PATH}/{class_id}"):
        os.makedirs(f"{TEMPLATE_IMAGE_PATH}/{class_id}", exist_ok=True)
      save_path = f"{TEMPLATE_IMAGE_PATH}/{class_id}/{origin_img_name}_{bbox_idx}.jpg"
      pil_image.save(save_path, 'JPEG')
  
  except Exception as e:
    print(e)
    return None
  
def parallel_process(ds, max_workers=8):
  """parallel_process.

  Args:
      ds (_type_): _description_
      max_workers (int, optional): _description_. Defaults to 8.
  """
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_annotation, ann) for ann in ds]
    for _ in tqdm(as_completed(futures), total=len(ds), desc="Processing Annotations"):
      pass 

if __name__ == "__main__":
  ds = LVIS_DetectionDataset.from_coco(
    images_directory_path=f"./DATASET/lvis/images/train2017",
    annotations_path=f"./DATASET/lvis/lvis_v1_train.json",
  )

  # os.makedirs(TEMPLATE_IMAGE_PATH, exist_ok=True)
  # for idx in range(len(ds.classes)):
  #   os.makedirs(f"{TEMPLATE_IMAGE_PATH}/{idx}", exist_ok=True)
  
  import multiprocessing
  with multiprocessing.Pool(processes=96) as pool:
    mp = list(tqdm(pool.imap(process_annotation, ds), total=len(ds), desc="Create Template Image"))
    # mp = pool.map(process_annotation, ds)
  
  sys.exit(0)

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