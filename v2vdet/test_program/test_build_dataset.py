import os, sys
import logging
from pathlib import Path

project_root = str(Path("~/erictsai/v2vdet").expanduser().resolve())
sys.path.append(project_root)

from ultralytics.data.build import build_yolo_dataset, build_dataloader
import pickle
import os
import random

import numpy as np
import time
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

# from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from v2vdet.v2vdet_ultralytics.data.dataset import (YOLODataset,
                                             V2V_COCO_Dataset, SA_V_V2VDataset, Each_Picture_Each_Class_SA_V_V2VDataset,
                                             YOLOMultiModalDataset)
from v2vdet.v2vdet_ultralytics.utils.misc import save_images_grid

from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
from ultralytics.utils import RANK, colorstr, TQDM
from ultralytics.utils.checks import check_file

from torch.utils.data import dataloader
from ultralytics.data.build import seed_worker
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2

def my_build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=False,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )

if __name__ == "__main__":
  with open("params.pkl", "rb") as f:
    loaded_params = pickle.load(f)

  with open("yolo_dataset_test.pkl", "rb") as f:
    loaded_yolo_dataset = pickle.load(f)

  cfg = loaded_params[0]
  # img_path = "DATASET/SA_V/min128_train_SA_V_img.txt"
  # # img_path = "DATASET/SA_V/min64_SA_V_img.txt"
  img_path = "DATASET/lvis/superseuper_minival.txt"
  img_path = "DATASET/lvis/train.txt"
  
  batch = 32
  data = loaded_params[3]
  data = loaded_yolo_dataset[1]
  data["train"] = img_path
  stride = loaded_params[6]
  # stride=32
  rect = False
  # cfg.mosaic=0.0

  # img_path = "DATASET/SA_V/min128_train_SA_V_img.txt"
  # img_path = "DATASET/SA_V/min64_SA_V_img.txt"  # Validation/Testing
  imgsz = 640
  batch = 4
  mode = "train"  # should not be 'train' when validation and testing

  # with open("test_program/test_build_SA_V.pkl", "rb") as f:
  #   loaded_params = pickle.load(f)
  # cfg = loaded_params[0]
  # cfg.mosaic = 0.0 ## must be set to 0 (disable), or crop template will be wrong
  # rect = False
  # stride = 32
  # data = loaded_params[3]

  # SA_V_V2V_dataset = Each_Picture_Each_Class_SA_V_V2VDataset(
  #   img_path=img_path,
  #   imgsz=cfg.imgsz,
  #   batch_size=batch,
  #   augment=(mode == "train"),  # augmentation
  #   template_hyp=cfg,
  #   hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
  #   rect=cfg.rect or rect,  # rectangular batches
  #   cache=False,
  #   single_cls=cfg.single_cls or False,
  #   stride=int(stride),
  #   pad=0.0 if mode == "train" else 0.5,
  #   prefix=colorstr(f"{mode}: "),
  #   task=cfg.task,
  #   classes=cfg.classes,
  #   data=data,
  #   fraction=cfg.fraction if mode == "train" else 1.0,
  #   train= (mode == "train"),
  #   )

  dataset = V2V_COCO_Dataset(
    img_path=img_path,
    imgsz=cfg.imgsz,
    batch_size=batch,
    augment=(mode == "train"),  # augmentation
    # hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
    rect=cfg.rect or rect,  # rectangular batches
    cache=cfg.cache or None,
    single_cls=cfg.single_cls or False,
    stride=int(stride),
    pad=0.0 if mode == "train" else 0.5,
    prefix=colorstr(f"{mode}: "),
    task=cfg.task,
    classes=cfg.classes,
    data=data,
    fraction=cfg.fraction if mode == "train" else 1.0,
    template_cache='disk',
  )

  # yolo_dataset = YOLODataset(
  #   img_path=img_path,
  #   imgsz=cfg.imgsz,
  #   batch_size=batch,
  #   augment=(mode == "train"),  # augmentation
  #   hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
  #   rect=cfg.rect or rect,  # rectangular batches
  #   cache=cfg.cache or None,
  #   single_cls=cfg.single_cls or False,
  #   stride=int(stride),
  #   pad=0.0 if mode == "train" else 0.5,
  #   prefix=colorstr(f"{mode}: "),
  #   task=cfg.task,
  #   classes=cfg.classes,
  #   data=data,
  #   fraction=cfg.fraction if mode == "train" else 1.0,
  # )

#   yolo_text_dataset = YOLOMultiModalDataset(
#     img_path=img_path,
#     imgsz=cfg.imgsz,
#     batch_size=batch,
#     augment=(mode == "train"),  # augmentation
#     hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
#     rect=cfg.rect or rect,  # rectangular batches
#     cache=cfg.cache or None,
#     single_cls=cfg.single_cls or False,
#     stride=int(stride),
#     pad=0.0 if mode == "train" else 0.5,
#     prefix=colorstr(f"{mode}: "),
#     task=cfg.task,
#     classes=cfg.classes,
#     data=data,
#     fraction=cfg.fraction if mode == "train" else 1.0,
#   )

  train_loader = my_build_dataloader(dataset=dataset,
                   batch=batch,
                   workers=8,
                   shuffle=(mode == "train"),
                   rank=-1)

  # train_loader.sampler.set_epoch(2)
  # pbar = enumerate(train_loader)
  pbar = TQDM(enumerate(train_loader), total=len(train_loader))
  for i, batch in pbar:
    # import time
    # time.sleep(0.00001)
    # print(batch)
    # continue
    # breakpoint()
    # print(batch.keys())

    time.sleep(0.00001)
    # save_images_grid(batch['template_feats'],
    #                  grid_size=(10,10),
    #                  titles=[idx for idx in range(80)],
    #                  save_path_name="test_img/template_by_test_program.png")

    # 假設有 80 張隨機 Tensor 圖片，每張大小為 (3, 32, 32)
    # num_images = 80
    # rows, cols = 10, 8
    # image_size = 32  # 假設每張圖片為 32x32

    # # # 生成 80 張隨機圖片 (C, H, W)
    # tensor_images = batch["template_feats"]

    # # # 轉換成 PIL Image 格式
    # to_pil = transforms.ToPILImage()
    # pil_images = [to_pil(tensor_image) for tensor_image in tensor_images[0]]

    # # # 建立子圖
    # fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))  # 設定適合的畫布大小

    # for i, ax in enumerate(axes.flat):
    #   ax.imshow(pil_images[i])  # 顯示圖片
    #   ax.axis("off")  # 隱藏座標軸
    #   ax.text(2, 5, f"{i+1}", color="white", fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

    # # # 調整子圖間距
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # # # 儲存為圖片
    # plt.savefig("test_img/combined_image.png", bbox_inches='tight', pad_inches=0, dpi=300)

    # # # for input_img in batch["img"]:
    # for input_img_idx, input_img in enumerate(batch["img"]):
    #   image_np = input_img.permute(1, 2, 0).numpy()
    #   image_np = np.ascontiguousarray(image_np)
    #   img_h, img_w, _ = image_np.shape
    #   indices = torch.where(batch['batch_idx'] == input_img_idx)[0].tolist()
    #   bbox_list = batch['bboxes'][indices]
    #   cls_list = batch['cls'][indices]
    #   W, H = batch['resized_shape'][0]
    #   for bbox, cls in zip(bbox_list, cls_list):

    #     x1 = int((bbox[0]-bbox[2]/2)*W)
    #     y1 = int((bbox[1]-bbox[3]/2)*H)
    #     x2 = int((bbox[0]+bbox[2]/2)*W)
    #     y2 = int((bbox[1]+bbox[3]/2)*H)

    #     cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(image_np, str(f"Class: {int(cls)+1}"), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #   cv2.imwrite(f"test_img/{input_img_idx}.jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    # import time
    # time.sleep(0.000001)
    # breakpoint()
