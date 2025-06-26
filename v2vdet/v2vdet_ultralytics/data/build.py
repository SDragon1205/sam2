import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

# from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
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
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file
from v2vdet.v2vdet_ultralytics.data.dataset import YOLODataset, YOLOMultiModalDataset
from v2vdet.v2vdet_ultralytics.data.dataset import V2V_Dataset, SA_V_V2VDataset, ObjectOrientedYOLODataset
from ultralytics.data.build import seed_worker

def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
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
    )
    
def build_object_oriented_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False, vision_encoder_input_size=224):
    """Build YOLO Dataset."""
    return ObjectOrientedYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
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
        vision_encoder_input_size=vision_encoder_input_size
    )

def build_v2v_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
  """Build V2V Dataset.
  Don't use, it still under development.
  Args:
      cfg (dict): Configuration dictionary
      img_path (str): Path to the image file
      batch (int): Batch size
      data (dict): Data dictionary
      mode (str): Mode (train or val)
      rect (bool): Rectangular batches
      stride (int): Stride
  """
  return V2V_Dataset(
      img_path=img_path,
      imgsz=cfg.imgsz,
      batch_size=batch,
      augment=(mode == "train"),  # augmentation
      hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
      rect=cfg.rect or rect,  # rectangular batches
      cache=cfg.cache or None,
      single_cls=cfg.single_cls or False,
      stride=int(stride),
      pad=0.0 if (mode == "train") else 0.5,
      prefix=colorstr(f"{mode}: "),
      task=cfg.task,
      classes=cfg.classes,
      data=data,
      fraction=cfg.fraction if (mode == "train") else 1.0,
      template_cache = cfg.template_cache,
  )

def build_SA_V_v2v_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
  """Build SA_V V2V Dataset.
  Args:
      cfg (dict): Configuration dictionary
      img_path (str): Path to the image file
      batch (int): Batch size
      data (dict): Data dictionary
      mode (str): Mode (train or val)
      rect (bool): Rectangular batches
      stride (int): Stride
  """
  dataset = SA_V_V2VDataset
  return dataset(
      img_path=img_path,
      imgsz=cfg.imgsz,
      batch_size=batch,
      augment=(mode == "train"),  # augmentation
      template_hyp=cfg,
      hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
      rect=cfg.rect or rect,  # rectangular batches
      cache=cfg.cache or None,
      single_cls=cfg.single_cls or False,
      stride=int(stride),
      pad=0.0 if (mode == "train") else 0.5,
      prefix=colorstr(f"{mode}: "),
      task=cfg.task,
      classes=cfg.classes,
      data=data,
      fraction=cfg.fraction if (mode == "train") else 1.0,
      train= (mode == "train"),
  )
  
def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
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