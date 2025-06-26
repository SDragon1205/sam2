import itertools
import os
import sys
import logging

from v2vdet.v2vdet_ultralytics.nn import (WorldModel,
                                   v2vdetModel,
                                   v2vWorldModel,
                                   V2V_with_Patch_Attn_Pooling_Model, V2V_with_2_Patch_Attn_Pooling_Model,
V2V_multi_scale_clip_Model)
from v2vdet.v2vdet_ultralytics.data import build_v2v_dataset, build_SA_V_v2v_dataset
from transformers import AutoImageProcessor, Dinov2Model, CLIPVisionModelWithProjection, BatchFeature
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import concurrent
import torch
import numpy as np
import random
import os
from copy import copy
import pickle
import time
import warnings
import math
from copy import deepcopy
from torch import nn, optim
import torch.multiprocessing as mp
from functools import partial
from multiprocessing import Pool

from ultralytics.models import yolo
from ultralytics.models.yolo.world.train import on_pretrain_routine_end

from ultralytics import __version__
from ultralytics.utils import (
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    checks
)
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.data import build_yolo_dataset, YOLOConcatDataset, build_grounding
from ultralytics.data.utils import check_det_dataset

from v2vdet.v2vdet_ultralytics.utils import (extract_class_crops,
                                      count_trainable_parameters,
                                      crop_and_resize_largest_bbox_per_class,
                                      origin_crop_and_resize,
                                      random_crop_img)
from v2vdet.v2vdet_ultralytics.utils import DEFAULT_CFG
from v2vdet.v2vdet_ultralytics.nn.modules import (
    C2fAttn,
    ImagePoolingAttn,
    WorldDetect,
    C2f_v2v_Attn,
    TemplateAttentionPooling
)

from v2vdet.v2vdet_ultralytics.models.v2vdet.val import (v2v_DetectionValidator, v2v_with_attn_pooling_DetectionValidator
)

def v2v_on_pretrain_routine_end(trainer):
  """Callback."""
  if RANK in {-1, 0}:
    # NOTE: for evaluation
    names = [name.split(
        "/")[0] for name in list(trainer.test_loader.dataset.data["names"].values())]

    record = np.full(trainer.data["nc"], False, dtype=bool)
    crop_img_list = [None]*trainer.data["nc"]
    image_list = []

    dataset_shuffled = trainer.test_loader.dataset.labels.copy()
    random.shuffle(dataset_shuffled)

    for label in dataset_shuffled:
      classes_in_this_pic = np.unique(label['cls'])
      with Image.open(label['im_file']) as pil_img_:
        pil_img = pil_img_.copy()
        image_list.append(pil_img)
        width, height = pil_img.size
        for cls in classes_in_this_pic:
          int_cls = int(cls)
          if (record[int_cls] == False):
            indices = np.where(label['cls'] == int_cls)[0]
            random_idx = random.choice(indices)

            if (label['bbox_format'] == 'xywh'):
              if (label['normalized']):
                bbox_w, bbox_h = label['bboxes'][random_idx][2] * \
                    width, label['bboxes'][random_idx][3]*height
                x1, y1 = max(0, label['bboxes'][random_idx][0]*width-bbox_w //
                             2), max(0, label['bboxes'][random_idx][1]*height-bbox_h//2)
                x2, y2 = min(width, label['bboxes'][random_idx][0]*width+bbox_w//2), min(
                    height, label['bboxes'][random_idx][1]*height+bbox_h//2)
                box = (int(x1), int(y1), int(x2), int(y2))
              else:
                bbox_w, bbox_h = label['bboxes'][random_idx][2], label['bboxes'][random_idx][3]
                x1, y1 = max(0, label['bboxes'][random_idx][0]-bbox_w //
                             2), max(0, label['bboxes'][random_idx][1]-bbox_h//2)
                x2, y2 = min(width, label['bboxes'][random_idx][0]+bbox_w//2), min(
                    height, label['bboxes'][random_idx][1]+bbox_h//2)
                box = (int(x1), int(y1), int(x2), int(y2))

            elif (label['bbox_format'] == 'xyxy'):
              if (label['normalized']):
                x1, y1 = max(0, label['bboxes'][random_idx][0] *
                             width), max(0, label['bboxes'][random_idx][1]*height)
                x2, y2 = min(width, label['bboxes'][random_idx][2]*width), min(
                    height, label['bboxes'][random_idx][3]*height)
                box = (int(x1), int(y1), int(x2), int(y2))
              else:
                x1, y1 = max(0, label['bboxes'][random_idx][0]), max(
                    0, label['bboxes'][random_idx][1])
                x2, y2 = min(width, label['bboxes'][random_idx][2]), min(
                    height, label['bboxes'][random_idx][3])
                box = (int(x1), int(y1), int(x2), int(y2))

            else:
              raise NotImplementedError(
                  f"{label['bbox_format']} is not supported")
            box_area = abs(box[2]-box[0])*abs(box[3]-box[1])
            if (box_area < 100):
              box_list = [max(0, box[0]-10),
                          max(0, box[1]-10),
                          min(width, box[2]+10),
                          min(height, box[3]+10)]
              box = tuple(box_list)

            cropped_img = pil_img.crop(box)
            crop_img_list[int_cls] = cropped_img
            record[int_cls] = True

    crop_size = (128, 128)
    for crop_img_idx, crop_img in enumerate(crop_img_list):

      if crop_img is None:
        random_pil_img = random.choice(image_list)
        w, h = random_pil_img.size
        x1 = random.randint(0, w - crop_size[0])
        y1 = random.randint(0, h - crop_size[1])
        box = (x1, y1, x1 + crop_size[0], y1 + crop_size[1])

        crop_pic = random_pil_img.copy()
        crop_img_list[crop_img_idx] = crop_pic.crop(box)

    de_parallel(trainer.ema.ema).set_classes(crop_img_list)

  device = next(trainer.model.parameters()).device

  # trainer.text_model = trainer.vision_encoder
  # for p in trainer.vision_encoder.parameters():
  #   p.requires_grad_(False)

class v2vWorldTrainer(yolo.detect.DetectionTrainer):
  """
  A class to fine-tune a world model on a close-set dataset.

  Example:
      ```python
      from ultralytics.models.yolo.world import WorldModel

      args = dict(model="yolov8s-world.pt", data="coco8.yaml", epochs=3)
      trainer = WorldTrainer(overrides=args)
      trainer.train()
      ```
  """

  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)

  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    model = v2vWorldModel(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(False)
    model.model.requires_grad_(True)
    model.vision_encoder.requires_grad_(False)

    self.vision_encoder = model.vision_encoder

    # self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model

  def get_validator(self):
      """Returns a DetectionValidator for YOLO model validation."""
      self.loss_names = "box_loss", "cls_loss", "dfl_loss"
      copy_args = copy(self.args)
      copy_args.batch = 1
      test_loader = self.get_dataloader(
          self.testset, batch_size=copy_args.batch if self.args.task == "obb" else copy_args.batch, rank=-1, mode="val"
      )
      return v2v_DetectionValidator(
          test_loader, save_dir=self.save_dir, args=copy_args, _callbacks=self.callbacks
      )
      
  # def get_validator(self):
  #     """Returns a DetectionValidator for YOLO model validation."""
  #     self.loss_names = "box_loss", "cls_loss", "dfl_loss"
      
  #     return v2v_DetectionValidator(
  #         self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
  #     )

  def build_dataset(self, img_path, mode="train", batch=None):
    """
    Build YOLO Dataset. From trainer from scratch.

    Args:
        img_path (List[str] | str): Path to the folder containing images.
        mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
        batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
    """

    gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
    if mode != "train":
      return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
    dataset = [
        build_yolo_dataset(self.args, im_path, batch,
                           self.data, stride=gs, multi_modal=True)
        if isinstance(im_path, str)
        else build_grounding(self.args, im_path["img_path"], im_path["json_file"], batch, stride=gs)
        for im_path in img_path
    ]

    return YOLOConcatDataset(dataset) if len(dataset) > 1 else dataset[0]

  # def build_dataset(self, img_path, mode="train", batch=None):
  #   """
  #   Build V2V_Template_YOLO_Backbone_Share_Param Dataset.

  #   Args:
  #       img_path (List[str] | str): Path to the folder containing images.
  #       mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
  #       batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
  #   """

  #   gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
  #   dataset = [build_v2v_dataset(self.args,
  #                                     img_path,
  #                                     batch,
  #                                     self.data,
  #                                     mode,
  #                                     stride=gs)]

  #   return YOLOConcatDataset(dataset) if len(dataset) > 1 else dataset[0]

  def get_dataset(self):
    """
    Get train, val path from data dict if it exists.

    Returns None if data format is not recognized.
    """

    final_data = {}
    data_yaml = self.args.data
    assert data_yaml.get(
        "train", False), "train dataset not found"  # object365.yaml
    assert data_yaml.get(
        "val", False), "validation dataset not found"  # lvis.yaml
    data = {k: [check_det_dataset(d) for d in v.get(
        "yolo_data", [])] for k, v in data_yaml.items()}
    assert len(
        data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
    val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
    for d in data["val"]:
      if d.get("minival") is None:  # for lvis dataset
        continue
      d["minival"] = str(d["path"] / d["minival"])
    for s in ["train", "val"]:
      final_data[s] = [d["train" if s == "train" else val_split]
                       for d in data[s]]
      # save grounding data if there's one
      grounding_data = data_yaml[s].get("grounding_data")
      if grounding_data is None:
        continue
      grounding_data = grounding_data if isinstance(
          grounding_data, list) else [grounding_data]
      for g in grounding_data:
        assert isinstance(
            g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
      final_data[s] += grounding_data
    # NOTE: to make training work properly, set `nc` and `names`

    final_data["nc"] = data["val"][0]["nc"]
    final_data["names"] = data["val"][0]["names"]
    self.data = final_data
    return final_data["train"], final_data["val"][0]

  def process_single_image(self, img_data, batch, crop_size, num_classes, device, aug_params):
    img_idx, img = img_data

    # 為每個類別隨機選取批次中的影像並裁剪
    random_crop = transforms.RandomCrop(size=crop_size)
    batch_size = len(batch['img'])
    single_sample_crop_img = [random_crop(
      batch['img'][random.randint(0, batch_size-1)]) for _ in range(num_classes)]

    # 尋找當前影像的相關資訊
    matches = (batch['batch_idx'] == img_idx).nonzero()
    if len(matches) > 0:
      batch_start = (batch['batch_idx'] == img_idx).nonzero()[0].item()
      batch_count = (batch['batch_idx'] == img_idx).sum().item()
      img_classes = batch['cls'][batch_start:batch_start + batch_count]
      img_boxes = batch['bboxes'][batch_start:batch_start + batch_count]

      # 執行最佳化的裁剪和大小調整
      cropped_positives = origin_crop_and_resize(
        img, img_boxes, classes=img_classes, size=crop_size, augment=True, aug_params=aug_params
      )

      # 儲存處理後的結果
      for crop_data in cropped_positives:
        single_sample_crop_img[int(crop_data['cls'])] = crop_data['crop_tensor_img'].to(device)

    return single_sample_crop_img

  # 主函數: 將原始 for 迴圈改為平行處理
  def parallel_process_images(self, batch, crop_size, num_classes, device, num_workers=None):
    # 如果未指定工作進程數，則使用可用 CPU 核心數量
    if num_workers is None:
      num_workers = mp.cpu_count()

    # 定義資料增強參數
    aug_params = {
      'rotation_range': (-30, 30),
      'scale_range': (0.5, 2),
      'brightness_range': (0.8, 1.2),
      'contrast_range': (0.8, 1.2),
      'prob': 1,
      'global_prob': 1
    }

    # 準備輸入數據
    batch_size = len(batch['img'])
    img_data_list = [(img_idx, img) for img_idx, img in enumerate(batch['img'])]

    # 建立部分函數，固定大部分參數
    process_func = partial(
      self.process_single_image,
      batch=batch,
      crop_size=crop_size,
      num_classes=num_classes,
      device=device,
      aug_params=aug_params
    )

    # 使用 spawn 方法（對 CUDA 兼容性更好）
    # 如果程式碼在初始化階段只運行一次，請在外部設定此項
    try:
      mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 方法已經設定，忽略錯誤
      pass

    # 使用進程池平行處理
    crop_img_list = []
    with mp.Pool(processes=num_workers) as pool:
      results = pool.map(process_func, img_data_list)

        # 合併所有結果
      for result in results:
        crop_img_list.extend(result)

    return crop_img_list

  def old_preprocess_batch(self, batch, crop_size=(224, 224)):
    """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""

    batch = super().preprocess_batch(batch)
    # NOTE: add template matching features

    num_classes = self.model.yaml['nc']  # Usually 80
    batch_size = len(batch['img'])

    # Prepare storage for batch embeddings
    crop_img_list = []
    random_crop = transforms.RandomCrop(size=crop_size)

    for img_idx, img in enumerate(batch['img']):

      single_sample_crop_img = [random_crop(
          batch['img'][random.randint(0, batch_size-1)]) for _ in range(num_classes)]

      # single_sample_crop_img = [torch.zeros((3, 224, 224)).to(self.device)]*num_classes

      matches = (batch['batch_idx'] == img_idx).nonzero()
      if len(matches) > 0:
        batch_start = (batch['batch_idx'] == img_idx).nonzero()[0].item()
        batch_count = (batch['batch_idx'] == img_idx).sum().item()
        img_classes = batch['cls'][batch_start:batch_start + batch_count]
        img_boxes = batch['bboxes'][batch_start:batch_start + batch_count]

        aug_params = {
            'rotation_range': (-30, 30),
            'scale_range': (0.5, 2),
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.8, 1.2),
            'prob': 0.5,
            'global_prob': 0.7
        }

        cropped_positives = origin_crop_and_resize(
            img, img_boxes, classes=img_classes, size=crop_size, augment=True, aug_params=aug_params
        )

        for crop_data in cropped_positives:
          single_sample_crop_img[int(
              crop_data['cls'])] = crop_data['crop_tensor_img'].to(self.device)

      crop_img_list.extend(single_sample_crop_img)

    crop_img_tensor = torch.stack(tensors=crop_img_list)
    batch["template_feats"] = crop_img_tensor.to(self.device)
    return batch
  
  @staticmethod
  def _multiprocessing_preprocess_batch(args):
    input_dict = args
    
    if input_dict['batch'] is None:
      raise ValueError("batch is required")
    if input_dict['idx'] is None:
      raise ValueError("idx is required")
    
    img_idx = input_dict['idx']
    batch = input_dict['batch']
    crop_size = input_dict['crop_size'] if not None else (224, 224)
    num_classes = input_dict['num_classes'] if not None else 80
    random_crop = input_dict['random_crop'] if not None else transforms.RandomCrop(size=crop_size)
    aug_params =  input_dict['aug_params'] if not None else {
            'rotation_range': (-30, 30),
            'scale_range': (0.5, 2),
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.8, 1.2),
            'prob': 0.5,
            'global_prob': 0.7
        }

    batch_size = len(batch['img']) 
    single_sample_crop_img = [random_crop(
    batch['img'][random.randint(0, batch_size-1)]) for _ in range(num_classes)]

    matches = (batch['batch_idx'] == img_idx).nonzero()
    if len(matches) > 0:
      batch_start = (batch['batch_idx'] == img_idx).nonzero()[0].item()
      batch_count = (batch['batch_idx'] == img_idx).sum().item()
      img_classes = batch['cls'][batch_start:batch_start + batch_count]
      img_boxes = batch['bboxes'][batch_start:batch_start + batch_count]

      cropped_positives = origin_crop_and_resize(
            batch['img'][img_idx], img_boxes, classes=img_classes, size=crop_size, augment=True, aug_params=aug_params
      )

      for crop_data in cropped_positives:
        single_sample_crop_img[int(crop_data['cls'])] = crop_data['crop_tensor_img']
    
    return torch.stack(single_sample_crop_img, dim=0)
  
  def preprocess_batch(self, batch, crop_size=(224, 224)):
    """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""

    batch = super().preprocess_batch(batch)
    # NOTE: add template matching features

    if hasattr(self.model, 'module'):
      num_classes = self.model.module.yaml['nc']
    else:
      num_classes = self.model.yaml['nc']  # Usually 80
    batch_size = len(batch['img'])

    # Prepare storage for batch embeddings
    crop_img_list = []
    random_crop = transforms.RandomCrop(size=crop_size)
    multi_processing_batch = deepcopy(batch)
    multi_processing_batch['img'] = [img.to('cpu') for img in batch['img']]

    multiprocessing_input_list = [{
      'idx': idx,
      'batch': multi_processing_batch,
      'crop_size': crop_size,
      'num_classes': num_classes,
      'random_crop': random_crop,
      'aug_params': {
        'rotation_range': (-30, 30),
        'scale_range': (0.5, 2),
        'brightness_range': (0.8, 1.2),
        'contrast_range': (0.8, 1.2),
        'prob': 0.5,
        'global_prob': 0.7
      }
    } for idx in range(batch_size)]
    # multiprocessing_input_list = list(enumerate(multiprocessing_input_list))
    
    # with Pool(processes=self.train_loader.num_workers) as pool:
    # if want multi processing
    # crop_img_list = self.multiprocessing_pool.map(self._multiprocessing_preprocess_batch, multiprocessing_input_list)
    # dont want multi processing
    for input_list in multiprocessing_input_list:
      crop_img_list.append(self._multiprocessing_preprocess_batch(input_list))

    batch["template_feats"] = torch.stack(tensors=crop_img_list).flatten(0, 1)
    return batch

  def _setup_train(self, world_size):
    """Builds dataloaders and optimizer on correct rank process."""
    # Model
    self.run_callbacks("on_pretrain_routine_start")
    ckpt = self.setup_model()
    self.temp_ckpt = ckpt
    self.model = self.model.to(self.device)
    self.set_model_attributes()

    # Freeze layers
    freeze_list = (
        self.args.freeze
        if isinstance(self.args.freeze, list)
        else range(self.args.freeze)
        if isinstance(self.args.freeze, int)
        else []
    )

    always_freeze_names = [".dfl"]  # always freeze these layers
    freeze_layer_names = [
        f"model.{x}." for x in freeze_list] + always_freeze_names
    for k, v in self.model.named_parameters():
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
      if any(x in k for x in freeze_layer_names):
        LOGGER.info(f"Freezing layer '{k}'")
        v.requires_grad = False

    # Check AMP
    self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
    if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
      # backup callbacks as check_amp() resets them
      callbacks_backup = callbacks.default_callbacks.copy()
      self.amp = torch.tensor(check_amp(self.model), device=self.device)
      callbacks.default_callbacks = callbacks_backup  # restore callbacks
    if RANK > -1 and world_size > 1:  # DDP
      # broadcast the tensor from rank 0 to all other ranks (returns None)
      # torch.dist.broadcast(self.amp, src=0)
      torch.distributed.broadcast(self.amp, src=0)
    self.amp = bool(self.amp)  # as boolean
    self.scaler = (
        torch.amp.GradScaler(
            "cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
    )
    if world_size > 1:
      self.model = torch.nn.parallel.DistributedDataParallel(
          self.model, device_ids=[RANK], find_unused_parameters=True)

    # Check imgsz
    gs = max(int(self.model.stride.max() if hasattr(
        self.model, "stride") else 32), 32)  # grid size (max stride)
    self.args.imgsz = check_imgsz(
        self.args.imgsz, stride=gs, floor=gs, max_dim=1)
    self.stride = gs  # for multiscale training

    # Batch size
    if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
      self.args.batch = self.batch_size = self.auto_batch()

    # Dataloaders
    batch_size = self.batch_size // max(world_size, 1)
    self.train_loader = self.get_dataloader(
        self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
    if RANK in {-1, 0}:
      # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
      self.test_loader = self.get_dataloader(
          self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size, rank=-1, mode="val"
      )
      self.validator = self.get_validator()
      metric_keys = self.validator.metrics.keys + \
          self.label_loss_items(prefix="val")
      self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
      self.ema = ModelEMA(self.model)
      if self.args.plots:
        self.plot_training_labels()

    # Optimizer
    # accumulate loss before optimizing
    self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
    weight_decay = self.args.weight_decay * self.batch_size * \
        self.accumulate / self.args.nbs  # scale weight_decay
    iterations = math.ceil(len(self.train_loader.dataset) /
                           max(self.batch_size, self.args.nbs)) * self.epochs
    self.optimizer = self.build_optimizer(
        model=self.model,
        name=self.args.optimizer,
        lr=self.args.lr0,
        momentum=self.args.momentum,
        decay=weight_decay,
        iterations=iterations,
    )

    # Scheduler

    self._setup_scheduler()
    self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
    self.resume_training(ckpt)
    self.scheduler.last_epoch = self.start_epoch - 1  # do not move
    # self.run_callbacks("on_pretrain_routine_end")

  def _do_train(self, world_size=1):
    """Train completed, evaluate and plot if specified by arguments."""
    if world_size > 1:
      self._setup_ddp(world_size)
    self._setup_train(world_size)

    nb = len(self.train_loader)  # number of batches
    nw = max(round(self.args.warmup_epochs * nb),
             100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
    last_opt_step = -1
    self.epoch_time = None
    self.epoch_time_start = time.time()
    self.train_time_start = time.time()
    self.run_callbacks("on_train_start")
    LOGGER.info(
        f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
        f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
        f"Logging results to {colorstr('bold', self.save_dir)}\n"
        f'Starting training for ' +
        (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
    )
    if self.args.close_mosaic:
      base_idx = (self.epochs - self.args.close_mosaic) * nb
      self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
    epoch = self.start_epoch
    # zero any resumed gradients to ensure stability on train start
    self.optimizer.zero_grad()
    while True:
      self.epoch = epoch
      self.run_callbacks("on_train_epoch_start")
      with warnings.catch_warnings():
        # suppress 'Detected lr_scheduler.step() before optimizer.step()'
        warnings.simplefilter("ignore")
        self.scheduler.step()

      self.model.train()
      # self.model.half()

      if RANK != -1:
        self.train_loader.sampler.set_epoch(epoch)
      pbar = enumerate(self.train_loader)

      # Update dataloader attributes (optional)
      if epoch == (self.epochs - self.args.close_mosaic):
        self._close_dataloader_mosaic()
        self.train_loader.reset()

      if RANK in {-1, 0}:
        LOGGER.info(self.progress_string())
        pbar = TQDM(enumerate(self.train_loader), total=nb)
      self.tloss = None
      self.start_time_for_template_cache = time.time()

      # self.multiprocessing_pool = Pool(processes=self.train_loader.num_workers)
      for i, batch in pbar:
        self.run_callbacks(event="on_train_batch_start")
        # self.optimizer.zero_grad(set_to_none=True)

        # Warmup
        ni = i + nb * epoch
        if ni <= nw:
          xi = [0, nw]  # x interp
          self.accumulate = max(
              1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
          for j, x in enumerate(self.optimizer.param_groups):
            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x["lr"] = np.interp(
                ni, xi, [self.args.warmup_bias_lr if j ==
                         0 else 0.0, x["initial_lr"] * self.lf(epoch)]
            )
            if "momentum" in x:
              x["momentum"] = np.interp(
                  ni, xi, [self.args.warmup_momentum, self.args.momentum])

        # Forward
        # with autocast(self.amp):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
          batch = self.preprocess_batch(batch=batch)
          self.loss, self.loss_items = self.model(batch)
          if RANK != -1:
            self.loss *= world_size
          self.tloss = (
              (self.tloss * i + self.loss_items) /
              (i + 1) if self.tloss is not None else self.loss_items
          )

        # Backward
        self.scaler.scale(self.loss).backward()
        # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html

        if ni - last_opt_step >= self.accumulate:
          self.optimizer_step()
          last_opt_step = ni

          # Timed stopping
          if self.args.time:
            self.stop = (
              time.time() - self.train_time_start) > (self.args.time * 3600)
            if RANK != -1:  # if DDP training
              broadcast_list = [self.stop if RANK == 0 else None]
              # broadcast 'stop' to all ranks
              # torch.dist.broadcast_object_list(broadcast_list, 0)
              torch.distributed.broadcast_object_list(broadcast_list, 0)
              self.stop = broadcast_list[0]
            if self.stop:  # training time exceeded
              break

        # Log
        if RANK in {-1, 0}:
          loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
          pbar.set_description(
              ("%11s" * 2 + "%11.4g" * (2 + loss_length))
              % (
                  f"{epoch + 1}/{self.epochs}",
                  f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                  *(self.tloss if loss_length >
                    1 else torch.unsqueeze(self.tloss, 0)),  # losses
                  batch["cls"].shape[0],  # batch size, i.e. 8
                  batch["img"].shape[-1],  # imgsz, i.e 640
              )
          )
          try:
            import wandb
            wandb.log({
              "train/box_loss": self.loss_items[0].detach().cpu().item(),
              "train/cls_loss": self.loss_items[1].detach().cpu().item(),
              "train/obj_loss": self.loss_items[2].detach().cpu().item(),
                })
          except:
            pass
          self.run_callbacks("on_batch_end")
          if self.args.plots and ni in self.plot_idx:
            self.plot_training_samples(batch, ni)

        # torch.cuda.empty_cache()
        self.run_callbacks("on_train_batch_end")
      
      # self.multiprocessing_pool.close()
      # self.multiprocessing_pool.join()

      self.lr = {f"lr/pg{ir}": x["lr"]
                 for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

      self.run_callbacks("on_train_epoch_end")
      torch.cuda.empty_cache()
      self.model.eval()
      if RANK in {-1, 0}:
        final_epoch = epoch + 1 >= self.epochs
        self.ema.update_attr(self.model, include=[
                             "yaml", "nc", "args", "names", "stride", "class_weights"])

        # Validation
        if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
          with torch.inference_mode():
            self.metrics, self.fitness = self.validate()

        self.save_metrics(
            metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
        self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
        if self.args.time:
          self.stop |= (
              time.time() - self.train_time_start) > (self.args.time * 3600)

        try:
          import wandb
          wandb.log({
            "epoch": epoch,
            **self.metrics
          })
          wandb.log({"epoch": epoch,
                    **self.lr}, step=epoch)
        except:
          pass

        # Save model
        if self.args.save or final_epoch:
          self.ema.ema = self.ema.ema.cpu()
          self.model = self.model.cpu()

          self.save_model()
          self.run_callbacks("on_model_save")

          self.ema.ema = self.ema.ema.cuda()
          self.model = self.model.cuda()

      # Scheduler
      t = time.time()
      self.epoch_time = t - self.epoch_time_start
      self.epoch_time_start = t
      if self.args.time:
        mean_epoch_time = (t - self.train_time_start) / \
            (epoch - self.start_epoch + 1)
        self.epochs = self.args.epochs = math.ceil(
            self.args.time * 3600 / mean_epoch_time)
        self._setup_scheduler()
        self.scheduler.last_epoch = self.epoch  # do not move
        self.stop |= epoch >= self.epochs  # stop if exceeded epochs
      self.run_callbacks("on_fit_epoch_end")
      self._clear_memory()

      # Early Stopping
      if RANK != -1:  # if DDP training
        broadcast_list = [self.stop if RANK == 0 else None]
        # broadcast 'stop' to all ranks
        # torch.dist.broadcast_object_list(broadcast_list, 0)
        torch.distributed.broadcast_object_list(broadcast_list, 0)
        self.stop = broadcast_list[0]
      if self.stop:
        break  # must break all DDP ranks
      epoch += 1

    if RANK in {-1, 0}:
      # Do final val with best.pt
      seconds = time.time() - self.train_time_start
      LOGGER.info(
          f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
      self.final_eval()
      if self.args.plots:
        self.plot_metrics()
      self.run_callbacks("on_train_end")
    self._clear_memory()
    self.run_callbacks("teardown")

  def save_model(self):
    """Save model training checkpoints with additional metadata."""
    import io
    from copy import copy, deepcopy
    from datetime import datetime, timedelta

    # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
    buffer = io.BytesIO()
    torch.save(
        {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            # resume and final checkpoints derive from EMA
            "model": None,
            "model_state_dict": deepcopy(self.model.state_dict()),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            "train_args": vars(self.args),  # save as dict
            "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
            "train_results": self.read_results_csv(),
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://si2lab.iee.nycu.edu.tw/",
        },
        buffer,
    )
    serialized_ckpt = buffer.getvalue()  # get the serialized content to save

    # Save checkpoints
    self.last.write_bytes(serialized_ckpt)  # save last.pt
    if self.best_fitness == self.fitness:
      self.best.write_bytes(serialized_ckpt)  # save best.pt
    if (self.save_period > 0) and (self.epoch % self.save_period == 0):
      # save epoch, i.e. 'epoch3.pt'
      (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
    # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
    #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # save mosaic checkpoint
  
  def final_eval(self):
    """Performs final evaluation and validation for object detection YOLO model."""
    ckpt = {}
    for f in self.last, self.best:
      if f.exists():
        if f is self.last:
          ckpt = strip_optimizer(f)
        elif f is self.best:
          k = "train_results"  # update best.pt train_metrics from last.pt
          strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
          # LOGGER.info(f"\nValidating {f}...")
          # self.validator.args.plots = self.args.plots
          # self.metrics = self.validator(model=f)
          # self.metrics.pop("fitness", None)
          # self.run_callbacks("on_fit_epoch_end")


class v2vWorldTrainerFromScratch(v2vWorldTrainer):
  """
  A class extending the WorldTrainer class for training a world model from scratch on open-set dataset.

  Example:
      ```python
      from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
      from ultralytics import YOLOWorld

      data = dict(
          train=dict(
              yolo_data=["Objects365.yaml"],
              grounding_data=[
                  dict(
                      img_path="../datasets/flickr30k/images",
                      json_file="../datasets/flickr30k/final_flickr_separateGT_train.json",
                  ),
                  dict(
                      img_path="../datasets/GQA/images",
                      json_file="../datasets/GQA/final_mixed_train_no_coco.json",
                  ),
              ],
          ),
          val=dict(yolo_data=["lvis.yaml"]),
      )

      model = YOLOWorld("yolov8s-worldv2.yaml")
      model.train(data=data, trainer=WorldTrainerFromScratch)
      ```
  """

  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)

  def build_dataset(self, img_path, mode="train", batch=None):
    """
    Build YOLO Dataset.

    Args:
        img_path (List[str] | str): Path to the folder containing images.
        mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
        batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
    """

    gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
    if mode != "train":
      return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
    dataset = [
        build_yolo_dataset(self.args, im_path, batch,
                           self.data, stride=gs, multi_modal=True)
        if isinstance(im_path, str)
        else build_grounding(self.args, im_path["img_path"], im_path["json_file"], batch, stride=gs)
        for im_path in img_path
    ]

    return YOLOConcatDataset(dataset) if len(dataset) > 1 else dataset[0]

  def get_dataset(self):
    """
    Get train, val path from data dict if it exists.

    Returns None if data format is not recognized.
    """

    final_data = {}
    data_yaml = self.args.data
    assert data_yaml.get(
        "train", False), "train dataset not found"  # object365.yaml
    assert data_yaml.get(
        "val", False), "validation dataset not found"  # lvis.yaml
    data = {k: [check_det_dataset(d) for d in v.get(
        "yolo_data", [])] for k, v in data_yaml.items()}
    assert len(
        data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
    val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
    for d in data["val"]:
      if d.get("minival") is None:  # for lvis dataset
        continue
      d["minival"] = str(d["path"] / d["minival"])
    for s in ["train", "val"]:
      final_data[s] = [d["train" if s == "train" else val_split]
                       for d in data[s]]
      # save grounding data if there's one
      grounding_data = data_yaml[s].get("grounding_data")
      if grounding_data is None:
        continue
      grounding_data = grounding_data if isinstance(
          grounding_data, list) else [grounding_data]
      for g in grounding_data:
        assert isinstance(
            g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
      final_data[s] += grounding_data
    # NOTE: to make training work properly, set `nc` and `names`

    final_data["nc"] = data["val"][0]["nc"]
    final_data["names"] = data["val"][0]["names"]
    self.data = final_data
    return final_data["train"], final_data["val"][0]

  def plot_training_labels(self):
    """DO NOT plot labels."""
    pass

  def final_eval(self):
    """Performs final evaluation and validation for object detection YOLO-World model."""
    val = self.args.data["val"]["yolo_data"][0]
    self.validator.args.data = val
    self.validator.args.split = "minival" if isinstance(
        val, str) and "lvis" in val else "val"
    return super().final_eval()


class v2vWorld_Attn_Pooling_Trainer(v2vWorldTrainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)
    # self.attn_pooling = TemplateAttentionPooling(
    #     hidden_size=768, proj_size=512).to(self.device)

  def preprocess_batch(self, batch):
    """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""

    batch = super().preprocess_batch(batch)

    return batch

  def get_model(self, cfg=None, weights=None, verbose=True):

    model = V2V_with_Patch_Attn_Pooling_Model(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.model.requires_grad_(False)
    model.vision_encoder.requires_grad_(False)
    model.attn_pooling.requires_grad_(True)
    self.vision_encoder = model.vision_encoder
    self.attn_pooling = model.attn_pooling

    self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model

  def get_dataset(self):
    """
    Get train, val path from data dict if it exists.

    Returns None if data format is not recognized.
    """

    final_data = {}
    data_yaml = self.args.data
    assert data_yaml.get(
        "train", False), "train dataset not found"  # object365.yaml
    assert data_yaml.get(
        "val", False), "validation dataset not found"  # lvis.yaml
    data = {k: [check_det_dataset(d) for d in v.get(
        "yolo_data", [])] for k, v in data_yaml.items()}
    assert len(
        data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
    val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
    for d in data["val"]:
      if d.get("minival") is None:  # for lvis dataset
        continue
      d["minival"] = str(d["path"] / d["minival"])
    for s in ["train", "val"]:
      final_data[s] = [d["train" if s == "train" else val_split]
                       for d in data[s]]
      # save grounding data if there's one
      grounding_data = data_yaml[s].get("grounding_data")
      if grounding_data is None:
        continue
      grounding_data = grounding_data if isinstance(
          grounding_data, list) else [grounding_data]
      for g in grounding_data:
        assert isinstance(
            g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
      final_data[s] += grounding_data
    # NOTE: to make training work properly, set `nc` and `names`

    final_data["nc"] = data["val"][0]["nc"]
    final_data["names"] = data["val"][0]["names"]
    self.data = final_data
    return final_data["train"], final_data["val"][0]

  def get_validator(self):
    """Returns a DetectionValidator for YOLO model validation."""
    self.loss_names = "box_loss", "cls_loss", "dfl_loss"
    return v2v_with_attn_pooling_DetectionValidator(
        self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
    )

  def _do_train(self, world_size=1):
    """Train completed, evaluate and plot if specified by arguments."""
    if world_size > 1:
      self._setup_ddp(world_size)

    self._setup_train(world_size)

    # if hasattr(self.ema.ema, 'vision_encoder'):
    #     self.ema.ema.vision_encoder = None

    nb = len(self.train_loader)  # number of batches
    nw = max(round(self.args.warmup_epochs * nb),
             100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
    last_opt_step = -1
    self.epoch_time = None
    self.epoch_time_start = time.time()
    self.train_time_start = time.time()
    self.run_callbacks("on_train_start")
    LOGGER.info(
        f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
        f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
        f"Logging results to {colorstr('bold', self.save_dir)}\n"
        f'Starting training for ' +
        (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
    )

    if self.args.close_mosaic:
      base_idx = (self.epochs - self.args.close_mosaic) * nb
      self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
    epoch = self.start_epoch
    # zero any resumed gradients to ensure stability on train start
    self.optimizer.zero_grad()
    while True:
      self.epoch = epoch
      self.run_callbacks("on_train_epoch_start")
      with warnings.catch_warnings():
        # suppress 'Detected lr_scheduler.step() before optimizer.step()'
        warnings.simplefilter("ignore")
        self.scheduler.step()

      # self.model.train()
      self.model.requires_grad_(False)
      self.vision_encoder.requires_grad_(False)
      self.attn_pooling.requires_grad_(True)
      self.model.train()

      if RANK != -1:
        self.train_loader.sampler.set_epoch(epoch)
      pbar = enumerate(self.train_loader)
      # Update dataloader attributes (optional)
      if epoch == (self.epochs - self.args.close_mosaic):
        self._close_dataloader_mosaic()
        self.train_loader.reset()

      if RANK in {-1, 0}:
        LOGGER.info(self.progress_string())
        pbar = TQDM(enumerate(self.train_loader), total=nb)

      self.tloss = None
      for i, batch in pbar:
        self.run_callbacks("on_train_batch_start")

        self.optimizer.zero_grad()

        # Warmup
        ni = i + nb * epoch
        if ni <= nw:
          xi = [0, nw]  # x interp
          self.accumulate = max(
              1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
          for j, x in enumerate(self.optimizer.param_groups):
            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x["lr"] = np.interp(
                ni, xi, [self.args.warmup_bias_lr if j ==
                         0 else 0.0, x["initial_lr"] * self.lf(epoch)]
            )
            if "momentum" in x:
              x["momentum"] = np.interp(
                  ni, xi, [self.args.warmup_momentum, self.args.momentum])

        # Forward
        with torch.autocast(device_type="cuda", enabled=self.amp):

          batch = self.preprocess_batch(batch)

          attn_pooling_input = self.hidden_states[-3].clone().to(self.device)
          self.attn_pooling_result = self.attn_pooling(attn_pooling_input)
          pooled_feature_proj = self.attn_pooling_result['pooled_feature_proj'].reshape(
              -1, self.model.model[-1].nc, self.attn_pooling_result['pooled_feature_proj'].shape[-1])
          batch["txt_feats"] = pooled_feature_proj
          self.loss, self.loss_items = self.model(batch)

          if RANK != -1:
            self.loss *= world_size
          self.tloss = (
              (self.tloss * i + self.loss_items) /
              (i + 1) if self.tloss is not None else self.loss_items
          )

        # Backward
        self.scaler.scale(self.loss).backward()

        # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
        if ni - last_opt_step >= self.accumulate:
          self.optimizer_step()
          last_opt_step = ni

          # Timed stopping
          if self.args.time:
            self.stop = (
                time.time() - self.train_time_start) > (self.args.time * 3600)
            if RANK != -1:  # if DDP training
              broadcast_list = [self.stop if RANK == 0 else None]
              # broadcast 'stop' to all ranks
              # torch.dist.broadcast_object_list(broadcast_list, 0)
              torch.distributed.broadcast_object_list(broadcast_list, 0)
              
              self.stop = broadcast_list[0]
            if self.stop:  # training time exceeded
              break

        # Log
        if RANK in {-1, 0}:
          loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
          pbar.set_description(
              ("%11s" * 2 + "%11.4g" * (2 + loss_length))
              % (
                  f"{epoch + 1}/{self.epochs}",
                  f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                  *(self.tloss if loss_length >
                    1 else torch.unsqueeze(self.tloss, 0)),  # losses
                  batch["cls"].shape[0],  # batch size, i.e. 8
                  batch["img"].shape[-1],  # imgsz, i.e 640
              )
          )
          self.run_callbacks("on_batch_end")
          if self.args.plots and ni in self.plot_idx:
            self.plot_training_samples(batch, ni)

        try:
          import wandb
          wandb.log({
            "train/box_loss": self.loss_items[0].detach().cpu().item(),
            "train/cls_loss": self.loss_items[1].detach().cpu().item(),
            "train/obj_loss": self.loss_items[2].detach().cpu().item(),
              })
        except:
          pass

        # torch.cuda.empty_cache()
        self.run_callbacks("on_train_batch_end")
        del batch

      self.lr = {f"lr/pg{ir}": x["lr"]
                 for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

      self.run_callbacks("on_train_epoch_end")
      if RANK in {-1, 0}:
        final_epoch = epoch + 1 >= self.epochs
        self.ema.update_attr(self.model, include=[
                             "yaml", "nc", "args", "names", "stride", "class_weights"])

        # Validation
        if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
          with torch.inference_mode():
            self.metrics, self.fitness = self.validate()

        self.save_metrics(
            metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
        self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
        if self.args.time:
          self.stop |= (
              time.time() - self.train_time_start) > (self.args.time * 3600)

        try:
          import wandb
          wandb.log({
            "epoch": epoch,
            **self.metrics
          })
          wandb.log({"epoch": epoch,
                    **self.lr}, step=epoch)
        except:
          pass

        # Save model
        if self.args.save or final_epoch:
          self.ema.ema = self.ema.ema.cpu()
          self.model = self.model.cpu()

          self.save_model()
          self.run_callbacks("on_model_save")

          self.ema.ema = self.ema.ema.cuda()
          self.model = self.model.cuda()

      # Scheduler
      t = time.time()
      self.epoch_time = t - self.epoch_time_start
      self.epoch_time_start = t
      if self.args.time:
        mean_epoch_time = (t - self.train_time_start) / \
            (epoch - self.start_epoch + 1)
        self.epochs = self.args.epochs = math.ceil(
            self.args.time * 3600 / mean_epoch_time)
        self._setup_scheduler()
        self.scheduler.last_epoch = self.epoch  # do not move
        self.stop |= epoch >= self.epochs  # stop if exceeded epochs
      self.run_callbacks("on_fit_epoch_end")
      self._clear_memory()

      # Early Stopping
      if RANK != -1:  # if DDP training
        broadcast_list = [self.stop if RANK == 0 else None]
        # broadcast 'stop' to all ranks
        # torch.dist.broadcast_object_list(broadcast_list, 0)
        torch.distributed.broadcast_object_list(broadcast_list, 0)
        self.stop = broadcast_list[0]
      if self.stop:
        break  # must break all DDP ranks
      epoch += 1
      del self.loss, self.loss_items

    if RANK in {-1, 0}:
      # Do final val with best.pt
      seconds = time.time() - self.train_time_start
      LOGGER.info(
          f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
      self.final_eval()
      if self.args.plots:
        self.plot_metrics()
      self.run_callbacks("on_train_end")
    self._clear_memory()
    self.run_callbacks("teardown")

class V2V_with_2_Patch_Attn_Pooling_Trainer(v2vWorldTrainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)

  def preprocess_batch(self, batch):
    """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""

    batch = super().preprocess_batch(batch)

    return batch

  def get_model(self, cfg=None, weights=None, verbose=True):

    model = V2V_with_2_Patch_Attn_Pooling_Model(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.model.requires_grad_(False)
    model.vision_encoder.requires_grad_(False)
    for temp_idx in range(2):
      model.attn_pooling[temp_idx].requires_grad_(True)
    self.vision_encoder = model.vision_encoder
    self.attn_pooling = model.attn_pooling

    self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model

  def get_dataset(self):
    """
    Get train, val path from data dict if it exists.

    Returns None if data format is not recognized.
    """

    final_data = {}
    data_yaml = self.args.data
    assert data_yaml.get(
        "train", False), "train dataset not found"  # object365.yaml
    assert data_yaml.get(
        "val", False), "validation dataset not found"  # lvis.yaml
    data = {k: [check_det_dataset(d) for d in v.get(
        "yolo_data", [])] for k, v in data_yaml.items()}
    assert len(
        data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
    val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
    for d in data["val"]:
      if d.get("minival") is None:  # for lvis dataset
        continue
      d["minival"] = str(d["path"] / d["minival"])
    for s in ["train", "val"]:
      final_data[s] = [d["train" if s == "train" else val_split]
                       for d in data[s]]
      # save grounding data if there's one
      grounding_data = data_yaml[s].get("grounding_data")
      if grounding_data is None:
        continue
      grounding_data = grounding_data if isinstance(
          grounding_data, list) else [grounding_data]
      for g in grounding_data:
        assert isinstance(
            g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
      final_data[s] += grounding_data
    # NOTE: to make training work properly, set `nc` and `names`

    final_data["nc"] = data["val"][0]["nc"]
    final_data["names"] = data["val"][0]["names"]
    self.data = final_data
    return final_data["train"], final_data["val"][0]

class V2V_multi_scale_clip_Trainer(v2vWorldTrainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)

  def preprocess_batch(self, batch):
    """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""

    batch = super().preprocess_batch(batch)

    return batch

  def get_model(self, cfg=None, weights=None, verbose=True):

    model = V2V_multi_scale_clip_Model(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(False)
    model.multi_scale_attn_pooling.requires_grad_(True)
    model.model.requires_grad_(True)
    model.vision_encoder.requires_grad_(False)

    self.vision_encoder = model.vision_encoder
    self.multi_scale_attn_pooling = model.multi_scale_attn_pooling

    self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model

  def _setup_train(self, world_size):
    """Builds dataloaders and optimizer on correct rank process."""
    super()._setup_train(world_size)

  def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
    """
    Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
    weight decay, and number of iterations.

    Args:
        model (torch.nn.Module): The model for which to build an optimizer.
        name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
            based on the number of iterations. Default: 'auto'.
        lr (float, optional): The learning rate for the optimizer. Default: 0.001.
        momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
        decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
        iterations (float, optional): The number of iterations, which determines the optimizer if
            name is 'auto'. Default: 1e5.

    Returns:
        (torch.optim.Optimizer): The constructed optimizer.
    """
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    if name == "auto":
      LOGGER.info(
          f"{colorstr('optimizer:')} 'optimizer=auto' found, "
          f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
          f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
      )
      nc = getattr(model, "nc", 10)  # number of classes
      lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
      name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
      self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

    # YOLO Part
    for module_name, module in model.model.named_modules():
      for param_name, param in module.named_parameters(recurse=False):
        fullname = f"{module_name}.{param_name}" if module_name else param_name
        if "bias" in fullname:  # bias (no decay)
          g[2].append(param)
        elif isinstance(module, bn):  # weight (no decay)
          g[1].append(param)
        else:  # weight (with decay)
          g[0].append(param)

    optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
    name = {x.lower(): x for x in optimizers}.get(name.lower())
    if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
      optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
      optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
      optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
      raise NotImplementedError(
          f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
          "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
      )

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(
      f"{colorstr('YOLO optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
      f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
    )


    # Multi-Scale Attention Pooling Part
    g = [], [], []  # optimizer parameter groups
    lr = 0.01
    for module_name, module in model.multi_scale_attn_pooling.named_modules():
      for param_name, param in module.named_parameters(recurse=False):
        fullname = f"{module_name}.{param_name}" if module_name else param_name
        if "bias" in fullname:  # bias (no decay)
          g[2].append(param)
        elif isinstance(module, bn):  # weight (no decay)
          g[1].append(param)
        else:  # weight (with decay)
          g[0].append(param)

    optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
    name = {x.lower(): x for x in optimizers}.get(name.lower())
    if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
      optimizer.add_param_group({"params": g[2], "lr": lr, "weight_decay": 0.0})
    elif name == "RMSProp":
      optimizer.add_param_group({"params": g[2], "lr": lr})
    elif name == "SGD":
      optimizer.add_param_group({"params": g[2], "lr": lr, "weight_decay": 0.0})
    else:
      raise NotImplementedError(
          f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
          "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
      )

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(
      f"{colorstr('Multi-Scale Attention Pooling optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
      f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
    )

    return optimizer
