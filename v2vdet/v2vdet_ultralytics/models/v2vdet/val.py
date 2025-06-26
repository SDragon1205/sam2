import json
import time
from pathlib import Path

import numpy as np
import torch
from copy import deepcopy
import torchvision.transforms as transforms
from multiprocessing import Pool
import random
from PIL import Image

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data import build_dataloader
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.data import YOLOConcatDataset
from ultralytics.data.augment import LoadVisualPrompt

from v2vdet.v2vdet_ultralytics.models.v2vdet.detect.val import DetectionValidator
from v2vdet.v2vdet_ultralytics.nn import (v2vdet_AutoBackend, 
                                   v2vdet_template_feats_AutoBackend,
                                   v2v_with_SAVPE_AutoBackend)
from v2vdet.v2vdet_ultralytics.utils import (prepare_v2v_crop_image,
                                      crop_and_resize,
                                      origin_crop_and_resize,
                                      optimized_apply_augmentation,
                                      random_crop_pil_image)
from v2vdet.v2vdet_ultralytics.data.build import build_v2v_dataset, build_SA_V_v2v_dataset, build_dataloader

class v2v_DetectionValidator(DetectionValidator):

  def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)
  
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
    # aug_params =  input_dict['aug_params'] if not None else {
    #         'rotation_range': (-30, 30),
    #         'scale_range': (0.5, 2),
    #         'brightness_range': (0.8, 1.2),
    #         'contrast_range': (0.8, 1.2),
    #         'prob': 0.5,
    #         'global_prob': 0.7
    #     }
    
    batch_size = len(batch['img'])

    single_sample_crop_img = []
    for _ in range (num_classes):
      random_sel_image = batch['img'][random.randint(0, batch_size-1)]
      if isinstance(random_sel_image, torch.Tensor):
        # If image is smaller than crop size
        if (random_sel_image.shape[1] < crop_size[0]) or (random_sel_image.shape[2] < crop_size[1]):
          C, H, W = random_sel_image.shape
          target_H, target_W = crop_size
          
          # Create black background 
          black_background = torch.zeros((C, target_H, target_W), dtype=random_sel_image.dtype, device=random_sel_image.device)
          
          # Calculate the paste location
          start_h = max(0, (target_H - H) // 2)
          start_w = max(0, (target_W - W) // 2)
          
          paste_h = min(H, target_H)
          paste_w = min(W, target_W)

          black_background[:, start_h:start_h + paste_h, start_w:start_w + paste_w] = random_sel_image[:, :paste_h, :paste_w]

          single_sample_crop_img.append(black_background)
        
        else:
          single_sample_crop_img.append(random_crop(random_sel_image))
      
      else:
        raise ValueError("Input 'random_sel_image' is not in torch.Tensor format. Image type is not supported")
    
    # single_sample_crop_img = [random_crop(
    # batch['img'][random.randint(0, batch_size-1)]) for _ in range(num_classes)]

    single_sample_crop_img = [torch.zeros((3, *crop_size)) for _ in range(num_classes)]

    matches = (batch['batch_idx'] == img_idx).nonzero()

    if len(matches) > 0:
      batch_start = (batch['batch_idx'] == img_idx).nonzero()[0].item()
      batch_count = (batch['batch_idx'] == img_idx).sum().item()
      img_classes = batch['cls'][batch_start:batch_start + batch_count]
      img_boxes = batch['bboxes'][batch_start:batch_start + batch_count]

      cropped_positives = origin_crop_and_resize(
            batch['img'][img_idx], img_boxes, classes=img_classes, 
            size=crop_size, 
            augment=False
      )

      for crop_data in cropped_positives:
        single_sample_crop_img[int(crop_data['cls'])] = crop_data['crop_tensor_img']
    
    return torch.stack(single_sample_crop_img, dim=0)
  
  def preprocess(self, batch, crop_size=(224, 224)):
    """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""

    batch = super().preprocess(batch)
    # NOTE: add template matching features

    num_classes = self.nc  # Usually 80
    batch_size = len(batch['img'])

    # Prepare storage for batch embeddings
    crop_img_list = []
    random_crop = transforms.RandomCrop(size=crop_size)
    multi_processing_batch = deepcopy(batch)
    multi_processing_batch['img'] = [img.to('cpu') for img in multi_processing_batch['img']]
    multi_processing_batch['cls'] = multi_processing_batch['cls'].to('cpu')
    multi_processing_batch['bboxes'] = multi_processing_batch['bboxes'].to('cpu')
    multi_processing_batch['batch_idx'] = multi_processing_batch['batch_idx'].to('cpu')
    
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
    # 
    # with Pool(processes=self.dataloader.num_workers*2) as pool:

    # if want multi processing
    # crop_img_list = self.multiprocessing_pool.map(self._multiprocessing_preprocess_batch, multiprocessing_input_list)
    # no want multi preprocessing
    for input_list in multiprocessing_input_list:
      crop_img_list.append(self._multiprocessing_preprocess_batch(input_list))

    batch["template_feats"] = torch.stack(tensors=crop_img_list).flatten(0, 1)
    return batch

  # def build_dataset(self, img_path, mode="val", batch=None):
  #   """
  #   Build YOLO Dataset.

  #   Args:
  #       img_path (str): Path to the folder containing images.
  #       mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
  #       batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
  #   """
  #   return build_v2v_dataset(self.args,
  #                                 img_path,
  #                                 batch,
  #                                 self.data,
  #                                 mode=mode,
  #                                 stride=self.stride)

  @smart_inference_mode()
  def __call__(self, trainer=None, model=None):
    """Executes validation process, running inference on dataloader and computing performance metrics."""
    self.training = trainer is not None
    augment = self.args.augment and (not self.training)
    if self.training:
      self.device = trainer.device
      self.data = trainer.data
      # force FP16 val during training
      self.args.half = self.device.type != "cpu" and trainer.amp
      model = trainer.ema.ema or trainer.model
      model = model.half() if self.args.half else model.float()
      # self.model = model
      self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
      self.args.plots &= trainer.stopper.possible_stop or (
          trainer.epoch == trainer.epochs - 1)
      model.eval()
    else:
      if str(self.args.model).endswith(".yaml"):
        LOGGER.warning(
            "WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
      callbacks.add_integration_callbacks(self)

      model = v2vdet_AutoBackend(
          weights=model or self.args.model,
          device=select_device(self.args.device, self.args.batch),
          dnn=self.args.dnn,
          data=self.args.data,
          fp16=self.args.half,
      )

      model.requires_grad_(False)
      self.device = model.device  # update device
      self.args.half = model.fp16  # update half
      stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
      imgsz = check_imgsz(self.args.imgsz, stride=stride)

      if engine:
        self.args.batch = model.batch_size
      elif not pt and not jit:
        # export.py models default to batch-size 1
        self.args.batch = model.metadata.get("batch", 1)
        LOGGER.info(
            f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

      if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
        self.data = check_det_dataset(self.args.data)
      elif self.args.task == "classify":
        self.data = check_cls_dataset(self.args.data, split=self.args.split)
      elif self.args.task == "detect":
        self.data = check_det_dataset(self.args.data['val']['yolo_data'][0])
      else:
        raise FileNotFoundError(
            emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

      if self.device.type in {"cpu", "mps"}:
        self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
      if not pt:
        self.args.rect = False
      self.stride = model.stride  # used in get_dataloader() for padding
      # self.dataloader = self.dataloader or self.get_dataloader(
      #     self.data.get(self.args.split), self.args.batch*2)
      self.dataloader = self.dataloader or self.get_dataloader(
          self.data.get(self.args.split), self.args.batch*2)

      model.eval()
      model.warmup(imgsz=(1 if pt else self.args.batch*2,
                   3, imgsz, imgsz))  # warmup

    self.run_callbacks("on_val_start")
    dt = (
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
    )
    bar = TQDM(self.dataloader, desc=self.get_desc(),
               total=len(self.dataloader))
    self.init_metrics(de_parallel(model))
    self.jdict = []  # empty before each val

    # self.multiprocessing_pool = Pool(processes=max(1, self.dataloader.num_workers//2))
    with torch.autocast(device_type="cuda", dtype=torch.float16):
      for batch_i, batch in enumerate(bar):
        self.run_callbacks("on_val_batch_start")
        self.batch_i = batch_i
        # Preprocess
        with dt[0]:
          batch = self.preprocess(batch=batch)

        # Inference
        with dt[1]:
          preds = model(batch["img"], template_feats=batch["template_feats"])

        # Loss
        with dt[2]:
          if self.training:
            self.loss += model.loss(batch, preds)[1]

        # Postprocess
        with dt[3]:
          preds = self.postprocess(preds)

        self.update_metrics(preds, batch)
        if self.args.plots and batch_i < 5:
          self.plot_val_samples(batch, batch_i)
          self.plot_predictions(batch, preds, batch_i)

        self.run_callbacks("on_val_batch_end")
    
    # self.multiprocessing_pool.close()
    # self.multiprocessing_pool.join()

    stats = self.get_stats()
    self.check_stats(stats)
    self.speed = dict(
        zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
    self.finalize_metrics()
    self.print_results()

    self.run_callbacks("on_val_end")
    if self.training:
      model.float()
      results = {
          **stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
      # return results as 5 decimal place floats
      return {k: round(float(v), 5) for k, v in results.items()}
    else:
      LOGGER.info(
          "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
              *tuple(self.speed.values())
          )
      )
      if self.args.save_json and self.jdict:
        with open(str(self.save_dir / "predictions.json"), "w") as f:
          LOGGER.info(f"Saving {f.name}...")
          json.dump(self.jdict, f)  # flatten and save
        stats = self.eval_json(stats)  # update stats
      if self.args.plots or self.args.save_json:
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
      return stats

  def get_dataloader(self, dataset_path, batch_size):
    """Construct and return dataloader."""
    dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
    return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader
  
class v2v_with_SAVPE_DetectionValidator(DetectionValidator):
  def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)

  def preprocess(self, batch):
    """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""

    batch = super().preprocess(batch)

    return batch

  @smart_inference_mode()
  def __call__(self, trainer=None, model=None):
    """Executes validation process, running inference on dataloader and computing performance metrics."""
    self.training = trainer is not None
    augment = self.args.augment and (not self.training)
    if self.training:
      self.device = trainer.device
      self.data = trainer.data
      # force FP16 val during training
      self.args.half = self.device.type != "cpu" and trainer.amp
      model = trainer.ema.ema or trainer.model
      model = model.half() if self.args.half else model.float()
      # self.model = model
      self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
      self.args.plots &= trainer.stopper.possible_stop or (
          trainer.epoch == trainer.epochs - 1)
      model.eval()
    else:
      if str(self.args.model).endswith(".yaml"):
        LOGGER.warning(
            "WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
      callbacks.add_integration_callbacks(self)

      model = v2v_with_SAVPE_AutoBackend(
          weights=model or self.args.model,
          device=select_device(self.args.device, self.args.batch),
          dnn=self.args.dnn,
          data=self.args.data,
          fp16=self.args.half,
      )

      model.requires_grad_(False)
      self.device = model.device  # update device
      self.args.half = model.fp16  # update half
      stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
      imgsz = check_imgsz(self.args.imgsz, stride=stride)

      if engine:
        self.args.batch = model.batch_size
      elif not pt and not jit:
        # export.py models default to batch-size 1
        self.args.batch = model.metadata.get("batch", 1)
        LOGGER.info(
            f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

      if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
        self.data = check_det_dataset(self.args.data)
      elif self.args.task == "classify":
        self.data = check_cls_dataset(self.args.data, split=self.args.split)
      elif self.args.task == "detect":
        self.data = check_det_dataset(self.args.data['val']['yolo_data'][0])
      else:
        raise FileNotFoundError(
            emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

      if self.device.type in {"cpu", "mps"}:
        self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
      if not pt:
        self.args.rect = False
      self.stride = model.stride  # used in get_dataloader() for padding
      # self.dataloader = self.dataloader or self.get_dataloader(
      #     self.data.get(self.args.split), self.args.batch*2)
      self.dataloader = self.dataloader or self.get_dataloader(
          self.data.get(self.args.split), self.args.batch*2)

      model.eval()
      model.warmup(imgsz=(1 if pt else self.args.batch*2,
                   3, imgsz, imgsz))  # warmup

    self.run_callbacks("on_val_start")
    dt = (
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
    )
    bar = TQDM(self.dataloader, desc=self.get_desc(),
               total=len(self.dataloader))
    self.init_metrics(de_parallel(model))
    self.jdict = []  # empty before each val

    # self.multiprocessing_pool = Pool(processes=max(1, self.dataloader.num_workers//2))
    
    with torch.autocast(device_type="cuda", dtype=torch.float16):
      for batch_i, batch in enumerate(bar):
        self.run_callbacks("on_val_batch_start")
        self.batch_i = batch_i
        # Preprocess
        with dt[0]:
          batch = self.preprocess(batch=batch)

        # Inference
        with dt[1]:
          batch['nc'] = self.nc
          preds = model(batch['img'], batch)

        # Loss
        with dt[2]:
          if self.training:
            self.loss += model.loss(batch, preds)[1]

        # Postprocess
        with dt[3]:
          preds = self.postprocess(preds)

        self.update_metrics(preds, batch)
        if self.args.plots and batch_i < 5:
          self.plot_val_samples(batch, batch_i)
          self.plot_predictions(batch, preds, batch_i)

        self.run_callbacks("on_val_batch_end")
    
    # self.multiprocessing_pool.close()
    # self.multiprocessing_pool.join()

    stats = self.get_stats()
    self.check_stats(stats)
    self.speed = dict(
        zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
    self.finalize_metrics()
    self.print_results()

    self.run_callbacks("on_val_end")
    if self.training:
      model.float()
      results = {
          **stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
      # return results as 5 decimal place floats
      return {k: round(float(v), 5) for k, v in results.items()}
    else:
      LOGGER.info(
          "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
              *tuple(self.speed.values())
          )
      )
      if self.args.save_json and self.jdict:
        with open(str(self.save_dir / "predictions.json"), "w") as f:
          LOGGER.info(f"Saving {f.name}...")
          json.dump(self.jdict, f)  # flatten and save
        stats = self.eval_json(stats)  # update stats
      if self.args.plots or self.args.save_json:
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
      return stats

  def build_dataset(self, img_path, mode="val", batch=None):
    """
    Build YOLO Dataset for training or validation with visual prompts.

    Args:
        img_path (List[str] | str): Path to the folder containing images or list of paths.
        mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
        batch (int, optional): Size of batches, used for rectangular training/validation.

    Returns:
        (Dataset): YOLO dataset configured for training or validation, with visual prompts for training mode.
    """
    self.args.rect = False
    dataset = super().build_dataset(img_path, mode, batch)
    return dataset
  
  def get_dataloader(self, dataset_path, batch_size):
    """Construct and return dataloader."""
    dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
    return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

class v2v_new_DetectionValidator(DetectionValidator):
  def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)

  def build_dataset(self, img_path, mode="val", batch=None, rect=False, stride=32):
      """
      Build YOLO Dataset.

      Args:
          img_path (str): Path to the folder containing images.
          mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
          batch (int, optional): Size of batches, this is for `rect`.

      Returns:
          (Dataset): YOLO dataset.
      """
      cfg = self.args
      
      from v2vdet.v2vdet_ultralytics.data.dataset import V2V_Dataset
      return V2V_Dataset(
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
          data=self.data,
          fraction=cfg.fraction if mode == "train" else 1.0,
      )

  @smart_inference_mode()
  def __call__(self, trainer=None, model=None):
    """Executes validation process, running inference on dataloader and computing performance metrics."""
    self.training = trainer is not None
    augment = self.args.augment and (not self.training)
    if self.training:
      self.device = trainer.device
      self.data = trainer.data
      # force FP16 val during training
      self.args.half = self.device.type != "cpu" and trainer.amp
      model = trainer.ema.ema or trainer.model
      model = model.half() if self.args.half else model.float()
      # self.model = model
      self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
      self.args.plots &= trainer.stopper.possible_stop or (
          trainer.epoch == trainer.epochs - 1)
      model.eval()
    else:
      if str(self.args.model).endswith(".yaml"):
        LOGGER.warning(
            "WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
      callbacks.add_integration_callbacks(self)

      model = v2vdet_AutoBackend(
          weights=model or self.args.model,
          device=select_device(self.args.device, self.args.batch),
          dnn=self.args.dnn,
          data=self.args.data,
          fp16=self.args.half,
      )

      model.requires_grad_(False)
      self.device = model.device  # update device
      self.args.half = model.fp16  # update half
      stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
      imgsz = check_imgsz(self.args.imgsz, stride=stride)

      if engine:
        self.args.batch = model.batch_size
      elif not pt and not jit:
        # export.py models default to batch-size 1
        self.args.batch = model.metadata.get("batch", 1)
        LOGGER.info(
            f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

      if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
        self.data = check_det_dataset(self.args.data)
      elif self.args.task == "classify":
        self.data = check_cls_dataset(self.args.data, split=self.args.split)
      elif self.args.task == "detect":
        self.data = check_det_dataset(self.args.data['val']['yolo_data'][0])
      else:
        raise FileNotFoundError(
            emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

      if self.device.type in {"cpu", "mps"}:
        self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
      if not pt:
        self.args.rect = False
      self.stride = model.stride  # used in get_dataloader() for padding
      # self.dataloader = self.dataloader or self.get_dataloader(
      #     self.data.get(self.args.split), self.args.batch*2)
      self.dataloader = self.dataloader or self.get_dataloader(
          self.data.get(self.args.split), self.args.batch*2)

      model.eval()
      model.warmup(imgsz=(1 if pt else self.args.batch*2,
                   3, imgsz, imgsz))  # warmup

    self.run_callbacks("on_val_start")
    dt = (
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
    )
    bar = TQDM(self.dataloader, desc=self.get_desc(),
               total=len(self.dataloader))
    self.init_metrics(de_parallel(model))
    self.jdict = []  # empty before each val

    # self.multiprocessing_pool = Pool(processes=max(1, self.dataloader.num_workers//2))
    with torch.autocast(device_type="cuda", dtype=torch.float16):
      for batch_i, batch in enumerate(bar):
        self.run_callbacks("on_val_batch_start")
        self.batch_i = batch_i
        # Preprocess
        with dt[0]:
          batch = self.preprocess(batch=batch)

        # Inference
        with dt[1]:
          preds = model(batch["img"], template_feats=batch["template_feats"])

        # Loss
        with dt[2]:
          if self.training:
            self.loss += model.loss(batch, preds)[1]

        # Postprocess
        with dt[3]:
          preds = self.postprocess(preds)

        self.update_metrics(preds, batch)
        if self.args.plots and batch_i < 5:
          self.plot_val_samples(batch, batch_i)
          self.plot_predictions(batch, preds, batch_i)

        self.run_callbacks("on_val_batch_end")
    
    # self.multiprocessing_pool.close()
    # self.multiprocessing_pool.join()

    stats = self.get_stats()
    self.check_stats(stats)
    self.speed = dict(
        zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
    self.finalize_metrics()
    self.print_results()

    self.run_callbacks("on_val_end")
    if self.training:
      model.float()
      results = {
          **stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
      # return results as 5 decimal place floats
      return {k: round(float(v), 5) for k, v in results.items()}
    else:
      LOGGER.info(
          "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
              *tuple(self.speed.values())
          )
      )
      if self.args.save_json and self.jdict:
        with open(str(self.save_dir / "predictions.json"), "w") as f:
          LOGGER.info(f"Saving {f.name}...")
          json.dump(self.jdict, f)  # flatten and save
        stats = self.eval_json(stats)  # update stats
      if self.args.plots or self.args.save_json:
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
      return stats

class v2v_with_attn_pooling_DetectionValidator(DetectionValidator):

  def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)

  @smart_inference_mode()
  def __call__(self, trainer=None, model=None):
    """Executes validation process, running inference on dataloader and computing performance metrics."""
    self.training = trainer is not None
    augment = self.args.augment and (not self.training)
    if self.training:
      self.device = trainer.device
      self.data = trainer.data
      # force FP16 val during training
      self.args.half = self.device.type != "cpu" and trainer.amp
      model = trainer.ema.ema or trainer.model
      model = model.half() if self.args.half else model.float()
      # self.model = model
      self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
      self.args.plots &= trainer.stopper.possible_stop or (
          trainer.epoch == trainer.epochs - 1)
      model.eval()
    else:
      if str(self.args.model).endswith(".yaml"):
        LOGGER.warning(
            "WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
      callbacks.add_integration_callbacks(self)

      model.training = False
      for param in model.parameters():
        param.requires_grad = False

      model = v2vdet_AutoBackend(
          weights=model or self.args.model,
          device=select_device(self.args.device, self.args.batch),
          dnn=self.args.dnn,
          data=self.args.data,
          fp16=self.args.half,
      )
      # self.model = model
      self.device = model.device  # update device
      self.args.half = model.fp16  # update half
      stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
      imgsz = check_imgsz(self.args.imgsz, stride=stride)

      if engine:
        self.args.batch = model.batch_size
      elif not pt and not jit:
        # export.py models default to batch-size 1
        self.args.batch = model.metadata.get("batch", 1)
        LOGGER.info(
            f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

      if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
        self.data = check_det_dataset(self.args.data)
      elif self.args.task == "classify":
        self.data = check_cls_dataset(self.args.data, split=self.args.split)
      else:
        raise FileNotFoundError(
            emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

      if self.device.type in {"cpu", "mps"}:
        self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
      if not pt:
        self.args.rect = False
      self.stride = model.stride  # used in get_dataloader() for padding
      self.dataloader = self.dataloader or self.get_dataloader(
          self.data.get(self.args.split), self.args.batch)

      model.eval()
      model.warmup(imgsz=(1 if pt else self.args.batch,
                   3, imgsz, imgsz))  # warmup

    self.run_callbacks("on_val_start")
    dt = (
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
    )
    bar = TQDM(self.dataloader, desc=self.get_desc(),
               total=len(self.dataloader))
    self.init_metrics(de_parallel(model))
    self.jdict = []  # empty before each val

    for batch_i, batch in enumerate(bar):

      self.run_callbacks("on_val_batch_start")
      self.batch_i = batch_i
      # Preprocess
      with dt[0]:
        batch = self.preprocess(batch)

      # Inference
      with dt[1]:
        # Figure out of class, set to 80 or 1203
        crop_img_list_tensor = prepare_v2v_crop_image(batch, self.nc)

        # crop_img_list_tensor = crop_img_list_tensor.reshape(-1, self.nc, crop_img_list_tensor.shape[1], crop_img_list_tensor.shape[2], crop_img_list_tensor.shape[3])
        batch["img"] = batch["img"].to(self.device)
        preds = model.module(
            batch["img"], crop_img_list=crop_img_list_tensor, augment=augment)

      # Loss
      with dt[2]:
        if self.training:
          self.loss += model.loss(batch, preds)[1]

      # Postprocess
      with dt[3]:
        preds = self.postprocess(preds)

      self.update_metrics(preds, batch)
      if self.args.plots and batch_i < 3:
        self.plot_val_samples(batch, batch_i)
        self.plot_predictions(batch, preds, batch_i)

      self.run_callbacks("on_val_batch_end")

    stats = self.get_stats()
    self.check_stats(stats)
    self.speed = dict(
        zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
    self.finalize_metrics()
    self.print_results()

    self.run_callbacks("on_val_end")
    if self.training:
      model.float()
      results = {
          **stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
      # return results as 5 decimal place floats
      return {k: round(float(v), 5) for k, v in results.items()}
    else:
      LOGGER.info(
          "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
              *tuple(self.speed.values())
          )
      )
      if self.args.save_json and self.jdict:
        with open(str(self.save_dir / "predictions.json"), "w") as f:
          LOGGER.info(f"Saving {f.name}...")
          json.dump(self.jdict, f)  # flatten and save
        stats = self.eval_json(stats)  # update stats
      if self.args.plots or self.args.save_json:
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
      return stats

class v2v_with_2_patch_attn_pooling_DetectionValidator(DetectionValidator):

  def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)

  def preprocess(self, batch):
    """Preprocesses batch of images for YOLO training."""
    super().preprocess(batch)
    return batch

  @smart_inference_mode()
  def __call__(self, trainer=None, model=None):
    """Executes validation process, running inference on dataloader and computing performance metrics."""
    self.training = trainer is not None
    augment = self.args.augment and (not self.training)
    if self.training:
      self.device = trainer.device
      self.data = trainer.data
      # force FP16 val during training
      self.args.half = self.device.type != "cpu" and trainer.amp
      model = trainer.ema.ema or trainer.model
      model = model.half() if self.args.half else model.float()
      # self.model = model
      self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
      self.args.plots &= trainer.stopper.possible_stop or (
          trainer.epoch == trainer.epochs - 1)
      model.eval()
    else:
      if str(self.args.model).endswith(".yaml"):
        LOGGER.warning(
            "WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
      callbacks.add_integration_callbacks(self)

      model.training = False
      for param in model.parameters():
        param.requires_grad = False

      model = v2vdet_AutoBackend(
          weights=model or self.args.model,
          device=select_device(self.args.device, self.args.batch),
          dnn=self.args.dnn,
          data=self.args.data,
          fp16=self.args.half,
      )
      # self.model = model
      self.device = model.device  # update device
      self.args.half = model.fp16  # update half
      stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
      imgsz = check_imgsz(self.args.imgsz, stride=stride)

      if engine:
        self.args.batch = model.batch_size
      elif not pt and not jit:
        # export.py models default to batch-size 1
        self.args.batch = model.metadata.get("batch", 1)
        LOGGER.info(
            f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

      if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
        self.data = check_det_dataset(self.args.data)
      elif self.args.task == "classify":
        self.data = check_cls_dataset(self.args.data, split=self.args.split)
      else:
        raise FileNotFoundError(
            emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

      if self.device.type in {"cpu", "mps"}:
        self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
      if not pt:
        self.args.rect = False
      self.stride = model.stride  # used in get_dataloader() for padding
      self.dataloader = self.dataloader or self.get_dataloader(
          self.data.get(self.args.split), self.args.batch)

      model.eval()
      model.warmup(imgsz=(1 if pt else self.args.batch,
                   3, imgsz, imgsz))  # warmup

    self.run_callbacks("on_val_start")
    dt = (
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
    )
    bar = TQDM(self.dataloader, desc=self.get_desc(),
               total=len(self.dataloader))
    self.init_metrics(de_parallel(model))
    self.jdict = []  # empty before each val

    for batch_i, batch in enumerate(bar):

      self.run_callbacks("on_val_batch_start")
      self.batch_i = batch_i
      # Preprocess
      with dt[0]:
        batch = self.preprocess(batch)
        crop_img_list_tensor = prepare_v2v_crop_image(batch, self.nc, device='cpu')

      # Inference
      with dt[1]:
        preds = model(batch["img"], crop_img_list = crop_img_list_tensor, augment=augment)

      # Loss
      with dt[2]:
        if self.training:
          self.loss += model.loss(batch, preds)[1]

      # Postprocess
      with dt[3]:
        preds = self.postprocess(preds)

      self.update_metrics(preds, batch)
      if self.args.plots and batch_i < 3:
        self.plot_val_samples(batch, batch_i)
        self.plot_predictions(batch, preds, batch_i)

      self.run_callbacks("on_val_batch_end")

    stats = self.get_stats()
    self.check_stats(stats)
    self.speed = dict(
        zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
    self.finalize_metrics()
    self.print_results()

    self.run_callbacks("on_val_end")
    if self.training:
      model.float()
      results = {
          **stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
      # return results as 5 decimal place floats
      return {k: round(float(v), 5) for k, v in results.items()}
    else:
      LOGGER.info(
          "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
              *tuple(self.speed.values())
          )
      )
      if self.args.save_json and self.jdict:
        with open(str(self.save_dir / "predictions.json"), "w") as f:
          LOGGER.info(f"Saving {f.name}...")
          json.dump(self.jdict, f)  # flatten and save
        stats = self.eval_json(stats)  # update stats
      if self.args.plots or self.args.save_json:
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
      return stats

class v2v_template_feats_DetectionValidator(DetectionValidator):

  def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)

  def preprocess(self, batch):
    """Preprocesses batch of images for YOLO training."""
    batch["img"] = batch["img"].to(self.device, non_blocking=True)
    batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
    batch["template_feats"] = batch["template_feats"].to(self.device, non_blocking=True)
    batch["template_feats"] = (batch["template_feats"].half() if self.args.half else batch["template_feats"].float()) / 255
    for k in ["batch_idx", "cls", "bboxes"]:
        batch[k] = batch[k].to(self.device)

    if self.args.save_hybrid:
        height, width = batch["img"].shape[2:]
        nb = len(batch["img"])
        bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
        self.lb = [
            torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
            for i in range(nb)
        ]

    return batch

  def build_dataset(self, img_path, mode="val", batch=None):
    """
    Build YOLO Dataset.

    Args:
        img_path (str): Path to the folder containing images.
        mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
        batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
    """
    return build_SA_V_v2v_dataset(self.args,
                                  img_path, batch,
                                  self.data,
                                  mode=mode,
                                  stride=self.stride)


  def get_dataloader(self, dataset_path, batch_size):
    """Construct and return dataloader."""
    dataset = self.build_dataset(dataset_path,
                                 batch=batch_size,
                                 mode="val")
    return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

  @smart_inference_mode()
  def __call__(self, trainer=None, model=None):
    """Executes validation process, running inference on dataloader and computing performance metrics."""
    self.training = trainer is not None
    augment = self.args.augment and (not self.training)
    if self.training:
      self.device = trainer.device
      self.data = trainer.data
      # force FP16 val during training
      self.args.half = self.device.type != "cpu" and trainer.amp
      model = trainer.ema.ema or trainer.model
      model = model.half() if self.args.half else model.float()
      # self.model = model
      self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
      self.args.plots &= trainer.stopper.possible_stop or (
          trainer.epoch == trainer.epochs - 1)
      model.eval()
    else:
      if str(self.args.model).endswith(".yaml"):
        LOGGER.warning(
            "WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
      callbacks.add_integration_callbacks(self)

      model = v2vdet_template_feats_AutoBackend(
          weights=model or self.args.model,
          device=select_device(self.args.device, self.args.batch),
          dnn=self.args.dnn,
          data=self.args.data,
          fp16=self.args.half,
      )

      model.requires_grad_(False)
      self.device = model.device  # update device
      self.args.half = model.fp16  # update half
      stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
      imgsz = check_imgsz(self.args.imgsz, stride=stride)

      if engine:
        self.args.batch = model.batch_size
      elif not pt and not jit:
        # export.py models default to batch-size 1
        self.args.batch = model.metadata.get("batch", 1)
        LOGGER.info(
            f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

      if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
        self.data = check_det_dataset(self.args.data)
      elif self.args.task == "classify":
        self.data = check_cls_dataset(self.args.data, split=self.args.split)
      elif self.args.task == "detect":
        self.data = check_det_dataset(self.args.data['val']['yolo_data'][0])
      else:
        raise FileNotFoundError(
            emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

      if self.device.type in {"cpu", "mps"}:
        self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
      if not pt:
        self.args.rect = False
      self.stride = model.stride  # used in get_dataloader() for padding
      self.dataloader = self.dataloader or self.get_dataloader(
          self.data.get(self.args.split), self.args.batch)

      model.eval()
      model.warmup(imgsz=(1 if pt else self.args.batch,
                   3, imgsz, imgsz))  # warmup

    self.run_callbacks("on_val_start")
    dt = (
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
    )
    bar = TQDM(self.dataloader, desc=self.get_desc(),
               total=len(self.dataloader))
    self.init_metrics(de_parallel(model))
    self.jdict = []  # empty before each val

    with torch.autocast(device_type="cuda", dtype=torch.float16):
      for batch_i, batch in enumerate(bar):
        self.run_callbacks("on_val_batch_start")
        self.batch_i = batch_i
        # Preprocess
        with dt[0]:
          batch = self.preprocess(batch)

        # Inference
        with dt[1]:
          preds = model(batch["img"], template_feats = batch["template_feats"])

        # Loss
        with dt[2]:
          if self.training:
            self.loss += model.loss(batch, preds)[1]

        # Postprocess
        with dt[3]:
          preds = self.postprocess(preds)

        self.update_metrics(preds, batch)
        if self.args.plots and batch_i < 3:
          self.plot_val_samples(batch, batch_i)
          self.plot_predictions(batch, preds, batch_i)

        self.run_callbacks("on_val_batch_end")

    stats = self.get_stats()
    self.check_stats(stats)
    self.speed = dict(
        zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
    self.finalize_metrics()
    self.print_results()

    self.run_callbacks("on_val_end")
    if self.training:
      model.float()
      results = {
          **stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
      # return results as 5 decimal place floats
      return {k: round(float(v), 5) for k, v in results.items()}
    else:
      LOGGER.info(
          "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
              *tuple(self.speed.values())
          )
      )
      if self.args.save_json and self.jdict:
        with open(str(self.save_dir / "predictions.json"), "w") as f:
          LOGGER.info(f"Saving {f.name}...")
          json.dump(self.jdict, f)  # flatten and save
        stats = self.eval_json(stats)  # update stats
      if self.args.plots or self.args.save_json:
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
      return stats
