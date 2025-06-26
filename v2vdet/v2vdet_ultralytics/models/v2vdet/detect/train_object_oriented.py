import itertools
import os
import sys
import logging


from v2vdet.v2vdet_ultralytics.data import build_v2v_dataset, build_SA_V_v2v_dataset
from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image
import torchvision.transforms as transforms
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
from pathlib import Path

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models import yolo
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
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
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.data import YOLOConcatDataset, build_grounding
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.augment import LoadVisualPrompt

from v2vdet.v2vdet_ultralytics.utils import (extract_class_crops,
                                      count_trainable_parameters,
                                      crop_and_resize_largest_bbox_per_class,
                                      origin_crop_and_resize,
                                      random_crop_img)
from v2vdet.v2vdet_ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT
from v2vdet.v2vdet_ultralytics.nn.modules import (
    C2fAttn,
    ImagePoolingAttn,
    WorldDetect,
    C2f_v2v_Attn,
    TemplateAttentionPooling
)
from v2vdet.v2vdet_ultralytics.engine.trainer import BaseTrainer
from v2vdet.v2vdet_ultralytics.data import build_yolo_dataset, build_object_oriented_yolo_dataset

from v2vdet.v2vdet_ultralytics.nn import (WorldModel,
                                   v2vdetModel,
                                   v2vWorldModel,
                                   V2V_with_Patch_Attn_Pooling_Model, 
                                   V2V_Template_YOLO_Backbone_Model_Contrastive_Loss_Model,
                                   V2V_with_2_Patch_Attn_Pooling_Model,
                                   V2V_multi_scale_clip_Model,
                                   V2V_template_SigLIP_Model,
                                   V2V_template_SigLIP_multi_scale_Model,
                                   V2V_template_SigLIP_multi_scale_multi_head_Model,
                                   V2V_template_SigLIP_with_new_dataset_Model,
                                   V2V_With_MultiScale_SAVPE_Model,
                                   V2V_With_MultiScale_SAVPE_SigLIP2_B_Model,
                                   V2V_With_MultiScale_SAVPE_SigLIP2_L_Model,
                                   V2V_With_MultiScale_SAVPE_PE_B16_Model,
                                   V2V_With_MultiScale_SAVPE_PE_L14_Model)
from v2vdet.v2vdet_ultralytics.nn.tasks import (V2V_Template_YOLO_Backbone_Model,                            
                                         V2V_Template_YOLO_Backbone_Share_Param_Model,
                                         V2V_DINO_Model,
                                         V2V_DINO_with_registers_Model,
                                         V2V_template_DINO_with_registers_multi_scale_Model,
                                         V2V_template_SigLIPv2_Model,
                                         V2V_template_SigLIPv2_multi_scale_Model,
                                          V2V_template_DINO_multi_scale_Model,
                                          V2V_With_MultiScale_SAVPE_Model)

from v2vdet.v2vdet_ultralytics.models.v2vdet.val import (v2v_DetectionValidator, v2v_with_attn_pooling_DetectionValidator,
v2v_with_SAVPE_DetectionValidator                                      
)

from v2vdet.v2vdet_ultralytics.models.v2vdet.detect.train import (V2V_With_MultiScale_SAVPE_Trainer)

class V2V_With_MultiScale_SAVPE_ObjectOriented_Trainer(V2V_With_MultiScale_SAVPE_Trainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)
    
    
  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    
    from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_ObjectOriented_Model
    model = V2V_With_MultiScale_SAVPE_ObjectOriented_Model(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(False)
    model.model.requires_grad_(True)
    model.patch_emb_savpe.requires_grad_(True)

    if self.args.frozen_vision_encoder is False:
      model.vision_encoder.requires_grad_(True)
    
    self.vision_encoder_patch_size = self.model.vision_encoder_patch_size

    # self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model
    
  def build_dataset(self, img_path, mode="train", batch=None):
    """
    Build YOLO Dataset for training or validation with visual prompts.

    Args:
        img_path (List[str] | str): Path to the folder containing images or list of paths.
        mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
        batch (int, optional): Size of batches, used for rectangular training/validation.

    Returns:
        (Dataset): YOLO dataset configured for training or validation, with visual prompts for training mode.
    """    
    if mode != "train":
      gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
      dataset = build_object_oriented_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs, vision_encoder_input_size=self.vision_encoder_patch_size)
    else:
      return build_object_oriented_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", vision_encoder_input_size=self.vision_encoder_patch_size)

    # if isinstance(dataset, YOLOConcatDataset):
    #     for d in dataset.datasets:
    #         d.transforms.append(LoadVisualPrompt())
    # else:
    #     dataset.transforms.append(LoadVisualPrompt())
    return dataset
  

  def get_validator(self):
      from v2vdet.v2vdet_ultralytics.models.v2vdet.oo_val import v2v_with_SAVPE_ObjectOriented_DetectionValidator
      """Returns a DetectionValidator for YOLO model validation."""
      self.loss_names = "box_loss", "cls_loss", "dfl_loss"
      copy_args = copy(self.args)
      # copy_args.batch = 1
      
      # test_loader = self.get_dataloader(
      #     self.testset, batch_size=copy_args.batch if self.args.task == "obb" else copy_args.batch, rank=-1, mode="val"
      # )
      test_loader = self.get_dataloader(
          self.testset, batch_size= copy_args.batch if self.args.task == "obb" else copy_args.eval_batch_size, rank=-1, mode="val"
      )

      return v2v_with_SAVPE_ObjectOriented_DetectionValidator(
          test_loader, save_dir=self.save_dir, args=copy_args, _callbacks=self.callbacks
      )
  
  def preprocess_batch(self, batch):
      """Preprocesses a batch of images by scaling and converting to float."""
      for i_t in ['i', 't']:
        batch[i_t]["img"] = batch[i_t]["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch[i_t]["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch[i_t]["img"] = imgs
      return batch
  
  def _close_dataloader_mosaic(self):
      """Update dataloaders to stop using mosaic augmentation."""
      if hasattr(self.train_loader.dataset, "mosaic"):
          self.train_loader.dataset.mosaic = False
      if hasattr(self.train_loader.dataset, "close_mosaic"):
          LOGGER.info("Closing dataloader mosaic")
          self.train_loader.dataset.close_mosaic(hyp=copy(self.args))
    
  def plot_training_labels(self):
    """Create a labeled training plot of the YOLO model."""
    boxes = np.concatenate([lb['i_label']["bboxes"] for lb in self.train_loader.dataset.labels], 0)
    cls = np.concatenate([lb['i_label']["cls"] for lb in self.train_loader.dataset.labels], 0)
    plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

  def auto_batch(self):
    """Get batch size by calculating memory occupation of model."""
    train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
    # 4 for mosaic augmentation
    max_num_obj = max(len(label['i_label']["cls"]) for label in train_dataset.labels) * 4
    return super().auto_batch(max_num_obj)
  
  def plot_training_samples(self, batch, ni):
      """Plots training samples with their annotations."""
      for gg in ['i', 't']:
        if gg == 'i':
          name = self.save_dir / f"train_batch{ni}_input.jpg"
        else:
          name = self.save_dir / f"train_batch{ni}_template.jpg"
        
        plot_images(
            images=batch[gg]["img"],
            batch_idx=batch[gg]["batch_idx"],
            cls=batch[gg]["cls"].squeeze(-1),
            bboxes=batch[gg]["bboxes"],
            paths=batch[gg]["im_file"],
            fname=name,
            on_plot=self.on_plot,
        )
  
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
      self._clear_memory()
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
          loss, self.loss_items = self.model(batch)
          self.loss = loss.sum()
          if RANK != -1:
            self.loss *= world_size
          self.tloss = (
              (self.tloss * i + self.loss_items) /
              (i + 1) if self.tloss is not None else self.loss_items
          )

        # Backward
        self.scaler.scale(self.loss).backward()
        # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html

        self.accumulate = self.accumulate if self.args.gradient_accumulation_steps == -1 else self.args.gradient_accumulation_steps
        if ni - last_opt_step >= self.accumulate or ni == nb:
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
                  batch['i']["cls"].shape[0],  # batch size, i.e. 8
                  batch['i']["img"].shape[-1],  # imgsz, i.e 640
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

        self.run_callbacks("on_train_batch_end")
        
      self.lr = {f"lr/pg{ir}": x["lr"]
                 for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

      self.run_callbacks("on_train_epoch_end")
      
      self.model.eval()
      
      if RANK in {-1, 0}:
        final_epoch = epoch + 1 >= self.epochs
        self.ema.update_attr(self.model, include=[
                             "yaml", "nc", "args", "names", "stride", "class_weights"])
        
        # Validation
        if (self.args.val and ( epoch%self.args.eval_period==0) ) or final_epoch or self.stopper.possible_stop or self.stop:
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
          # self.ema.ema = self.ema.ema.cpu()
          # self.model = self.model.cpu()

          self.save_model()
          self.run_callbacks("on_model_save")

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
    
class V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented_Trainer(V2V_With_MultiScale_SAVPE_ObjectOriented_Trainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)
    
  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented_Model
    model = V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented_Model(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(False)
    model.model.requires_grad_(True)
    model.patch_emb_savpe.requires_grad_(True)

    if self.args.frozen_vision_encoder is False:
      model.vision_encoder.requires_grad_(True)
    
    self.vision_encoder_patch_size = model.vision_encoder_patch_size

    # self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model
  
class V2V_With_MultiScale_SAVPE_SigLIP2_L_ObjectOriented_Trainer(V2V_With_MultiScale_SAVPE_ObjectOriented_Trainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)
    
  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_SigLIP2_L_ObjectOriented_Model
    model = V2V_With_MultiScale_SAVPE_SigLIP2_L_ObjectOriented_Model(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(False)
    model.model.requires_grad_(True)
    model.patch_emb_savpe.requires_grad_(True)

    if self.args.frozen_vision_encoder is False:
      model.vision_encoder.requires_grad_(True)
    
    self.vision_encoder_patch_size = model.vision_encoder_patch_size
    # self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model
  
class V2V_With_MultiScale_SAVPE_DINO2_B_ObjectOriented_Trainer(V2V_With_MultiScale_SAVPE_ObjectOriented_Trainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)
    
  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    from v2vdet.v2vdet_ultralytics.nn.tasks_oo import  V2V_With_MultiScale_SAVPE_DINOv2_B_ObjectOriented_Model as MODEL
    model = MODEL(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(False)
    model.model.requires_grad_(True)
    model.patch_emb_savpe.requires_grad_(True)

    if self.args.frozen_vision_encoder is False:
      model.vision_encoder.requires_grad_(True)

    # self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    self.vision_encoder_patch_size = model.vision_encoder_patch_size
    return model
  
class V2V_With_MultiScale_SAVPE_DINO2_L_ObjectOriented_Trainer(V2V_With_MultiScale_SAVPE_ObjectOriented_Trainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)
    
  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    from v2vdet.v2vdet_ultralytics.nn.tasks_oo import  V2V_With_MultiScale_SAVPE_DINOv2_L_ObjectOriented_Model as MODEL
    model = MODEL(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(False)
    model.model.requires_grad_(True)
    model.patch_emb_savpe.requires_grad_(True)

    if self.args.frozen_vision_encoder is False:
      model.vision_encoder.requires_grad_(True)

    # self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)
    self.vision_encoder_patch_size = model.vision_encoder_patch_size
    return model
  
class V2V_With_MultiScale_SAVPE_PE_B16_ObjectOriented_Trainer(V2V_With_MultiScale_SAVPE_ObjectOriented_Trainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)
    
  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_PE_B16_ObjectOriented_Model as MODEL
    model = MODEL(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(False)
    model.model.requires_grad_(True)
    model.patch_emb_savpe.requires_grad_(True)

    if self.args.frozen_vision_encoder is False:
      model.vision_encoder.requires_grad_(True)
    
    self.vision_encoder_patch_size = model.vision_encoder_patch_size
    # self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model

class V2V_With_MultiScale_SAVPE_PE_L14_ObjectOriented_Trainer(V2V_With_MultiScale_SAVPE_PE_B16_ObjectOriented_Trainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)
    
  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_PE_L14_ObjectOriented_Model as MODEL
    model = MODEL(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(False)
    model.model.requires_grad_(True)
    model.patch_emb_savpe.requires_grad_(True)

    if self.args.frozen_vision_encoder is False:
      model.vision_encoder.requires_grad_(True)

    # self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)
    self.vision_encoder_patch_size = model.vision_encoder_patch_size
    return model

class V2V_With_MultiScale_SAVPE_YOLOE_ObjectOriented_Trainer(V2V_With_MultiScale_SAVPE_ObjectOriented_Trainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)
    
  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_YOLOE_ObjectOriented_Model as MODEL
    model = MODEL(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      gg = torch.load(weights)
      model.load(weights=gg['model'])

    model.requires_grad_(False)
    model.model.requires_grad_(True)
    model.patch_emb_savpe.requires_grad_(True)
    # self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model
  