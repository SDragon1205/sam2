# from v2vdet.v2vdet_ultralytics.nn import

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
from multiprocessing import Pool

from ultralytics.models import yolo
from ultralytics.models.yolo.world.train import on_pretrain_routine_end
from ultralytics import __version__
from ultralytics.utils import (
    DEFAULT_CFG,
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
                                      count_trainable_parameters, crop_and_resize_largest_bbox_per_class,
                                      random_crop_img)

from v2vdet.v2vdet_ultralytics.models.v2vdet.val import (v2v_DetectionValidator, v2v_with_attn_pooling_DetectionValidator, v2v_template_feats_DetectionValidator)
from v2vdet.v2vdet_ultralytics.models.v2vdet.train.train_v2v_clip import (v2v_on_pretrain_routine_end, v2vWorldTrainer)
from v2vdet.v2vdet_ultralytics.nn.tasks import (V2V_Template_YOLO_Backbone_Model, V2V_Template_YOLO_Backbone_Share_Param_Model,
V2V_DINO_Model)
from v2vdet.v2vdet_ultralytics.data.build import (build_SA_V_v2v_dataset)

class V2V_Template_YOLO_Backbone_Trainer(v2vWorldTrainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)

  def build_dataset(self, img_path, mode="train", batch=None):
    """
    Build YOLO Dataset.

    Args:
        img_path (str): Path to the folder containing images.
        mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
        batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
    """
    gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
    return build_yolo_dataset(
        self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs
    )

  def get_model(self, cfg=None, weights=None, verbose=True):

    model = V2V_Template_YOLO_Backbone_Model(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(False)
    model.model.requires_grad_(True)
    # model.template_backbone_model.requires_grad_(True)
    # param=sum(p.numel() for p in model.template_backbone_model.parameters())

    # self.template_backbone_model = model.template_backbone_model

    self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model

  def get_validator(self):
      """Returns a DetectionValidator for YOLO model validation."""
      self.loss_names = "box_loss", "cls_loss", "dfl_loss"
      return v2v_template_feats_DetectionValidator(
          self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
      )

  # def _setup_train(self, world_size):
  #   """Builds dataloaders and optimizer on correct rank process."""
  #   super()._setup_train(world_size)

  def no_build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
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

  def do_train(self, world_size=1):
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

      self.multiprocessing_pool = Pool(processes=self.dataloader.num_workers * 2)
      for i, batch in pbar:
        self.run_callbacks(event="on_train_batch_start")
        self.optimizer.zero_grad(set_to_none=True)

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
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with autocast(self.amp):
          batch = self.preprocess_batch(batch=batch, crop_size=(224, 224))
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
              torch.dist.broadcast_object_list(broadcast_list, 0)
              self.stop = broadcast_list[0]
            if self.stop:  # training time exceeded
              break

        # self._clear_memory()
        # self.optimizer.zero_grad(set_to_none=True)
        # torch.cuda.empty_cache()

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
        
      self.multiprocessing_pool.close()
      self.multiprocessing_pool.join()

      self.lr = {f"lr/pg{ir}": x["lr"]
                 for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

      self.run_callbacks("on_train_epoch_end")
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
          }, step=epoch)
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
      # self._clear_memory()

      # Early Stopping
      if RANK != -1:  # if DDP training
        broadcast_list = [self.stop if RANK == 0 else None]
        # broadcast 'stop' to all ranks
        torch.dist.broadcast_object_list(broadcast_list, 0)
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

class V2V_Template_YOLO_Backbone_Share_Param_Trainer(v2vWorldTrainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)

  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    model = V2V_Template_YOLO_Backbone_Share_Param_Model(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(True)
    # model.model.requires_grad_(True)

    # self.template_backbone_model = model.template_backbone_model

    self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model

  # def get_validator(self):
  #     """Returns a DetectionValidator for YOLO model validation."""
  #     self.loss_names = "box_loss", "cls_loss", "dfl_loss"
  #     return v2v_template_feats_DetectionValidator(
  #         self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
  #     )

class SA_V_V2V_Template_YOLO_Backbone_Share_Param_Trainer(yolo.detect.DetectionTrainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)

  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    model = V2V_Template_YOLO_Backbone_Share_Param_Model(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(True)
    model.model.requires_grad_(True)

    self.template_backbone_model = model.template_backbone_model

    # self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model

  def get_validator(self):
      """Returns a DetectionValidator for YOLO model validation."""
      self.loss_names = "box_loss", "cls_loss", "dfl_loss"
      return v2v_template_feats_DetectionValidator(
          self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
      )

  def build_dataset(self, img_path, mode="train", batch=None):
      """
      Build V2V_Template_YOLO_Backbone_Share_Param Dataset.

      Args:
          img_path (List[str] | str): Path to the folder containing images.
          mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
          batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
      """

      gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
      dataset = [build_SA_V_v2v_dataset(self.args,
                                        img_path,
                                        batch,
                                        self.data,
                                        mode,
                                        stride=gs)]

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

  def preprocess_batch(self, batch):
    """Preprocesses a batch of images by scaling and converting to float."""
    batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
    batch['template_feats'] = batch['template_feats'].to(self.device, non_blocking=True).float() / 255
    if self.args.multi_scale:
      imgs = batch["img"]
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
      batch["img"] = imgs
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
      torch.dist.broadcast(self.amp, src=0)
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

      for i, batch in pbar:
        self.run_callbacks(event="on_train_batch_start")
        self.optimizer.zero_grad(set_to_none=True)

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
        with autocast(self.amp):
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
              torch.dist.broadcast_object_list(broadcast_list, 0)
              self.stop = broadcast_list[0]
            if self.stop:  # training time exceeded
              break

        # self._clear_memory()
        # self.optimizer.zero_grad(set_to_none=True)
        # torch.cuda.empty_cache()

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

      self.lr = {f"lr/pg{ir}": x["lr"]
                 for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

      self.run_callbacks("on_train_epoch_end")
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
        torch.dist.broadcast_object_list(broadcast_list, 0)
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

class V2V_Template_YOLO_Backbone_Share_Param_Only_Train_Linear_Layer_Trainer(V2V_Template_YOLO_Backbone_Share_Param_Trainer):
  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
      overrides = {}
    super().__init__(cfg, overrides, _callbacks)

  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    model = V2V_Template_YOLO_Backbone_Share_Param_Model(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(False)
    model.backbone_c2f_align_linear_layer.requires_grad_(True)

    # self.template_backbone_model = model.template_backbone_model

    # self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model

class v2v_DINO_Trainer(v2vWorldTrainer):
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

  def preprocess_batch(self, batch):
    """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""

    batch = super().preprocess_batch(batch)

    return batch

  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    model = V2V_DINO_Model(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)

    model.requires_grad_(False)
    model.model.requires_grad_(True)
    model.DINO_linear_layer.requires_grad_(True)
    model.vision_encoder.requires_grad_(False)

    self.vision_encoder = model.vision_encoder
    self.DINO_linear_layer = model.DINO_linear_layer

    # self.add_callback("on_pretrain_routine_end", v2v_on_pretrain_routine_end)

    return model



