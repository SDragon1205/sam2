import itertools
from v2vdet.v2vdet_ultralytics.nn import WorldModel, v2vdetModel, v2vWorldModel
from transformers import AutoImageProcessor, Dinov2Model
import torch
import numpy as np
import random, os
import copy
import time
import warnings
import gc
import math
import subprocess
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
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
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run

from ultralytics.utils.torch_utils import de_parallel
from ultralytics.data import build_yolo_dataset, YOLOConcatDataset, build_grounding
from ultralytics.data.utils import check_det_dataset

from v2vdet.v2vdet_ultralytics.utils import (extract_class_crops, 
                                      count_trainable_parameters, crop_and_resize_largest_bbox_per_class,
                                      random_crop_img)
from v2vdet.v2vdet_ultralytics.utils import process_images_parallel

from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import concurrent

def v2vdet_on_pretrain_routine_end(self, trainer):
    """Callback."""
    if RANK in {-1, 0}:
        # NOTE: for evaluation
        # names = [name.split("/")[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
        
        crop_img_list, _ = extract_class_crops(trainer.test_loader.dataset.labels)
        de_parallel(trainer.ema.ema).set_classes(query_crop_imgs=crop_img_list)
    device = next(trainer.model.parameters()).device

    # trainer.query_image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    trainer.Dinov2Model = Dinov2Model.from_pretrained("facebook/dinov2-base")
    trainer.Dinov2Model = trainer.Dinov2Model.to(device)

    # NOTE: freeze the parameters
    # for p in trainer.query_image_processor.parameters():
    #   p.requires_grad_(False)
    for param in trainer.Dinov2Model.parameters():
        param.requires_grad_(False)

    # for freeze_idx in range (10):
    #   for param in trainer.model.model[freeze_idx].parameters():
    #     param.requires_grad_(False)

    # trainer.ema.ema = trainer.model

def on_pretrain_routine_end(trainer):
    """Callback."""
    if RANK in {-1, 0}:
        # NOTE: for evaluation
        names = [name.split("/")[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
        de_parallel(trainer.ema.ema).set_classes(names, cache_clip_model=False)
    device = next(trainer.model.parameters()).device
    trainer.text_model, _ = trainer.clip.load("ViT-B/32", device=device)
    for p in trainer.text_model.parameters():
        p.requires_grad_(False)

def process_image_batch(args):
    img_idx, batch, num_classes, crop_size, batch_img = args
    
    class_crops = []
    
    random_img_indices = [random.randint(0, len(batch["im_file"])-1) 
                         for _ in range(num_classes)]
    random_imgs = [batch_img[idx] for idx in random_img_indices]
    
    class_crops = [random_crop_img(img, crop_size) for img in random_imgs]
    
    matches = (batch['batch_idx'] == img_idx).nonzero()
    if len(matches) > 0:
        batch_start = matches[0].item()
        batch_count = (batch['batch_idx'] == img_idx).sum()
        
        cropped_positives = crop_and_resize_largest_bbox_per_class(
            batch["img"][img_idx],
            batch['bboxes'][batch_start:batch_start + batch_count],
            batch['cls'][batch_start:batch_start + batch_count],
            size=crop_size
        )

        for crop_data in cropped_positives:
            class_crops[int(crop_data['cls'])] = crop_data['crop_img']
    
    return img_idx, class_crops

class WorldTrainer(yolo.detect.DetectionTrainer):
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

      # Import and assign clip
      try:
          import clip
      except ImportError:
          checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
          import clip
      self.clip = clip

    def get_model(self, cfg=None, weights=None, verbose=True):
      """Return WorldModel initialized with specified config and weights."""
      # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
      # NOTE: Following the official config, nc hard-coded to 80 for now.
      model = WorldModel(
          cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
          ch=3,
          nc=min(self.data["nc"], 80),
          verbose=verbose and RANK == -1,
      )

      if weights:
          model.load(weights)
      self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

      return model

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
          self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
      )

    def preprocess_batch(self, batch):
      """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""

      batch = super().preprocess_batch(batch)
      # NOTE: add text features
      texts = list(itertools.chain(*batch["texts"]))
      text_token = self.clip.tokenize(texts)
      text_token = text_token.to(self.device)
      txt_feats = self.text_model.encode_text(text_token)  # torch.float32
      txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
      batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
      
      return batch

class WorldTrainerFromScratch(WorldTrainer):
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
          build_yolo_dataset(self.args, im_path, batch, self.data, stride=gs, multi_modal=True)
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
      assert data_yaml.get("train", False), "train dataset not found"  # object365.yaml
      assert data_yaml.get("val", False), "validation dataset not found"  # lvis.yaml
      data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
      assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
      val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
      for d in data["val"]:
          if d.get("minival") is None:  # for lvis dataset
              continue
          d["minival"] = str(d["path"] / d["minival"])
      for s in ["train", "val"]:
          final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
          # save grounding data if there's one
          grounding_data = data_yaml[s].get("grounding_data")
          if grounding_data is None:
              continue
          grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
          for g in grounding_data:
              assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
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
      self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
      return super().final_eval()

'''
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

      # Import and assign clip
      try:
          import clip
      except ImportError:
          checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
          import clip
      self.clip = clip

    def get_model(self, cfg=None, weights=None, verbose=True):
      """Return WorldModel initialized with specified config and weights."""
      # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
      # NOTE: Following the official config, nc hard-coded to 80 for now.
      model = WorldModel(
          cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
          ch=3,
          nc=min(self.data["nc"], 80),
          verbose=verbose and RANK == -1,
      )

      if weights:
          model.load(weights)
      self.add_callback("v2v_on_pretrain_routine_end", v2v_on_pretrain_routine_end)

      return model

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
          self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
      )

    def preprocess_batch(self, batch):
      """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""

      batch = super().preprocess_batch(batch)
      # NOTE: add text features

      num_classes = self.model.yaml['nc']  # Usually 80
      batch_size = len(batch['img'])
      crop_size = (224, 224)
      
      # Prepare storage for batch embeddings
      crop_img_list = []
      random_crop = transforms.RandomCrop(size=crop_size)
      for img_idx, img in enumerate(batch['img']):
        single_sample_crop_img = [random_crop(batch['img'][random.randint(0, batch_size-1)]) for _ in range(num_classes)]
        matches = (batch['batch_idx'] == img_idx).nonzero()
        if len(matches) > 0:
          batch_start = (batch['batch_idx'] == img_idx).nonzero()[0].item()
          batch_count = (batch['batch_idx'] == img_idx).sum().item()
          img_classes = batch['cls'][batch_start:batch_start + batch_count]
          img_boxes = batch['bboxes'][batch_start:batch_start + batch_count]
          cropped_positives = crop_and_resize_largest_bbox_per_class(
            img, img_boxes, img_classes, size=crop_size
          )
          for crop_data in cropped_positives:
            single_sample_crop_img[int(crop_data['cls'])] = crop_data['crop_tensor_img'].to(self.device) 
        
        crop_img_list.extend(single_sample_crop_img)    

     
        # Extract features using CLIP
      with torch.inference_mode():
        self.text_model.eval()
        inputs = [crop_img_list[i:i + batch_size] for i in range(0, batch_size*num_classes, batch_size)]
        for clip_idx, clip_input in enumerate(inputs):
          clip_input = torch.stack(clip_input).squeeze(1).to(self.device)
          embeddings = self.text_model.encode_image(clip_input)  # torch.float32
          new_tensor = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
          if (clip_idx>0):
            crop_img_feats = torch.cat([crop_img_feats, new_tensor], dim=0)
          else:
            crop_img_feats = new_tensor.clone()

      # txt feats here is crop image embedding
      batch["txt_feats"] = crop_img_feats.reshape(-1, num_classes, crop_img_feats.shape[-1]).to("cpu").clone()
      return batch
    
    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
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
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                
                self.optimizer.zero_grad()
                
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
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
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                torch.cuda.empty_cache()
                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory()

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")

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
          build_yolo_dataset(self.args, im_path, batch, self.data, stride=gs, multi_modal=True)
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
      assert data_yaml.get("train", False), "train dataset not found"  # object365.yaml
      assert data_yaml.get("val", False), "validation dataset not found"  # lvis.yaml
      data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
      assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
      val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
      for d in data["val"]:
          if d.get("minival") is None:  # for lvis dataset
              continue
          d["minival"] = str(d["path"] / d["minival"])
      for s in ["train", "val"]:
          final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
          # save grounding data if there's one
          grounding_data = data_yaml[s].get("grounding_data")
          if grounding_data is None:
              continue
          grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
          for g in grounding_data:
              assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
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
      self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
      return super().final_eval()
'''
class v2vTrainer(yolo.detect.DetectionTrainer):
  """
  A class to fine-tune a v2v model on a close-set dataset.

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

    # Import and assign clip
    self.query_image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    self.Dinov2Model = Dinov2Model.from_pretrained("facebook/dinov2-base")

  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    model = v2vdetModel(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=3,
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )

    if weights:
      model.load(weights)
    self.add_callback("v2vdet_on_pretrain_routine_end", v2vdet_on_pretrain_routine_end)
    
    trainable_params, size_mb = count_trainable_parameters(model)
    LOGGER.info(f"Trainable para counts: {trainable_params/1000000:,}M")
    return model

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
        self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
    )
    
  def preprocess_batch(self, batch):
    """
    Preprocess batch for training by extracting image features using DINOv2.
    Includes both positive samples from actual objects and random negative samples.
    """
    batch = super().preprocess_batch(batch)
    
    num_classes = self.model.yaml['nc']  # Usually 80
    batch_size = self.batch_size
    crop_size = (256, 256)
    
    # Prepare storage for batch embeddings
    batch_embeddings = []
    
    # Process each image in batch
    for img_idx, img in enumerate(batch['img']):
      # Get class indices for current image
      
      # Initialize with negative samples by getting random crops from random images in batch all at once

      random_img_indices = [random.randint(0, len(batch["im_file"])-1) for _ in range(num_classes)]
      random_imgs = [Image.open(batch["im_file"][idx]).convert('RGB') for idx in random_img_indices]
      class_crops = [random_crop_img(img, crop_size) for img in random_imgs]
      
      matches = (batch['batch_idx'] == img_idx).nonzero()
      if len(matches) > 0:  # Did not match any class in the pic
        batch_start = (batch['batch_idx'] == img_idx).nonzero()[0].item()
        batch_count = (batch['batch_idx'] == img_idx).sum()
        img_classes = batch['cls'][batch_start:batch_start + batch_count]
        img_boxes = batch['bboxes'][batch_start:batch_start + batch_count]
      
        # Get positive samples - actual objects from image
        cropped_positives = crop_and_resize_largest_bbox_per_class(
          img, img_boxes, img_classes, size=crop_size
        )
        
        # Replace negative samples with positive ones where available
        for crop_data in cropped_positives:
          class_crops[int(crop_data['cls'])] = crop_data['crop_img']     

      # Extract features using DINOv2
      with torch.inference_mode():
        inputs = self.query_image_processor(images=class_crops, return_tensors="pt")
        inputs = inputs.to(self.device)
        self.Dinov2Model = self.Dinov2Model.to(self.device)
        self.Dinov2Model = self.Dinov2Model.half()
        embeddings = self.Dinov2Model(**inputs).pooler_output
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1) # Normalize embeddings
        embeddings = embeddings * 0.01 # Scale embeddings, make it similar to CLIP
        batch_embeddings.append(embeddings)
        # Clean up GPU memory
        # del inputs
        # torch.cuda.empty_cache()

    # Stack all embeddings into batch tensor
    batch['cls_emb'] = torch.stack(batch_embeddings).to("cpu")
    return batch
      
class v2vTrainerFromScratch(v2vTrainer):
  """
  A class extending the v2vTrainer class for training a world model from scratch on open-set dataset.

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
        build_yolo_dataset(self.args, im_path, batch, self.data, stride=gs, multi_modal=True)
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
    assert data_yaml.get("train", False), "train dataset not found"  # object365.yaml
    assert data_yaml.get("val", False), "validation dataset not found"  # lvis.yaml
    data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
    assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
    val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
    for d in data["val"]:
        if d.get("minival") is None:  # for lvis dataset
            continue
        d["minival"] = str(d["path"] / d["minival"])
    for s in ["train", "val"]:
        final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
        # save grounding data if there's one
        grounding_data = data_yaml[s].get("grounding_data")
        if grounding_data is None:
            continue
        grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
        for g in grounding_data:
            assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
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
    self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
    return super().final_eval()