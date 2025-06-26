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
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.plotting import output_to_target, plot_images

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
from v2vdet.v2vdet_ultralytics.data import build_yolo_dataset, build_object_oriented_yolo_dataset

class ObjectOrientedDetectionValidator(DetectionValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    This class implements validation functionality specific to object detection tasks, including metrics calculation,
    prediction processing, and visualization of results.

    Attributes:
        nt_per_class (np.ndarray): Number of targets per class.
        nt_per_image (np.ndarray): Number of targets per image.
        is_coco (bool): Whether the dataset is COCO.
        is_lvis (bool): Whether the dataset is LVIS.
        class_map (list): Mapping from model class indices to dataset class indices.
        metrics (DetMetrics): Object detection metrics calculator.
        iouv (torch.Tensor): IoU thresholds for mAP calculation.
        niou (int): Number of IoU thresholds.
        lb (list): List for storing ground truth labels for hybrid saving.
        jdict (list): List for storing JSON detection results.
        stats (dict): Dictionary for storing statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionValidator
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml")
        >>> validator = DetectionValidator(args=args)
        >>> validator()
    """
    
    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)
        
    def preprocess(self, batch):
        """
        Preprocess batch of images for YOLO validation.

        Args:
            batch (dict): Batch containing images and annotations.

        Returns:
            (dict): Preprocessed batch.
        """
        for i_t in ['i', 't']:
          batch[i_t]["img"] = batch[i_t]["img"].to(self.device, non_blocking=True)
          batch[i_t]["img"] = (batch[i_t]["img"].half() if self.args.half else batch[i_t]["img"].float()) / 255
          for k in ["batch_idx", "cls", "bboxes"]:
              batch[i_t][k] = batch[i_t][k].to(self.device)

        if self.args.save_hybrid and self.args.task == "detect":
            height, width = batch['i']["img"].shape[2:]
            nb = len(batch['i']["img"])
            bboxes = batch['i']["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch['i']["cls"][batch["batch_idx"] == i], bboxes[batch['i']["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch

    # def init_metrics(self, model):
    #     """
    #     Initialize evaluation metrics for YOLO detection validation.

    #     Args:
    #         model (torch.nn.Module): Model to validate.
    #     """
    #     val = self.data.get(self.args.split, "")  # validation path
    #     self.is_coco = (
    #         isinstance(val, str)
    #         and "coco" in val
    #         and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
    #     )  # is COCO
    #     self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
    #     self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
    #     self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
    #     self.names = model.names
    #     self.nc = len(model.names)
    #     self.end2end = getattr(model, "end2end", False)
    #     self.metrics.names = self.names
    #     self.metrics.plot = self.args.plots
    #     self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
    #     self.seen = 0
    #     self.jdict = []
    #     self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    # def get_desc(self):
    #     """Return a formatted string summarizing class metrics of YOLO model."""
    #     return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    # def postprocess(self, preds):
    #     """
    #     Apply Non-maximum suppression to prediction outputs.

    #     Args:
    #         preds (torch.Tensor): Raw predictions from the model.

    #     Returns:
    #         (List[torch.Tensor]): Processed predictions after NMS.
    #     """
    #     return ops.non_max_suppression(
    #         preds,
    #         self.args.conf,
    #         self.args.iou,
    #         labels=self.lb,
    #         nc=self.nc,
    #         multi_label=True,
    #         agnostic=self.args.single_cls or self.args.agnostic_nms,
    #         max_det=self.args.max_det,
    #         end2end=self.end2end,
    #         rotated=self.args.task == "obb",
    #     )

    # def _prepare_batch(self, si, batch):
    #     """
    #     Prepare a batch of images and annotations for validation.

    #     Args:
    #         si (int): Batch index.
    #         batch (dict): Batch data containing images and annotations.

    #     Returns:
    #         (dict): Prepared batch with processed annotations.
    #     """
    #     idx = batch['i']["batch_idx"] == si
    #     cls = batch['i']["cls"][idx].squeeze(-1)
    #     bbox = batch['i']["bboxes"][idx]
    #     ori_shape = batch['i']["ori_shape"][si]
    #     imgsz = batch['i']["img"].shape[2:]
    #     ratio_pad = batch['i']["ratio_pad"][si]
    #     if len(cls):
    #         bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
    #         ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
    #     return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    # def _prepare_pred(self, pred, pbatch):
    #     """
    #     Prepare predictions for evaluation against ground truth.

    #     Args:
    #         pred (torch.Tensor): Model predictions.
    #         pbatch (dict): Prepared batch information.

    #     Returns:
    #         (torch.Tensor): Prepared predictions in native space.
    #     """
    #     predn = pred.clone()
    #     ops.scale_boxes(
    #         pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
    #     )  # native-space pred
    #     return predn

    def update_metrics(self, preds, batch):
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (List[torch.Tensor]): List of predictions from the model.
            batch (dict): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch['i'])
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['i']["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['i']['im_file'][si]).stem}.txt",
                )


    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        dataset = build_object_oriented_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)
        # if isinstance(dataset, YOLOConcatDataset):
        #     for d in dataset.datasets:
        #         d.transforms.append(LoadVisualPrompt())
        # else:
        #     dataset.transforms.append(LoadVisualPrompt())
        return dataset

    def plot_val_samples(self, batch, ni):
        """
        Plot validation image samples.

        Args:
            batch (dict): Batch containing images and annotations.
            ni (int): Batch index.
        """
        for gg in ['i', 't']:
          if gg == 'i':
            name = self.save_dir / f"val_batch{ni}_input_without_groundtruth.jpg"
          else:
            name = self.save_dir / f"val_batch{ni}_template_without_groundtruth.jpg"
          
          plot_images(
              images=batch[gg]["img"],
              batch_idx=batch[gg]["batch_idx"],
              cls=batch[gg]["cls"].squeeze(-1),
              paths=batch[gg]["im_file"],
              fname=name,
              on_plot=self.on_plot,
          )
        
        for gg in ['i', 't']:
          if gg == 'i':
            name = self.save_dir / f"val_batch{ni}_input.jpg"
          else:
            name = self.save_dir / f"val_batch{ni}_template.jpg"
          
          plot_images(
              images=batch[gg]["img"],
              batch_idx=batch[gg]["batch_idx"],
              cls=batch[gg]["cls"].squeeze(-1),
              bboxes=batch[gg]["bboxes"],
              paths=batch[gg]["im_file"],
              fname=name,
              on_plot=self.on_plot,
          )

    def plot_predictions(self, batch, preds, ni):
        """
        Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (dict): Batch containing images and annotations.
            preds (List[torch.Tensor]): List of predictions from the model.
            ni (int): Batch index.
        """
        plot_images(
            batch['i']["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch['i']["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            # names=self.names,  # If you want to show class names
            on_plot=self.on_plot,
        )  # pred

class v2v_with_SAVPE_ObjectOriented_DetectionValidator(ObjectOrientedDetectionValidator):
  def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)

  # def preprocess(self, batch):
  #   """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""

  #   batch = super().preprocess(batch)

  #   return batch
  
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
    
    model.names = self.data['names']
    self.init_metrics(de_parallel(model))
    self.jdict = []  # empty before each val
    
    # self.names = self.data['names']
    # self.nc = self.data['nc']
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
          preds = model(batch['i']['img'], batch)

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