# Ultralytics Eric YOLO 🚀, AGPL-3.0 license

from v2vdet.v2vdet_ultralytics.utils.loss import (v2vDetectionLoss,
                                           v2v_E2EDetectLoss,
                                           v2v_E2EDetectLoss,
                                           v2v_v8DetectionLoss)
from v2vdet.v2vdet_ultralytics.utils.misc import load_images
from v2vdet.v2vdet_ultralytics.nn.modules import (
    C2fAttn,
    ImagePoolingAttn,
    WorldDetect,
    C2f_v2v_Attn,
    A2C2f_Template_MaxSigmoidAttn,
    TemplateAttentionPooling,
    MultiLevelTemplateAttentionPooling,
    MultiHeadTemplateAttentionPooling,
    v2vDetect
)
from v2vdet.v2vdet_ultralytics.nn.modules.block import PATCH_EMBEDDING_SAVPE

import v2vdet.v2vdet_ultralytics.perception_models.core.vision_encoder.pe as pe
from ultralytics.nn.tasks import (yaml_model_load,
                                  guess_model_scale)
from ultralytics.nn import (BaseModel)
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics import YOLO
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, YAML
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect
)

from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.torch_utils import (smart_inference_mode)
from ultralytics.nn.autobackend import check_class_names

from transformers import AutoImageProcessor, Dinov2Model
from transformers import (CLIPProcessor, CLIPVisionModel, CLIPImageProcessor, SiglipVisionModel, CLIPVisionModelWithProjection, BatchFeature)
from transformers import AutoModel, AutoProcessor

import torchvision.transforms as transforms
from torch.utils.checkpoint import checkpoint
from typing import List, Union
from PIL import Image
import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
import pickle
import contextlib
import os
import sys
from pathlib import Path
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

class DetectionModel(BaseModel):
    """YOLO detection model."""

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """
        Initialize the YOLO detection model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                else:
                  if isinstance(m, (Segment, Pose, OBB)):
                    return self.forward(x)[0]
                  else:
                    gg = self.forward(x)
                    if isinstance(gg, tuple):
                      return gg[0]
                    else:
                      return gg
                # return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """
        Perform augmentations on input image x and return augmented inference and train outputs.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (torch.Tensor): Augmented inference output.
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """
        De-scale predictions following augmented inference (inverse operation).

        Args:
            p (torch.Tensor): Predictions tensor.
            flips (int): Flip type (0=none, 2=ud, 3=lr).
            scale (float): Scale factor.
            img_size (tuple): Original image size (height, width).
            dim (int): Dimension to split at.

        Returns:
            (torch.Tensor): De-scaled predictions.
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """
        Clip YOLO augmented inference tails.

        Args:
            y (List[torch.Tensor]): List of detection tensors.

        Returns:
            (List[torch.Tensor]): Clipped detection tensors.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class v2vdetModel(DetectionModel):
  """v2vdet Model."""

  # model, input channels, number of classes
  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize v2vdetModel world model with given config and parameters."""
    self.cls_emb = torch.randn(
        1, nc or 80, 768)  # features placeholder, during initialize will go feature function which need this, or will be error

    self.Dinov2Model = None  # DINO model placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

  def set_classes(self, query_crop_imgs, batch=80):
    """
    Set class embeddings for the model using query crop images processed through DINOv2.
    This method initializes a DINOv2 image processor and encoder, processes query crop images
    to generate class embeddings, and updates the model's number of classes accordingly.
    Args:
      query_crop_imgs (list): List of tuples containing (class_id, cropped_image).
        Cropped images should be PIL Image objects containing individual object instances.
        Can be generated using extract_class_crops() from v2vdet.v2vdet_ultralytics.utils.
      batch (int, optional): Batch size for processing images. Defaults to 80.
    Returns:
      None: Updates self.cls_emb with class embeddings and modifies model's number of classes
    Example:
      >>> from v2vdet.v2vdet_ultralytics.utils import extract_class_crops
      >>> crop_img_list, _ = extract_class_crops(trainer.test_loader.dataset.labels)
      >>> model.set_classes(crop_img_list)
    Note:
      - Requires facebook/dinov2-base model
      - Updates self.cls_emb with class embeddings
      - Modifies self.model[-1].nc to match number of classes
    """

    self.query_image_processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-base")
    self.Dinov2Model = Dinov2Model.from_pretrained("facebook/dinov2-base")

    crop_img_list_cls = [crop_img[0] for crop_img in query_crop_imgs]
    crop_img_list_only_img = [crop_img[1] for crop_img in query_crop_imgs]

    inputs = self.query_image_processor(
        crop_img_list_only_img, return_tensors="pt")

    self.Dinov2Model = self.Dinov2Model.to(
        'cuda' if torch.cuda.is_available() else 'cpu')
    self.Dinov2Model.eval()
    inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.inference_mode():
      outputs = self.Dinov2Model(**inputs)

    self.cls_emb = outputs.pooler_output.to('cpu')
    self.model[-1].nc = self.cls_emb.shape[0]

  def predict(self, x, profile=False, visualize=False, cls_emb=None, augment=False, embed=None):
    """
    Perform a forward pass through the model.

    Args:
        x (torch.Tensor): The input tensor.
        profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
        visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
        txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
        augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
        embed (list, optional): A list of feature vectors/embeddings to return.

    Returns:
        (torch.Tensor): Model's output tensor.
    """

    cls_emb = (self.cls_emb if cls_emb is None else cls_emb).to(
        device=x.device, dtype=x.dtype)
    if len(cls_emb) != len(x):
      cls_emb = cls_emb.repeat(len(x), 1, 1)
    ori_cls_emb = cls_emb.clone()
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:  # except the head part
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [
            x if j == -1 else y[j] for j in m.f]  # from earlier layers
      if profile:
        self._profile_one_layer(m, x, dt)
      if isinstance(m, C2f_v2v_Attn):
        x = m(x, cls_emb)
      elif isinstance(m, WorldDetect):
        x = m(x, ori_cls_emb)
      elif isinstance(m, ImagePoolingAttn):
        cls_emb = m(x, cls_emb)
      else:
        x = m(x)  # run

      y.append(x if m.i in self.save else None)  # save output
      if visualize:
        feature_visualization(x, m.type, m.i, save_dir=visualize)
      if embed and m.i in embed:
        embeddings.append(nn.functional.adaptive_avg_pool2d(
            x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
        if m.i == max(embed):
          return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x

  def init_criterion(self):
    """Initialize the loss criterion for the DetectionModel."""
    return v2v_E2EDetectLoss(self) if getattr(self, "end2end", False) else v2vDetectionLoss(self)

  def loss(self, batch, preds=None):
    """
    Compute loss.

    Args:
        batch (dict): Batch to compute loss on.
        preds (torch.Tensor | List[torch.Tensor]): Predictions.
    """
    if not hasattr(self, "criterion"):
      self.criterion = self.init_criterion()

    if preds is None:
      preds = self.forward(batch["img"], cls_emb=batch["cls_emb"])

    return self.criterion(preds, batch)

  def crop_img_preprocess(self, targets, batch_size):
    nl, ne = targets.shape
    if nl == 0:
      out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
    else:
      i = targets[:, 0]  # image index
      _, counts = i.unique(return_counts=True)
      counts = counts.to(dtype=torch.int32)
      out = torch.zeros(batch_size, counts.max(), ne - 1)
      for j in range(batch_size):
        matches = i == j
        n = matches.sum()
        if n:
          out[j, :n] = targets[matches, 1:]

class v2vBaseModel(DetectionModel):
  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True, init_txt_feats=None):
    """Initialize YOLOv8 world model with given config and parameters."""
    self.txt_feats = init_txt_feats or torch.randn(1, nc or 80, 512)  # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model

  def loss(self, batch, preds=None):
    """
    Compute loss.

    Args:
        batch (dict): Batch to compute loss on.
        preds (torch.Tensor | List[torch.Tensor]): Predictions.
    """
    if not hasattr(self, "criterion"):
      self.criterion = self.init_criterion()
    if preds is None:
      preds = self.forward(batch["img"], template_feats=batch["template_feats"])
    return self.criterion(preds, batch)

  def predict(self, x, profile=False, visualize=False, template_feats=None, augment=False, embed=None):
    """
    Perform a forward pass through the model.

    Args:
        x (torch.Tensor): The input tensor.
        profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
        visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
        txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
        augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
        embed (list, optional): A list of feature vectors/embeddings to return.

    Returns:
        (torch.Tensor): Model's output tensor.
    """
    bs = self.bs = x.shape[0]  # batch size

    if template_feats is not None:
      self.set_classes(template_feats, bs=bs)

    template_feats = self.template_feats.to(
          device=x.device, dtype=x.dtype)

    if len(template_feats) != len(x):
      template_feats = template_feats.repeat(len(x), 1, 1)

    ori_template_feats = template_feats.clone()
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:  # except the head part
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [
            x if j == -1 else y[j] for j in m.f]  # from earlier layers
      if profile:
        self._profile_one_layer(m, x, dt)
      if isinstance(m, C2fAttn):
        x = m(x, template_feats)
      elif isinstance(m, WorldDetect):
        x = m(x, ori_template_feats)
      elif isinstance(m, ImagePoolingAttn):
        template_feats = m(x, template_feats)
      else:
        x = m(x)  # run

      y.append(x if m.i in self.save else None)  # save output
      if visualize:
        feature_visualization(x, m.type, m.i, save_dir=visualize)
      if embed and m.i in embed:
        embeddings.append(nn.functional.adaptive_avg_pool2d(
            x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
        if m.i == max(embed):
          return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x

class v2v_Template_feats_BaseModel(DetectionModel):
  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True, init_txt_feats=None):
    """Initialize YOLOv8 world model with given config and parameters."""
    self.txt_feats = init_txt_feats or torch.randn(1, nc or 80, 512)  # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model

  def loss(self, batch, preds=None):
    """
    Compute loss.

    Args:
        batch (dict): Batch to compute loss on.
        preds (torch.Tensor | List[torch.Tensor]): Predictions.
    """
    if not hasattr(self, "criterion"):
      self.criterion = self.init_criterion()
    if preds is None:
      preds = self.forward(batch["img"], template_feats=batch["txt_feats"])
    return self.criterion(preds, batch)

class v2vWorldModel(DetectionModel):
  """v2v World Model. (Image clip)"""

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""
    self.template_feats = torch.randn(1, nc or 80, 512)  # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model

    self.vision_processor = CLIPImageProcessor().from_pretrained(pretrained_model_name_or_path="openai/clip-vit-base-patch32")
    self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")  # CLIP model placeholder

  def set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor], bs=1):
    """Set classes in advance so that model could do offline-inference without clip model."""

    if isinstance(crop_imgs, torch.Tensor):
      if crop_imgs.dim() == 5:
        bs = crop_imgs.shape[0]
        class_num = crop_imgs.shape[1]
        crop_imgs = crop_imgs.view(-1, *crop_imgs.shape[2:])
      else:
        class_num = crop_imgs.shape[0]

      nc = crop_imgs.shape[0]//bs
      clip_input = BatchFeature(
        {'pixel_values': crop_imgs}, tensor_type='pt')
    elif isinstance(crop_imgs, List):
      nc = len(crop_imgs)//bs
      if isinstance(crop_imgs[0], Image.Image):
        clip_input = self.vision_processor.preprocess(
        crop_imgs, return_tensors="pt")
      else:
        raise ValueError("crop_imgs should be List[Image.Image] or torch.Tensor")
    else:
      raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")
      # nc = self.model[-1].nc = self.yaml['nc']

    self.nc = nc

    clip_input = clip_input.to(self.vision_encoder.device)
    vision_model_output = self.vision_encoder(
      **clip_input,
      output_hidden_states=False,
      return_dict=True)['image_embeds']

    # cls_token = deepcopy(vision_model_output)
    crop_img_feats = vision_model_output / vision_model_output.norm(dim=-1, keepdim=True)
    self.template_feats = crop_img_feats.reshape(bs,
                                            nc, crop_img_feats.shape[-1])
    self.model[-1].nc = nc

  def predict(self, x, template_feats=None, profile=False, visualize=False, augment=False, embed=None):
    """
    Perform a forward pass through the model.

    Args:
        x (torch.Tensor): The input tensor.
        profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
        visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
        template_feats (torch.Tensor): The template image features, use it if it's given. Defaults to None.
        augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
        embed (list, optional): A list of feature vectors/embeddings to return.

    Returns:
        (torch.Tensor): Model's output tensor.
    """
    bs = self.bs = x.shape[0]  # batch size

    if template_feats is not None:
      self.set_classes(template_feats, bs)

    template_feats = self.template_feats.to(device=x.device, dtype=x.dtype)

    if len(template_feats) != len(x):
      template_feats = template_feats.repeat(len(x), 1, 1)

    template_feats_origin = template_feats.clone()
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:  # except the head part
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [
            x if j == -1 else y[j] for j in m.f]  # from earlier layers
      if profile:
        self._profile_one_layer(m, x, dt)
      if isinstance(m, C2fAttn):
        x = m(x, template_feats)
      elif isinstance(m, WorldDetect):
        x = m(x, template_feats_origin)
      elif isinstance(m, ImagePoolingAttn):
        template_feats = m(x, template_feats)
      else:
        x = m(x)  # run

      y.append(x if m.i in self.save else None)  # save output
      if visualize:
        feature_visualization(x, m.type, m.i, save_dir=visualize)
      if embed and m.i in embed:
        embeddings.append(nn.functional.adaptive_avg_pool2d(
            x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
        if m.i == max(embed):
          return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x

  def loss(self, batch, preds=None):
    """
    Compute loss.

    Args:
        batch (dict): Batch to compute loss on.
        preds (torch.Tensor | List[torch.Tensor]): Predictions.
    """
    if not hasattr(self, "criterion"):
      self.criterion = self.init_criterion()
    if preds is None:
      preds = self.forward(batch["img"], template_feats=batch["template_feats"])
    return self.criterion(preds, batch)

class V2V_with_Patch_Attn_Pooling_Model(v2vWorldModel):
  """V2V_with_Patch_Attn_Pooling. (Take CLIP patch's to do attention pooling)"""

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""

    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model
    self.attn_pooling = TemplateAttentionPooling(
        hidden_size=768, proj_size=512)
    # self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

  @torch.inference_mode()
  def set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor], batch=16, cache_clip_model=True):
    """Set classes in advance so that model could do offline-inference without clip model."""
    super().set_classes(crop_imgs, batch, cache_clip_model)
    self.attn_pooling_result = self.attn_pooling(self.hidden_states[-3])
    self.txt_feats = self.attn_pooling_result['pooled_feature_proj'].reshape(-1,
            len(crop_imgs),
            self.attn_pooling_result['pooled_feature_proj'].shape[-1])

  @torch.inference_mode()
  def batches_set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor], bs, cache_clip_model=True):
    """Set classes in advance so that model could do offline-inference without clip model."""
    super().batches_set_classes(crop_imgs, bs, cache_clip_model)

    device = f"{next(self.attn_pooling.parameters()).device}"
    attn_pooling_input = self.hidden_states[-3].clone()
    batch = len(attn_pooling_input)//self.nc + 1

    result = []
    for batch_idx in range(batch):
      start_idx = batch_idx * self.nc
      end_idx = min(start_idx + self.nc, len(attn_pooling_input))
      if (start_idx >= end_idx):
        break
      attn_pooling_input_batch = attn_pooling_input[start_idx:end_idx].to(device)
      attn_pooling_result = self.attn_pooling(attn_pooling_input_batch)
      result.append(attn_pooling_result['pooled_feature_proj'].clone())
      del attn_pooling_result
      del attn_pooling_input_batch

    self.attn_pooling_result = {'pooled_feature_proj': torch.cat(result, dim=0)}
    self.txt_feats = self.attn_pooling_result['pooled_feature_proj'].reshape(bs, self.nc, self.attn_pooling_result['pooled_feature_proj'].shape[-1])


class V2V_with_2_Patch_Attn_Pooling_Model(DetectionModel):
  """V2V_with_Patch_Attn_Pooling. (Take CLIP patch's to do attention pooling)"""

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""

    self.txt_feats = [torch.randn(1, nc or 80, 512) for i in range (2)]  # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model

    self.vision_processor = CLIPImageProcessor().from_pretrained(
        pretrained_model_name_or_path="openai/clip-vit-base-patch32")
    self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32")  # CLIP model placeholder
    self.attn_pooling = nn.ModuleList([TemplateAttentionPooling(
        hidden_size=768, proj_size=512) for i in range(2)])

    # self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

  def set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor], batch=16, cache_clip_model=True):
    """Set classes in advance so that model could do offline-inference without clip model."""
    nc = self.model[-1].nc = self.yaml['nc']
    self.nc = nc

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    for param in self.vision_encoder.parameters():
      param.requires_grad = False

    if isinstance(crop_imgs, torch.Tensor):
      from transformers import BatchFeature
      clip_input = BatchFeature(
          {'pixel_values': crop_imgs}, tensor_type='pt')
    else:
      clip_input = self.vision_processor.preprocess(
          crop_imgs, return_tensors="pt")

    clip_input = clip_input.to(self.vision_encoder.device)
    # with torch.inference_mode():
    vision_model_output = self.vision_encoder(
        **clip_input, output_hidden_states=True, return_dict=True)
    self.hidden_states = [vision_model_output["hidden_states"][-2], vision_model_output["hidden_states"][-3]]

    self.attn_pooling_result = [self.attn_pooling[temp_idx](self.hidden_states[temp_idx]) for temp_idx in range(2)]
    self.txt_feats = [self.attn_pooling_result[temp_i]['pooled_feature_proj'].reshape(-1, len(crop_imgs), self.attn_pooling_result[temp_i]['pooled_feature_proj'].shape[-1]) for temp_i in range(2)]

    self.model[-1].nc = len(crop_imgs)

  def batches_set_classes(self, crop_img_list_tensor: Union[List[Image.Image], torch.Tensor], bs):
    if not hasattr(self, 'nc'):
      self.nc = self.yaml['nc']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # nc = self.model[-1].nc = self.nc
    # if isinstance(crop_img_list_tensor, torch.Tensor):
    #   crop_img_list_tensor = crop_img_list_tensor.to('cpu')
    nc = crop_img_list_tensor.shape[0]//bs

    if isinstance(crop_img_list_tensor, torch.Tensor):
      # crop_img_list_tensor = crop_img_list_tensor.to('cuda')
      with torch.inference_mode():
        vision_model_output = self.vision_encoder(crop_img_list_tensor, output_hidden_states=True, return_dict=True)

    elif isinstance(crop_img_list_tensor, List[Image.Image]):
      clip_input = self.vision_processor.preprocess(crop_img_list_tensor, return_tensors="pt ")
      with torch.inference_mode():
        vision_model_output = self.vision_encoder(clip_input['pixel_values'].to(self.vision_encoder.device), output_hidden_states=True, return_dict=True)

    else:
      raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")

    # * Let CLIP's patch pass into attention pooling layer!
    self.attn_pooling_result = [self.attn_pooling[0](vision_model_output['hidden_states'][-2]), self.attn_pooling[1](vision_model_output['hidden_states'][-3])]
    self.txt_feats = [self.attn_pooling_result[temp_idx]['pooled_feature_proj'].clone().reshape(
                    -1,
                    nc,
                    self.attn_pooling_result[temp_idx]['pooled_feature_proj'].shape[-1])
                    for temp_idx in range(2)]

    del vision_model_output, self.attn_pooling_result

  def dataloader_set_classes(self, crop_img_list_tensor, bs):

    if not hasattr(self, 'nc'):
      self.nc = 80
    # nc = self.model[-1].nc = self.nc
    crop_img_list_tensor = crop_img_list_tensor.to('cpu')
    nc = crop_img_list_tensor.shape[0]//bs
    # nc = self.nc

    dataset = torch.utils.data.TensorDataset(crop_img_list_tensor)
    dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=bs*8,
      pin_memory=True,
      num_workers=4
    )

    if torch.cuda.is_available():
      device = "cuda"
    else:
      device = "cpu"

    all_cls_tokens = []
    attn_pooling_result = [[], []]

    for batch in dataloader:
      batch[0] = batch[0].to('cuda')
      with torch.inference_mode():
        vision_model_output = self.vision_encoder(batch[0], output_hidden_states=True, return_dict=True)

      attn_pooling_result[0].append(self.attn_pooling[0](vision_model_output['hidden_states'][-2])['pooled_feature_proj'])
      attn_pooling_result[1].append(self.attn_pooling[1](vision_model_output['hidden_states'][-3])['pooled_feature_proj'])

    all_attn_pooling_result = [torch.cat(attn_pooling_result[temp_idx], dim=0) for temp_idx in range(2)]
    all_attn_pooling_result_norm = [all_attn_pooling_result[temp_idx]/all_attn_pooling_result[temp_idx].norm(p=2, dim=-1, keepdim=True) for temp_idx in range(2)]
    self.txt_feats = [all_attn_pooling_result_norm[temp_idx].reshape(-1, nc, all_attn_pooling_result_norm[temp_idx].shape[-1]) for temp_idx in range(2)]

  def predict(self, x, profile=False, visualize=False, txt_feats=None, crop_img_list=None, augment=False, embed=None):
    """
    Perform a forward pass through the model.

    Args:
        x (torch.Tensor): The input tensor.
        profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
        visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
        txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
        augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
        embed (list, optional): A list of feature vectors/embeddings to return.

    Returns:
        (torch.Tensor): Model's output tensor.
    """
    bs = self.bs = x.shape[0]  # batch size

    if txt_feats is not None:
      # self.dataloader_set_classes(txt_feats, bs=bs)
      self.batches_set_classes(txt_feats, bs)
      txt_feats = [self.txt_feats[temp_idx].to(
          device=x.device, dtype=x.dtype) for temp_idx in range(2)]
    elif crop_img_list is not None:
      self.dataloader_set_classes(crop_img_list, bs)
      txt_feats = [self.txt_feats[temp_idx].to(
          device=x.device, dtype=x.dtype) for temp_idx in range(2)]
    else:
      txt_feats = [self.txt_feats[temp_idx].to(x.device, x.dtype) for temp_idx in range(2)]
      # txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(
      #     device=x.device, dtype=x.dtype)

    for temp_idx, temp_txt_feats in enumerate(txt_feats):
      if temp_txt_feats is None: pass
      elif len(temp_txt_feats) != len(x):
        txt_feats[temp_idx] = txt_feats[temp_idx].repeat(len(x), 1, 1)

    ori_txt_feats = [txt_feats[temp_idx].clone() for temp_idx in range(2)]
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:  # except the head part
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [
            x if j == -1 else y[j] for j in m.f]  # from earlier layers
      if profile:
        self._profile_one_layer(m, x, dt)
      if isinstance(m, C2fAttn):
        if m.i == 12 or m.i == 19:
          x = m(x, txt_feats[0])
        elif m.i == 15 or m.i == 22:
          x = m(x, txt_feats[1])
      elif isinstance(m, WorldDetect):
        x = m(x, ori_txt_feats[0])
      elif isinstance(m, ImagePoolingAttn):
        txt_feats[0] = m(x, txt_feats[0])
      else:
        x = m(x)  # run

      y.append(x if m.i in self.save else None)  # save output
      if visualize:
        feature_visualization(x, m.type, m.i, save_dir=visualize)
      if embed and m.i in embed:
        embeddings.append(nn.functional.adaptive_avg_pool2d(
            x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
        if m.i == max(embed):
          return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x

  def loss(self, batch, preds=None):
    """
    Compute loss.

    Args:
        batch (dict): Batch to compute loss on.
        preds (torch.Tensor | List[torch.Tensor]): Predictions.
    """
    if not hasattr(self, "criterion"):
      self.criterion = self.init_criterion()
    if preds is None:
      preds = self.forward(batch["img"], txt_feats=batch["template_feats"])
    return self.criterion(preds, batch)

class V2V_multi_scale_clip_Model(v2vBaseModel):
  """V2V with multi scaling clip patch. (Take multiple CLIP patch's to do attention pooling)
  
  Define the layer you want in the .yaml, using like this:
  >>> want_layers: [-1, -3, -5]
  """
  

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""

    self.template_feats = torch.randn(1, nc or 80, 512)  # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model

    self.vision_processor = CLIPImageProcessor().from_pretrained(
        pretrained_model_name_or_path="openai/clip-vit-base-patch32")
    self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path="openai/clip-vit-base-patch32")  # CLIP model placeholder
    
    self.want_layers = self.yaml['want_layers'] if (self.yaml['want_layers'] is not None) else [-1]
    
    LOGGER.info(f"Using layers: {self.want_layers} for multi scale attention pooling.")
    
    self.multi_scale_attn_pooling = MultiLevelTemplateAttentionPooling(hidden_size=768, proj_size=512, num_levels=len(self.want_layers))

    # self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

  def set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor], bs=1):
    """Set classes in advance so that model could do offline-inference without clip model."""

    if isinstance(crop_imgs, torch.Tensor):
      if crop_imgs.dim() == 5:
        bs = crop_imgs.shape[0]
        class_num = crop_imgs.shape[1]
        crop_imgs = crop_imgs.view(-1, *crop_imgs.shape[2:])
      else:
        class_num = crop_imgs.shape[0]

      nc = crop_imgs.shape[0]//bs
      vision_encoder_input = BatchFeature(
        {'pixel_values': crop_imgs}, tensor_type='pt')
    elif isinstance(crop_imgs, List):
      nc = len(crop_imgs)//bs
      if isinstance(crop_imgs[0], Image.Image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        tensor_list = [transform(img) for img in crop_imgs]

        nc = len(tensor_list)

        vision_encoder_input = torch.stack(tensor_list)
      else:
        raise ValueError("crop_imgs should be List[Image.Image] or torch.Tensor")
    else:
      raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")
      # nc = self.model[-1].nc = self.yaml['nc']

    self.nc = nc

    if (nc>256):
      for batch_idx in range(0, bs):
        batch_input = vision_encoder_input['pixel_values'][nc*batch_idx:nc*(batch_idx+1)]
        batch_input = batch_input.to(self.vision_encoder.device)
        vision_model_output = self.vision_encoder(
          batch_input,
          output_hidden_states=True,
          return_dict=True)

        temp_hidden_states = [vision_model_output['hidden_states'][h_idx] for h_idx in self.want_layers]
        temp_attn_pooling_result = self.multi_scale_attn_pooling(temp_hidden_states)['pooled_feature_proj']
        if batch_idx == 0:
          attn_pooling_result = temp_attn_pooling_result
        else:
          attn_pooling_result = torch.cat([attn_pooling_result, temp_attn_pooling_result], dim=0)

    else:
      vision_encoder_input = vision_encoder_input.to(self.vision_encoder.device)
      if isinstance(vision_encoder_input, dict) or isinstance(vision_encoder_input, BatchFeature):
        vision_model_output = self.vision_encoder(
          **vision_encoder_input,
          output_hidden_states=True,
          return_dict=True)
      else:
        vision_model_output = self.vision_encoder(
          vision_encoder_input,
          output_hidden_states=True,
          return_dict=True)

      hidden_states = [[] for _ in range(len(self.want_layers))]
      for idx, layer_num in enumerate(self.want_layers):
        hidden_states[idx] = vision_model_output['hidden_states'][layer_num]

      attn_pooling_result = self.multi_scale_attn_pooling(hidden_states)['pooled_feature_proj']

    self.template_feats = attn_pooling_result.reshape(-1,
                                            nc,
                                            attn_pooling_result.shape[-1])

    self.model[-1].nc = nc

  def predict(self, x, profile=False, visualize=False, template_feats=None, augment=False, embed=None):
    """
    Perform a forward pass through the model.

    Args:
        x (torch.Tensor): The input tensor.
        profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
        visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
        txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
        augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
        embed (list, optional): A list of feature vectors/embeddings to return.

    Returns:
        (torch.Tensor): Model's output tensor.
    """
    bs = self.bs = x.shape[0]  # batch size

    if template_feats is not None:
      self.set_classes(template_feats, bs=bs)

    template_feats = self.template_feats.to(
          device=x.device, dtype=x.dtype)

    if len(template_feats) != len(x):
      template_feats = template_feats.repeat(len(x), 1, 1)

    ori_template_feats = template_feats.clone()
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:  # except the head part
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [
            x if j == -1 else y[j] for j in m.f]  # from earlier layers
      if profile:
        self._profile_one_layer(m, x, dt)
      if isinstance(m, C2fAttn):
        x = m(x, template_feats)
      elif isinstance(m, WorldDetect):
        x = m(x, ori_template_feats)
      elif isinstance(m, ImagePoolingAttn):
        template_feats = m(x, template_feats)
      else:
        x = m(x)  # run

      y.append(x if m.i in self.save else None)  # save output
      if visualize:
        feature_visualization(x, m.type, m.i, save_dir=visualize)
      if embed and m.i in embed:
        embeddings.append(nn.functional.adaptive_avg_pool2d(
            x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
        if m.i == max(embed):
          return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x

class V2V_Template_YOLO_Backbone_Model(DetectionModel):

  if torch.cuda.is_available():
    device = "cuda"
  elif torch.backends.mps.is_available():
    device = "mps"
  else:
    device = "cpu"

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""

    self.template_feats = torch.randn(1, nc or 80, 512)
    # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model
    self._init_vision_encoder(cfg="ultralytics/cfg/models/v8/yolov8s.yaml", ckpt="ckpt/yolov8s-world.pt", ch=ch, nc=nc, verbose=verbose)

  def _init_vision_encoder(self, cfg="yolov8s.yaml", ckpt=None, ch=3, nc=None, verbose=True) -> None:
    """This function is _init_vision_encoder. This function will initial vision encoder in yolo model for template images. Will produce a vision encoder ***self.template_backbone_model*** with pretrain weight if given.

    Args:
        cfg (str, optional): _description_. Defaults to "yolov8s.yaml".
        ckpt (_type_, optional): _description_. Defaults to None.
        ch (int, optional): _description_. Defaults to 3.
        nc (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to True.
    """

    template_backbone_model = YOLO(model=cfg, task='detect')

    if (ckpt is not None):
        template_backbone_model = template_backbone_model.load(ckpt)

    self.template_backbone_model = template_backbone_model.model
    LOGGER.info(f"self.template_backbone_model has loaded!")

  def _vision_encoder_forward(self, x, profile=False, visualize=False, embed=None, return_tensor=False):
    
    """
    Perform a forward pass through the network.

    Args:
      x (torch.Tensor): The input tensor to the model.
      profile (bool):  Print the computation time of each layer if True, defaults to False.
      visualize (bool): Save the feature maps of the model if True, defaults to False.
      embed (list, optional): A list of feature vectors/embeddings to return.

    Returns:
      (torch.Tensor): The last output of the model.
    """

    first_param = next(self.template_backbone_model.parameters())
    if x.dtype != first_param.dtype:
        LOGGER.debug(f"Converting input tensor from {x.dtype} to {first_param.dtype}")
        x = x.to(dtype=first_param.dtype)

    y, dt, embeddings = [], [], []  # outputs

    for m in self.template_backbone_model:
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
      if profile:
        self._profile_one_layer(m, x, dt)

      # utilize Gradient Checkpoint mod to avoid VRAM OOM issue.
      # if m.i<5:
      #   x = checkpoint(m, x)
      # else:
      # x = m(x)  # run
      if hasattr(self.args, 'gradient_checkpointing'):  
        if self.args.gradient_checkpointing:
          x = checkpoint(m, x)
        else:
          x = m(x)    
      else:
        x = m(x)
      
      y.append(x if m.i in self.save else None)  # save output
      if visualize:
        feature_visualization(x, m.type, m.i, save_dir=visualize)
      if embed and m.i in embed:
        embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
        if m.i == max(embed):
          if return_tensor:
            return torch.cat(embeddings, 1)
          else:
            return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x

  def set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor]):
    """Set classes in advance so that model could do offline-inference without template backbone model."""

    if isinstance(crop_imgs, torch.Tensor):
      if crop_imgs.dim() == 5:
        bs = crop_imgs.shape[0]
        class_num = crop_imgs.shape[1]
        crop_imgs = crop_imgs.view(-1, *crop_imgs.shape[2:])
      else:
        class_num = crop_imgs.shape[0] // self.bs
        bs = self.bs
      nc = class_num

    elif isinstance(crop_imgs, List):
      nc = len(crop_imgs)
      if isinstance(crop_imgs[0], Image.Image or np.ndarray):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        tensor_list = [transform(img) for img in crop_imgs]
        # with open ("inf.pkl", 'rb') as f:
        #   inf = pickle.load(f)
        # tensor_list = [inf['template_feats'][i] for i in range(80)]
        # for i in range(len(temp_tensor_list)):
        #   tensor_list[i] = temp_tensor_list[i]

        nc = len(tensor_list)

      elif isinstance(crop_imgs[0], torch.Tensor):
        transform = transforms.Compose([
            transforms.Resize((224, 224))])
        tensor_list = crop_imgs
      else:
        raise NotImplementedError("crop_imgs should be either List[Image.Image] or torch.Tensor")

      crop_imgs = torch.stack(tensor_list)

    else:
      nc = self.model[-1].nc = self.yaml['nc']

    self.nc = nc
    self.model[-1].nc = nc

    # if (nc>256):
    #   step = 4
    #   for batch_idx in range(0, bs, step):
    #     batch_clip_input = crop_imgs[nc*batch_idx:nc*(batch_idx+step)]
    #     batch_clip_input = batch_clip_input.to(self.device)
    #     temp_template_feats = self._vision_encoder_forward(batch_clip_input, embed=[9], return_tensor=True)
    #     if batch_idx == 0:
    #       template_feats = temp_template_feats
    #     else:
    #       template_feats = torch.cat([template_feats, temp_template_feats], dim=0)
    # else:
    crop_imgs = crop_imgs.to(self.device)
    template_feats = self._vision_encoder_forward(crop_imgs, embed=[9], return_tensor=True)
    # 9 is the last layer of the yolo backbone (SPPF layer)
    template_feats = template_feats.reshape(-1, nc, template_feats.shape[-1])

    template_feats_norm = template_feats / template_feats.norm(p=2, dim=-1, keepdim=True)
    self.template_feats = template_feats_norm
    
    if self.backbone_c2f_align_linear_layer is not None:
      self.template_feats = self.backbone_c2f_align_linear_layer(self.template_feats)

  def predict(self, x, profile=False, visualize=False, template_feats=None, augment=False, embed=None):
    """
    Perform a forward pass through the model.

    Args:
        x (torch.Tensor): The input tensor.
        profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
        visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
        template_feats (torch.Tensor): The template image features, use it if it's given. Defaults to None.
        augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
        embed (list, optional): A list of feature vectors/embeddings to return.

    Returns:
        (torch.Tensor): Model's output tensor.
    """
    bs = self.bs = x.shape[0]  # batch size

    if template_feats is not None:
      self.set_classes(template_feats)

    x = x.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
    template_feats = self.template_feats.to(device=x.device, dtype=x.dtype)

    if len(template_feats) != len(x):
      template_feats = template_feats.repeat(len(x), 1, 1)

    ori_template_feats = template_feats.clone()
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:  # except the head part
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [
            x if j == -1 else y[j] for j in m.f]  # from earlier layers
      if profile:
        self._profile_one_layer(m, x, dt)
      if isinstance(m, C2fAttn):
        x = m(x, template_feats)
      elif isinstance(m, WorldDetect):
        x = m(x, ori_template_feats)
      elif isinstance(m, ImagePoolingAttn):
        template_feats = m(x, template_feats)
      else:
        x = m(x)  # run

      y.append(x if m.i in self.save else None)  # save output
      if visualize:
        feature_visualization(x, m.type, m.i, save_dir=visualize)
      if embed and m.i in embed:
        embeddings.append(nn.functional.adaptive_avg_pool2d(
            x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
        if m.i == max(embed):
          return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x

  def loss(self, batch, preds=None):
    """
    Compute loss.

    Args:
        batch (dict): Batch to compute loss on
        preds (torch.Tensor | List[torch.Tensor]): Predictions.
    """
    if getattr(self, "criterion", None) is None:
        self.criterion = self.init_criterion()

    preds = self.forward(batch["img"], template_feats=batch["template_feats"]) if preds is None else preds
    return self.criterion(preds, batch)

class V2V_Template_YOLO_Backbone_Model_Contrastive_Loss_Model(DetectionModel):

  if torch.cuda.is_available():
    device = "cuda"
  elif torch.backends.mps.is_available():
    device = "mps"
  else:
    device = "cpu"

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""

    self.template_feats = torch.randn(1, nc or 80, 512)
    # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model
    self._init_vision_encoder(cfg="ultralytics/cfg/models/v8/yolov8s.yaml", ckpt="ckpt/yolov8s-world.pt", ch=ch, nc=nc, verbose=verbose)

  def _init_vision_encoder(self, cfg="yolov8s.yaml", ckpt=None, ch=3, nc=None, verbose=True) -> None:
    """This function is _init_vision_encoder. This function will initial vision encoder in yolo model for template images. Will produce a vision encoder ***self.template_backbone_model*** with pretrain weight if given.

    Args:
        cfg (str, optional): _description_. Defaults to "yolov8s.yaml".
        ckpt (_type_, optional): _description_. Defaults to None.
        ch (int, optional): _description_. Defaults to 3.
        nc (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to True.
    """

    # template_backbone_model = YOLO(model=cfg, task='detect')

    # if (ckpt is not None):
    #     template_backbone_model = template_backbone_model.load(ckpt)

    # self.template_backbone_model = template_backbone_model.model

    self.template_backbone_model = self.model[:10]
    if (self.yaml['scale'] == "m"):
      ## 
      self.backbone_c2f_align_linear_layer = nn.Sequential(
              nn.Linear(in_features=576, out_features=512),
              nn.LayerNorm(512),
              nn.SiLU()
      )
      ## 
    else:
      self.backbone_c2f_align_linear_layer = None
    LOGGER.info(f"self.template_backbone_model has loaded!")

  def _vision_encoder_forward(self, x, profile=False, visualize=False, embed=None, return_tensor=False):
    
    """
    Perform a forward pass through the network.

    Args:
      x (torch.Tensor): The input tensor to the model.
      profile (bool):  Print the computation time of each layer if True, defaults to False.
      visualize (bool): Save the feature maps of the model if True, defaults to False.
      embed (list, optional): A list of feature vectors/embeddings to return.

    Returns:
      (torch.Tensor): The last output of the model.
    """

    first_param = next(self.template_backbone_model.parameters())
    if x.dtype != first_param.dtype:
        LOGGER.debug(f"Converting input tensor from {x.dtype} to {first_param.dtype}")
        x = x.to(dtype=first_param.dtype)

    y, dt, embeddings = [], [], []  # outputs

    for m in self.template_backbone_model:
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
      if profile:
        self._profile_one_layer(m, x, dt)

      # utilize Gradient Checkpoint mod to avoid VRAM OOM issue.
      # if m.i<5:
      #   x = checkpoint(m, x)
      # else:
      # x = m(x)  # run
      x = checkpoint(m, x)
      y.append(x if m.i in self.save else None)  # save output
      if visualize:
        feature_visualization(x, m.type, m.i, save_dir=visualize)
      if embed and m.i in embed:
        embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
        if m.i == max(embed):
          if return_tensor:
            return torch.cat(embeddings, 1)
          else:
            return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x

  def set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor]):
    """Set classes in advance so that model could do offline-inference without template backbone model."""

    if isinstance(crop_imgs, torch.Tensor):
      if crop_imgs.dim() == 5:
        bs = crop_imgs.shape[0]
        class_num = crop_imgs.shape[1]
        crop_imgs = crop_imgs.view(-1, *crop_imgs.shape[2:])
      else:
        class_num = crop_imgs.shape[0] // self.bs
        bs = self.bs
      nc = class_num

    elif isinstance(crop_imgs, List):
      nc = len(crop_imgs)
      if isinstance(crop_imgs[0], Image.Image or np.ndarray):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        tensor_list = [transform(img) for img in crop_imgs]
        # with open ("inf.pkl", 'rb') as f:
        #   inf = pickle.load(f)
        # tensor_list = [inf['template_feats'][i] for i in range(80)]
        # for i in range(len(temp_tensor_list)):
        #   tensor_list[i] = temp_tensor_list[i]

        nc = len(tensor_list)

      elif isinstance(crop_imgs[0], torch.Tensor):
        transform = transforms.Compose([
            transforms.Resize((224, 224))])
        tensor_list = crop_imgs
      else:
        raise NotImplementedError("crop_imgs should be either List[Image.Image] or torch.Tensor")

      crop_imgs = torch.stack(tensor_list)

    else:
      nc = self.model[-1].nc = self.yaml['nc']

    self.nc = nc
    self.model[-1].nc = nc

    # if (nc>256):
    #   step = 4
    #   for batch_idx in range(0, bs, step):
    #     batch_clip_input = crop_imgs[nc*batch_idx:nc*(batch_idx+step)]
    #     batch_clip_input = batch_clip_input.to(self.device)
    #     temp_template_feats = self._vision_encoder_forward(batch_clip_input, embed=[9], return_tensor=True)
    #     if batch_idx == 0:
    #       template_feats = temp_template_feats
    #     else:
    #       template_feats = torch.cat([template_feats, temp_template_feats], dim=0)
    # else:
    crop_imgs = crop_imgs.to(self.device)
    template_feats = self._vision_encoder_forward(crop_imgs, embed=[9], return_tensor=True)
    # 9 is the last layer of the yolo backbone (SPPF layer)
    template_feats = template_feats.reshape(-1, nc, template_feats.shape[-1])

    template_feats_norm = template_feats / template_feats.norm(p=2, dim=-1, keepdim=True)
    self.template_feats = template_feats_norm
    
    if self.backbone_c2f_align_linear_layer is not None:
      self.template_feats = self.backbone_c2f_align_linear_layer(self.template_feats)

  def predict(self, x, profile=False, visualize=False, template_feats=None, augment=False, embed=None):
    """
    Perform a forward pass through the model.

    Args:
        x (torch.Tensor): The input tensor.
        profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
        visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
        template_feats (torch.Tensor): The template image features, use it if it's given. Defaults to None.
        augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
        embed (list, optional): A list of feature vectors/embeddings to return.

    Returns:
        (torch.Tensor): Model's output tensor.
    """
    bs = self.bs = x.shape[0]  # batch size

    if template_feats is not None:
      self.set_classes(template_feats)

    x = x.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
    template_feats = self.template_feats.to(device=x.device, dtype=x.dtype)

    if len(template_feats) != len(x):
      template_feats = template_feats.repeat(len(x), 1, 1)

    ori_template_feats = template_feats.clone()
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:  # except the head part
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [
            x if j == -1 else y[j] for j in m.f]  # from earlier layers
      if profile:
        self._profile_one_layer(m, x, dt)
      if isinstance(m, C2fAttn):
        x = m(x, template_feats)
      elif isinstance(m, WorldDetect):
        # Only request loss outputs if training; during evaluation, use inference mode.
        if self.training:
            out = m(x, ori_template_feats, return_loss_outputs=True)
            if isinstance(out, tuple):
                x, bbox_feats = out
            else:
                x = out
        else:
            # In eval mode, do NOT pass return_loss_outputs; the head returns final predictions.
            x = m(x, ori_template_feats)
        # out = m(x, ori_template_feats, return_loss_outputs=True)
        # if isinstance(out, tuple):
        #         x, bbox_feats = out
        # else:
        #     x = out
        # x, bbox_feats = m(x, ori_template_feats)
        # x = m(x, ori_template_feats)
      elif isinstance(m, ImagePoolingAttn):
        template_feats = m(x, template_feats)
      else:
        x = m(x)  # run

      y.append(x if m.i in self.save else None)  # save output
      if visualize:
        feature_visualization(x, m.type, m.i, save_dir=visualize)
      if embed and m.i in embed:
        embeddings.append(nn.functional.adaptive_avg_pool2d(
            x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
        if m.i == max(embed):
          return torch.unbind(torch.cat(embeddings, 1), dim=0)
    if self.training:
        return x, bbox_feats, ori_template_feats
    else:
        return x
    # return x, bbox_feats, ori_template_feats
    # return x

  def loss(self, batch, preds=None):
    """
    Compute loss.

    Args:
        batch (dict): Batch to compute loss on
        preds (torch.Tensor | List[torch.Tensor]): Predictions.
    """
    if getattr(self, "criterion", None) is None:
      self.criterion = self.init_criterion()

    if preds is None:
      preds, bbox_feats, template_feats = self.predict(batch["img"], template_feats=batch["template_feats"])
      return self.criterion(preds, batch, bbox_feats, template_feats)
    else:
      n_cls = int(batch["template_feats"].shape[0] / batch["img"].shape[0])
      return self.criterion(preds, batch, n_cls=n_cls)
      # return self.criterion(preds, batch)
    # else:
    #     preds, bbox_feats, template_feats = preds

    # preds = self.forward(batch["img"], template_feats=batch["template_feats"]) if preds is None else preds
    # return self.criterion(preds, batch)
  def init_criterion(self):
    """Initialize the loss criterion for the DetectionModel."""
    # return v2v_E2EDetectLoss(self) if getattr(self, "end2end", False) else v2vDetectionLoss(self)
    return v2v_E2EDetectLoss(self) if getattr(self, "end2end", False) else v2v_v8DetectionLoss(self)

class V2V_Template_YOLO_Backbone_Share_Param_Model(V2V_Template_YOLO_Backbone_Model):

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

  def _init_vision_encoder(self, cfg="yolov8s.yaml", ckpt=None, ch=3, nc=None, verbose=True) -> None:
    """
    This function is _init_vision_encoder. This function will initial vision encoder in yolo model for template images. Will produce a vision encoder ***self.template_backbone_model*** with pretrain weight if given.
    """

    self.template_backbone_model = self.model[:10]
    if (self.yaml['scale'] == "m"):
      ## 
      self.backbone_c2f_align_linear_layer = nn.Sequential(
              nn.Linear(in_features=576, out_features=512),
              nn.LayerNorm(512),
              nn.SiLU()
      )
      ## 
    else:
      self.backbone_c2f_align_linear_layer = None
    # before layer 10 is yolov8's backbone
    # LOGGER.info(f"self.template_backbone_model has loaded!")  
  
  
class V2V_Template_YOLO_Backbone_Share_Param_For_Only_Train_Linear_Layer(V2V_Template_YOLO_Backbone_Model):

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

  def _init_vision_encoder(self, cfg="yolov8s.yaml", ckpt=None, ch=3, nc=None, verbose=True) -> None:
    """
    This function is _init_vision_encoder. This function will initial vision encoder in yolo model for template images. Will produce a vision encoder ***self.template_backbone_model*** with pretrain weight if given.
    """

    # self.template_backbone_model = YOLO("yolov8m-world.pt").model.model[:10]
    self.template_backbone_model = self.model[:10]
    self.backbone_c2f_align_linear_layer = nn.Sequential(
              nn.Linear(576, 512),
              nn.BatchNorm1d(512),
              nn.SiLU() # Same as the output of the backbone (SPPF layer)
    )
    # if (self.yaml['scale'] == "m"):
    #   self.backbone_c2f_align_linear_layer = nn.Linear(576, out_features=512)
    # else:
    #   self.backbone_c2f_align_linear_layer = None
    # before layer 10 is yolov8's backbone
    # LOGGER.info(f"self.template_backbone_model has loaded!")

class V2V_DINO_Model(v2vBaseModel):
  """V2V with DINO input. (Take DINO into model)"""

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""

    from transformers import Dinov2Config, Dinov2Model

    self.template_feats = torch.randn(1, nc or 80, 512)  # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model

    self.vision_processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path='facebook/dinov2-base')
    self.vision_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path='facebook/dinov2-base')  # CLIP model placeholder
    self.MLP = nn.Linear(in_features=768, out_features=512)

    # self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

  def set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor], bs=1):
    """Set classes in advance so that model could do offline-inference without clip model."""

    if isinstance(crop_imgs, torch.Tensor):
      if crop_imgs.dim() == 5:
        bs = crop_imgs.shape[0]
        crop_imgs = crop_imgs.view(-1, *crop_imgs.shape[2:])

      nc = crop_imgs.shape[0]//bs
      vision_enc_input = BatchFeature(
        {'pixel_values': crop_imgs}, tensor_type='pt')
    elif isinstance(crop_imgs, List):
      nc = len(crop_imgs)//bs
      if isinstance(crop_imgs[0], Image.Image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        temp_tensor_list = [transform(img) for img in crop_imgs]
        with open ("inf.pkl", 'rb') as f:
          inf = pickle.load(f)
        tensor_list = [inf['template_feats'][i] for i in range(nc)]
        for i in range(len(temp_tensor_list)):
          tensor_list[i] = temp_tensor_list[i]

        nc = len(tensor_list)

        # clip_input = self.vision_processor.preprocess(
        # crop_imgs, return_tensors="pt")

        vision_enc_input = torch.stack(tensor_list)
      else:
        raise ValueError("crop_imgs should be List[Image.Image] or torch.Tensor")
    else:
      raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")
      # nc = self.model[-1].nc = self.yaml['nc']

    self.nc = nc

    # with torch.inference_mode():
      # if (nc>256):
      #   for batch_idx in range(0, bs):
      #     batch_vision_enc_input = vision_enc_input['pixel_values'][nc*batch_idx:nc*(batch_idx+1)]
      #     batch_vision_enc_input = batch_vision_enc_input.to(self.vision_encoder.device)
      #     vision_model_output = self.vision_encoder(
      #       batch_vision_enc_input,
      #       output_hidden_states=False,
      #       return_dict=True)

      #     # temp_hidden_states = [vision_model_output['hidden_states'][h_idx] for h_idx in want_layer]
      #     # temp_attn_pooling_result = self.multi_scale_attn_pooling(temp_hidden_states)['pooled_feature_proj']
      #     if batch_idx == 0:
      #       dino_result = vision_model_output['pooler_output']
      #     else:
      #       dino_result = torch.cat([dino_result, vision_model_output['pooler_output']], dim=0)

      # else:
    vision_enc_input = vision_enc_input.to(self.vision_encoder.device)
    if isinstance(vision_enc_input, dict) or isinstance(vision_enc_input, BatchFeature):
        vision_model_output = self.vision_encoder(
          **vision_enc_input,
          output_hidden_states=False,
          return_dict=True)
    else:
      vision_model_output = self.vision_encoder(
        vision_enc_input,
        output_hidden_states=False,
        return_dict=True)

    # hidden_states = [[] for _ in range(len(want_layer))]
    # for idx, layer_num in enumerate(want_layer):
    #   hidden_states[idx] = vision_model_output['last_hidden_state']

    dino_result = vision_model_output['pooler_output']

      # attn_pooling_result = self.multi_scale_attn_pooling(hidden_states)['pooled_feature_proj']

    dino_result = dino_result.reshape(-1, nc, dino_result.shape[-1])
    # self.template_feats = checkpoint(self.MLP, dino_result)
    self.template_feats = self.MLP(dino_result)
    self.model[-1].nc = nc

  def predict(self, x, profile=False, visualize=False, template_feats=None, augment=False, embed=None):
    """
    Perform a forward pass through the model.

    Args:
        x (torch.Tensor): The input tensor.
        profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
        visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
        txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
        augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
        embed (list, optional): A list of feature vectors/embeddings to return.

    Returns:
        (torch.Tensor): Model's output tensor.
    """
    bs = self.bs = x.shape[0]  # batch size

    if template_feats is not None:
      self.set_classes(template_feats, bs=bs)

    template_feats = self.template_feats.to(
          device=x.device, dtype=x.dtype)

    if len(template_feats) != len(x):
      template_feats = template_feats.repeat(len(x), 1, 1)

    ori_template_feats = template_feats.clone()
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:  # except the head part
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [
            x if j == -1 else y[j] for j in m.f]  # from earlier layers
      if profile:
        self._profile_one_layer(m, x, dt)
      if isinstance(m, C2fAttn):
        x = m(x, template_feats)
      elif isinstance(m, WorldDetect):
        x = m(x, ori_template_feats)
      elif isinstance(m, ImagePoolingAttn):
        template_feats = m(x, template_feats)
      else:
        x = m(x)  # run

      y.append(x if m.i in self.save else None)  # save output
      if visualize:
        feature_visualization(x, m.type, m.i, save_dir=visualize)
      if embed and m.i in embed:
        embeddings.append(nn.functional.adaptive_avg_pool2d(
            x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
        if m.i == max(embed):
          return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x

class V2V_template_DINO_multi_scale_Model(V2V_DINO_Model):
  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model
    
    self.want_layers = self.yaml['want_layers'] if (self.yaml['want_layers'] is not None) else [-1]
    LOGGER.info(f"Using layers: {self.want_layers} for multi scale attention pooling.")
    
    del self.MLP
    self.ATTN_POOLING = MultiLevelTemplateAttentionPooling(hidden_size=768, proj_size=512, num_patches=257, num_levels=len(self.want_layers))

  def set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor], bs=1):
    """Set classes in advance so that model could do offline-inference without clip model."""

    if isinstance(crop_imgs, torch.Tensor):
      if crop_imgs.dim() == 5:
        bs = crop_imgs.shape[0]
        class_num = crop_imgs.shape[1]
        crop_imgs = crop_imgs.view(-1, *crop_imgs.shape[2:])
      else:
        class_num = crop_imgs.shape[0]

      nc = crop_imgs.shape[0]//bs
      vision_encoder_input = BatchFeature(
        {'pixel_values': crop_imgs}, tensor_type='pt')
    elif isinstance(crop_imgs, List):
      nc = len(crop_imgs)//bs
      if isinstance(crop_imgs[0], Image.Image):
        tensor_list = [self.transform(img) for img in crop_imgs]

        nc = len(tensor_list)
        vision_encoder_input = torch.stack(tensor_list)
      else:
        raise ValueError("crop_imgs should be List[Image.Image] or torch.Tensor")
    else:
      raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")
    self.nc = nc
    
    vision_encoder_input = vision_encoder_input.to(self.vision_encoder.device)
    if isinstance(vision_encoder_input, dict) or isinstance(vision_encoder_input, BatchFeature):
      vision_model_output = self.vision_encoder(
        **vision_encoder_input,
        output_hidden_states=True,
        return_dict=True)
    else:
      vision_model_output = self.vision_encoder(
        vision_encoder_input,
        output_hidden_states=True,
        return_dict=True)

    hidden_states = [[] for _ in range(len(self.want_layers))]
    for idx, layer_num in enumerate(self.want_layers):
      hidden_states[idx] = vision_model_output['hidden_states'][layer_num]

    attn_pooling_result = self.ATTN_POOLING(hidden_states)['pooled_feature_proj']

    self.template_feats = attn_pooling_result.reshape(-1,
                                          nc,
                                          attn_pooling_result.shape[-1])

    self.model[-1].nc = nc
    
class V2V_DINO_with_registers_Model(V2V_DINO_Model):
  """V2V with DINO (with register version) input.
     VISION TRANSFORMERS NEED REGISTERS: https://arxiv.org/pdf/2309.16588
  """

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""

    from transformers import Dinov2Config, Dinov2Model

    self.template_feats = torch.randn(1, nc or 80, 512)  # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model

    self.vision_processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path='facebook/dinov2-with-registers-base')
    self.vision_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path='facebook/dinov2-with-registers-base')  
    
class V2V_template_DINO_with_registers_multi_scale_Model(V2V_template_DINO_multi_scale_Model):

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model
    self.vision_processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path='facebook/dinov2-with-registers-base')
    self.vision_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path='facebook/dinov2-with-registers-base')  
    
    self.ATTN_POOLING = MultiLevelTemplateAttentionPooling(hidden_size=768, proj_size=512, num_patches=261, num_levels=len(self.want_layers))

class WorldModel(DetectionModel):
  """YOLOv8 World Model."""

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""
    self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
    self.clip_model = None  # CLIP model placeholder

    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model

  def set_classes(self, text, batch=80, cache_clip_model=True):
    """Set classes in advance so that model could do offline-inference without clip model."""

    # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # if (
    #     not getattr(self, "clip_model", None) and cache_clip_model
    # ):  # for backwards compatibility of models lacking clip_model attribute
    #     self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # model = self.clip_model if cache_clip_model else CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    try:
      import clip
    except ImportError:
      check_requirements("git+https://github.com/ultralytics/CLIP.git")
      import clip

    if (
        not getattr(self, "clip_model", None) and cache_clip_model
    ):  # for backwards compatibility of models lacking clip_model attribute
      self.clip_model = clip.load("ViT-B/32")[0]

    model = self.clip_model if cache_clip_model else clip.load(
        "ViT-B/32")[0]
    device = next(model.parameters()).device

    text_token = clip.tokenize(text).to(device)
    # text_token = self.processor(text = text, return_tensors="pt", padding=True).to(device)

    txt_feats = [model.encode_text(token).detach()
                 for token in text_token.split(batch)]
    txt_feats = txt_feats[0] if len(
        txt_feats) == 1 else torch.cat(txt_feats, dim=0)
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
    self.model[-1].nc = len(text)

  def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
    """
    Perform a forward pass through the model.

    Args:
        x (torch.Tensor): The input tensor.
        profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
        visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
        txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
        augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
        embed (list, optional): A list of feature vectors/embeddings to return.

    Returns:
        (torch.Tensor): Model's output tensor.
    """
    txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(
        device=x.device, dtype=x.dtype)
    # if (txt_feats is None):
    #     txt_feats = self.txt_feats.clone()
    # txt_feats = txt_feats.to(device=x.device, dtype=x.dtype)

    if len(txt_feats) != len(x):
      txt_feats = txt_feats.repeat(len(x), 1, 1)
    ori_txt_feats = txt_feats.clone()
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:  # except the head part
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [
            x if j == -1 else y[j] for j in m.f]  # from earlier layers
      if profile:
        self._profile_one_layer(m, x, dt)
      if isinstance(m, C2fAttn):
        x = m(x, txt_feats)
      elif isinstance(m, WorldDetect):
        x = m(x, ori_txt_feats)
        # x = m(x, txt_feats)
      elif isinstance(m, ImagePoolingAttn):
        txt_feats = m(x, txt_feats)
      else:
        x = m(x)  # run

      y.append(x if m.i in self.save else None)  # save output
      if visualize:
        feature_visualization(x, m.type, m.i, save_dir=visualize)
      if embed and m.i in embed:
        embeddings.append(nn.functional.adaptive_avg_pool2d(
            x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
        if m.i == max(embed):
          return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x

  def loss(self, batch, preds=None):
    """
    Compute loss.

    Args:
        batch (dict): Batch to compute loss on.
        preds (torch.Tensor | List[torch.Tensor]): Predictions.
    """
    if not hasattr(self, "criterion"):
      self.criterion = self.init_criterion()
    if preds is None:
      if batch["txt_feats"] is None:
        breakpoint()
      preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
    return self.criterion(preds, batch)

class V2V_template_SigLIP_Model(v2vBaseModel):
  """V2V with SigLIP patch. (Take SigLIP's class token')"""

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""

    self.template_feats = torch.randn(1, nc or 80, 512)  # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model

    self.vision_processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path="google/siglip-base-patch16-224")
    self.vision_encoder = SiglipVisionModel.from_pretrained(
        pretrained_model_name_or_path="google/siglip-base-patch16-224")  
    # self.multi_scale_attn_pooling = MultiLevelTemplateAttentionPooling(hidden_size=768, proj_size=512, num_levels=1)
    self.MLP = nn.Linear(in_features=768, out_features=512)

    # self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    self.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

  def set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor], bs=1):
    """Set classes in advance so that model could do offline-inference without clip model."""

    if isinstance(crop_imgs, torch.Tensor):
      if crop_imgs.dim() == 5:
        bs = crop_imgs.shape[0]
        class_num = crop_imgs.shape[1]
        crop_imgs = crop_imgs.view(-1, *crop_imgs.shape[2:])
      else:
        class_num = crop_imgs.shape[0]

      nc = crop_imgs.shape[0]//bs
      vision_encoder_input = BatchFeature(
        {'pixel_values': crop_imgs}, tensor_type='pt')
    elif isinstance(crop_imgs, List):
      nc = len(crop_imgs)//bs
      if isinstance(crop_imgs[0], Image.Image):
        tensor_list = [self.transform(img) for img in crop_imgs]

        nc = len(tensor_list)
        vision_encoder_input = torch.stack(tensor_list)
      else:
        raise ValueError("crop_imgs should be List[Image.Image] or torch.Tensor")
    else:
      raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")
    self.nc = nc

    # if (nc>256):
    #   for batch_idx in range(0, bs):
    #     batch_vision_encoder_input = vision_encoder_input['pixel_values'][nc*batch_idx:nc*(batch_idx+1)].to(self.vision_encoder.device)
    #     vision_model_output = self.vision_encoder(
    #       batch_vision_encoder_input,
    #       return_dict=True)

    #     mlp_out = self.MLP(vision_model_output['pooler_output'])
    #     if batch_idx == 0:
    #       result = mlp_out
    #     else:
    #       result = torch.cat([result, mlp_out], dim=0)

    # else:
    vision_encoder_input = vision_encoder_input.to(self.vision_encoder.device)
    if isinstance(vision_encoder_input, dict) or isinstance(vision_encoder_input, BatchFeature):
      vision_model_output = self.vision_encoder(
        **vision_encoder_input,
        return_dict=True)
    else:
      vision_model_output = self.vision_encoder(
        vision_encoder_input,
        return_dict=True)

    result = self.MLP(vision_model_output['pooler_output'])

    self.template_feats = result.reshape(-1,
                                          nc,
                                          result.shape[-1])

    self.model[-1].nc = nc

class V2V_template_SigLIP_with_new_dataset_Model(V2V_template_SigLIP_Model):
  """V2V with SigLIP patch. (Take SigLIP's class token')"""

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""
    self.template_feats = torch.randn(1, nc or 80, 512)  # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model
  
  def set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor], bs=1):
    """Set classes in advance so that model could do offline-inference without clip model."""

    if isinstance(crop_imgs, torch.Tensor):
      if crop_imgs.dim() == 5:
        bs = crop_imgs.shape[0]
        class_num = crop_imgs.shape[1]
        crop_imgs = crop_imgs.view(-1, *crop_imgs.shape[2:])
      else:
        class_num = crop_imgs.shape[0]

      nc = crop_imgs.shape[0]//bs
      vision_encoder_input = BatchFeature(
        {'pixel_values': crop_imgs}, tensor_type='pt')
    elif isinstance(crop_imgs, List):
      nc = len(crop_imgs)//bs
      if isinstance(crop_imgs[0], Image.Image):
        tensor_list = [self.transform(img) for img in crop_imgs]

        nc = len(tensor_list)
        vision_encoder_input = torch.stack(tensor_list)
      else:
        raise ValueError("crop_imgs should be List[Image.Image] or torch.Tensor")
    else:
      raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")
    self.nc = nc

    vision_encoder_input = vision_encoder_input.to(self.vision_encoder.device)
    if isinstance(vision_encoder_input, dict) or isinstance(vision_encoder_input, BatchFeature):
      vision_model_output = self.vision_encoder(
        **vision_encoder_input,
        return_dict=True)
    else:
      vision_model_output = self.vision_encoder(
        vision_encoder_input,
        return_dict=True)

    result = self.MLP(vision_model_output['pooler_output'])

    self.template_feats = result.reshape(-1,
                                          nc,
                                          result.shape[-1])

    self.model[-1].nc = nc

class V2V_template_SigLIP_multi_scale_Model(V2V_template_SigLIP_Model):
  """V2V with multi scaling SigLIP patch. (Take multiple SigLIP patch's to do attention pooling)"""

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model
    
    self.want_layers = self.yaml['want_layers'] if (self.yaml['want_layers'] is not None) else [-1]
    LOGGER.info(f"Using layers: {self.want_layers} for multi scale attention pooling.")
    
    del self.MLP
    self.ATTN_POOLING = MultiLevelTemplateAttentionPooling(hidden_size=768, proj_size=512, num_patches=196, num_levels=len(self.want_layers))
  
  def set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor], bs=1):
    """Set classes in advance so that model could do offline-inference without clip model."""

    if isinstance(crop_imgs, torch.Tensor):
      if crop_imgs.dim() == 5:
        bs = crop_imgs.shape[0]
        class_num = crop_imgs.shape[1]
        crop_imgs = crop_imgs.view(-1, *crop_imgs.shape[2:])
      else:
        class_num = crop_imgs.shape[0]

      nc = crop_imgs.shape[0]//bs
      vision_encoder_input = BatchFeature(
        {'pixel_values': crop_imgs}, tensor_type='pt')
    elif isinstance(crop_imgs, List):
      nc = len(crop_imgs)//bs
      if isinstance(crop_imgs[0], Image.Image):
        tensor_list = [self.transform(img) for img in crop_imgs]

        nc = len(tensor_list)
        vision_encoder_input = torch.stack(tensor_list)
      else:
        raise ValueError("crop_imgs should be List[Image.Image] or torch.Tensor")
    else:
      raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")
    self.nc = nc

    # if (nc>256):
    #   for batch_idx in range(0, bs):
    #     batch_vision_encoder_input = vision_encoder_input['pixel_values'][nc*batch_idx:nc*(batch_idx+1)].to(self.vision_encoder.device)
    #     vision_model_output = self.vision_encoder(
    #       batch_vision_encoder_input,
    #       output_hidden_states=True,
    #       return_dict=True)

    #     temp_hidden_states = [vision_model_output['hidden_states'][h_idx] for h_idx in self.want_layers]
    #     temp_attn_pooling_result = self.ATTN_POOLING(temp_hidden_states)['pooled_feature_proj']
    #     if batch_idx == 0:
    #       attn_pooling_result = temp_attn_pooling_result
    #     else:
    #       attn_pooling_result = torch.cat([attn_pooling_result, temp_attn_pooling_result], dim=0)

    # else:
    vision_encoder_input = vision_encoder_input.to(self.vision_encoder.device)
    if isinstance(vision_encoder_input, dict) or isinstance(vision_encoder_input, BatchFeature):
      vision_model_output = self.vision_encoder(
        **vision_encoder_input,
        output_hidden_states=True,
        return_dict=True)
    else:
      vision_model_output = self.vision_encoder(
        vision_encoder_input,
        output_hidden_states=True,
        return_dict=True)

    hidden_states = [[] for _ in range(len(self.want_layers))]
    for idx, layer_num in enumerate(self.want_layers):
      hidden_states[idx] = vision_model_output['hidden_states'][layer_num]

    attn_pooling_result = self.ATTN_POOLING(hidden_states)['pooled_feature_proj']

    self.template_feats = attn_pooling_result.reshape(-1,
                                          nc,
                                          attn_pooling_result.shape[-1])

    self.model[-1].nc = nc

class V2V_template_SigLIPv2_Model(V2V_template_SigLIP_Model):
  """V2V with SigLIPv2 patch. (Take SigLIPv2's class token')"""

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""

    self.template_feats = torch.randn(1, nc or 80, 512)  # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model

    self.vision_processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path="google/siglip2-base-patch16-224")
    self.vision_encoder = SiglipVisionModel.from_pretrained(
        pretrained_model_name_or_path="google/siglip2-base-patch16-224") 

class V2V_template_SigLIPv2_multi_scale_Model(V2V_template_SigLIPv2_Model):
  """V2V with multi scaling SigLIP patch. (Take multiple SigLIP patch's to do attention pooling)"""

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model
    
    self.want_layers = self.yaml['want_layers'] if (self.yaml['want_layers'] is not None) else [-1]
    LOGGER.info(f"Using layers: {self.want_layers} for multi scale attention pooling.")
    
    del self.MLP
    self.ATTN_POOLING = MultiLevelTemplateAttentionPooling(hidden_size=768, proj_size=512, num_patches=196, num_levels=len(self.want_layers))

  def set_classes(self, crop_imgs: Union[List[Image.Image], torch.Tensor], bs=1):
    """Set classes in advance so that model could do offline-inference without clip model."""

    if isinstance(crop_imgs, torch.Tensor):
      if crop_imgs.dim() == 5:
        bs = crop_imgs.shape[0]
        class_num = crop_imgs.shape[1]
        crop_imgs = crop_imgs.view(-1, *crop_imgs.shape[2:])
      else:
        class_num = crop_imgs.shape[0]

      nc = crop_imgs.shape[0]//bs
      vision_encoder_input = BatchFeature(
        {'pixel_values': crop_imgs}, tensor_type='pt')
    elif isinstance(crop_imgs, List):
      nc = len(crop_imgs)//bs
      if isinstance(crop_imgs[0], Image.Image):
        tensor_list = [self.transform(img) for img in crop_imgs]

        nc = len(tensor_list)
        vision_encoder_input = torch.stack(tensor_list)
      else:
        raise ValueError("crop_imgs should be List[Image.Image] or torch.Tensor")
    else:
      raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")
    self.nc = nc

    vision_encoder_input = vision_encoder_input.to(self.vision_encoder.device)
    if isinstance(vision_encoder_input, dict) or isinstance(vision_encoder_input, BatchFeature):
      vision_model_output = self.vision_encoder(
        **vision_encoder_input,
        output_hidden_states=True,
        return_dict=True)
    else:
      vision_model_output = self.vision_encoder(
        vision_encoder_input,
        output_hidden_states=True,
        return_dict=True)

    hidden_states = [[] for _ in range(len(self.want_layers))]
    for idx, layer_num in enumerate(self.want_layers):
      hidden_states[idx] = vision_model_output['hidden_states'][layer_num]

    attn_pooling_result = self.ATTN_POOLING(hidden_states)['pooled_feature_proj']

    self.template_feats = attn_pooling_result.reshape(-1,
                                          nc,
                                          attn_pooling_result.shape[-1])

    self.model[-1].nc = nc

class V2V_template_SigLIP_multi_scale_multi_head_Model(V2V_template_SigLIP_multi_scale_Model):
  """V2V with multi scaling SigLIP patch, with multi-head template attention pooling. (Take multiple SigLIP patch's to do attention pooling)"""

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model

    del self.ATTN_POOLING
    self.ATTN_POOLING = MultiHeadTemplateAttentionPooling(hidden_size=768, proj_size=512, num_patches=196, num_levels=len(self.want_layers))

class V2V_With_MultiScale_SAVPE_Model(DetectionModel):
    """V2V with SAVPE detection model."""

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.want_layers = self.yaml['want_layers'] if (self.yaml['want_layers'] is not None) else [-1]
    
        LOGGER.info(f"Using layers: {self.want_layers} for multi scale.")
        
        self.image_processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path="google/siglip2-base-patch16-224")
        self.vision_encoder = SiglipVisionModel.from_pretrained(
        pretrained_model_name_or_path="google/siglip2-base-patch16-224") 
        self.vision_encoder_patch_size = 224
        self.patch_emb_savpe = PATCH_EMBEDDING_SAVPE()

        self.vpe = torch.zeros(1, 80, 512)  # features placeholder
        # self.origin_nc = self.nc
    
    def inference_set_classes(self, imgs, bboxes, class_names):
        from ultralytics.data.augment import LoadVisualPrompt
        
        device = next(self.vision_encoder.parameters()).device
        
        transform_vision_encoder = transforms.Compose([
            transforms.Resize((self.vision_encoder_patch_size, self.vision_encoder_patch_size), antialias=True),
            transforms.ToTensor()
        ])

        transform_visual_prompt = transforms.Compose([
            transforms.Resize((640, 640), antialias=True),
            transforms.ToTensor()
        ])
        
        imgs_list = []
        if isinstance(imgs, list):
            for img in imgs:
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                elif not isinstance(img, Image.Image):
                    raise ValueError("imgs should be List[Image.Image] or np.array")
                imgs_list.append(img)
        else:
            if isinstance(imgs, str):
                imgs_list = [Image.open(imgs).convert("RGB")]
            elif isinstance(imgs, np.ndarray):
                imgs_list = [Image.fromarray(imgs)]
            elif isinstance(imgs, Image.Image):
                imgs_list = [imgs]
            else:
                raise ValueError("imgs should be List[Image.Image] or np.array")
        
        template_img_origin_size = [img.size for img in imgs_list]
        img_tensor_list = [transform_vision_encoder(img) for img in imgs_list]
        batch_tensor = torch.stack(img_tensor_list)
        
        # For Vision Encoder
        vision_encoder_inputs = {'pixel_values': batch_tensor.to(device)}
        logging.info(f"Vision encoder input shape: {batch_tensor.shape}")
        
        # For Visual Prompt
        visual_prompt_list = []
        class_to_idx = {name: idx for idx, name in enumerate(set(class_names))}
        logging.info(f"Class to index mapping: {class_to_idx}")
        for img_idx, img in enumerate(imgs_list):
            class_idx = torch.tensor([class_to_idx[class_names[img_idx]]]).unsqueeze(0)
            
            if isinstance(bboxes, list):
                bbox = bboxes[img_idx]
            elif isinstance(bboxes, torch.Tensor):
                if bboxes.dim() == 2:
                    bbox = bboxes[img_idx].unsqueeze(0)
                elif bboxes.dim() == 1:
                    bbox = bboxes
                else:
                    raise ValueError("bboxes has some issues.")
            else:
                raise ValueError("bboxes should be List[Image.Image] or torch.tensor in 2D or 1D")
            
            origin_size = template_img_origin_size[img_idx]
            n_bbox = torch.zeros_like(bbox, dtype=torch.float)
            n_bbox[0][0] = bbox[0][0]/origin_size[1]
            n_bbox[0][1] = bbox[0][1]/origin_size[0]
            n_bbox[0][2] = bbox[0][2]/origin_size[1]
            n_bbox[0][3] = bbox[0][3]/origin_size[0]
            
            visual_prompt_label = {
                'img': transform_visual_prompt(img),
                'bboxes': torch.tensor(n_bbox),
                'cls': class_idx,
            }
            
            load_visual_prompt_func = LoadVisualPrompt()
            visual_prompt_label = load_visual_prompt_func(visual_prompt_label)
            visual_prompt_list.append(visual_prompt_label['visuals'])
        
        visuals_tensor = torch.stack(visual_prompt_list)
        logging.info(f"Visuals tensor shape: {visuals_tensor.shape}")

        if isinstance(self.vision_encoder, pe.VisionTransformer):
            vision_model_output = self.vision_encoder(vision_encoder_inputs['pixel_values'], output_hidden_list=self.want_layers, strip_cls_token=True)
            savpe_input = vision_model_output['hidden_states'].permute(1, 0, 2, 3)
            vpe = self.patch_emb_savpe(savpe_input, visuals_tensor)
        else:
            vision_model_output = self.vision_encoder(**vision_encoder_inputs, output_hidden_states=True, return_dict=True)
            hidden_states_list = [[] for _ in range(len(self.want_layers))]
            for idx, layer_num in enumerate(self.want_layers):
                hidden_states_list[idx] = vision_model_output['hidden_states'][layer_num]
            hidden_states_tensor = torch.stack(tensors=hidden_states_list, dim=1)
            vpe = self.patch_emb_savpe(hidden_states_tensor, visuals_tensor)
        
        # Correct dimension handling
        logging.info(f"Raw VPE shape: {vpe.shape}")
        # Squeeze leading dimension if present
        if vpe.size(0) == 1:
            vpe = vpe.squeeze(0)  # Shape: [num_images, embed_dim]
            logging.info(f"VPE shape after squeeze: {vpe.shape}")
        else:
            vpe = vpe.squeeze(0)  # Shape: [num_images, embed_dim]
            logging.info(f"VPE shape after squeeze: {vpe.shape}")
        
        # Ensure vpe has shape [num_images, num_patches, embed_dim]
        if vpe.dim() == 2:
            vpe = vpe.unsqueeze(1)  # Shape: [num_images, 1, embed_dim]
        logging.info(f"VPE shape after unsqueeze: {vpe.shape}")
        
        batch_size = vpe.size(0)
        num_patches = vpe.size(1)
        logging.info(f"Batch size: {batch_size}, Num patches: {num_patches}")
        
        # Group embeddings by class and compute mean
        class_indices = [class_to_idx[name] for name in class_names]
        num_classes = len(class_to_idx)
        class_embeddings = []
        
        for cls_idx in range(num_classes):
            cls_mask = torch.tensor([i == cls_idx for i in class_indices], device=vpe.device)
            cls_name = list(class_to_idx.keys())[cls_idx]
            logging.info(f"Class {cls_idx} ({cls_name}): {cls_mask.sum()} images")
            
            if cls_mask.sum() == 0:
                logging.warning(f"No valid images for class index {cls_idx} ({cls_name})")
                class_embeddings.append(torch.zeros(1, num_patches, vpe.size(2), device=vpe.device))
                continue
            
            # Select embeddings for this class
            cls_vpe = vpe[cls_mask]  # Shape: [num_images_for_class, num_patches, embed_dim]
            logging.info(f"cls_vpe shape for class {cls_idx}: {cls_vpe.shape}")
            
            # Compute mean across images for this class
            cls_mean_vpe = cls_vpe.mean(dim=0, keepdim=True)  # Shape: [1, num_patches, embed_dim]
            logging.info(f"cls_mean_vpe shape for class {cls_idx}: {cls_mean_vpe.shape}")
            class_embeddings.append(cls_mean_vpe)
        
        if not class_embeddings:
            raise ValueError("No valid class embeddings computed")
        
        # Stack mean embeddings
        self.vpe = torch.cat(class_embeddings, dim=0)  # Shape: [num_classes, num_patches, embed_dim]
        self.nc = self.vpe.shape[0]
        self.model[-1].nc = self.nc
        logging.info(f"Final number of classes (self.nc): {self.nc}, VPE final shape: {self.vpe.shape}")
        # import pdb; pdb.set_trace()
        self.vpe = self.vpe.permute(1, 0, 2)
        # Validate self.nc against expected model configuration
        # num_anchors = getattr(self.model[-1], 'na', 3)  # Assume 3 anchors if na not set
        # expected_no = self.nc * (5 + num_anchors)
        # logging.info(f"Expected no (nc * (5 + num_anchors)): {expected_no}")
        # if expected_no != getattr(self.model[-1], 'no', expected_no):
        #     logging.error(f"Mismatch in detection head: expected no={expected_no}, but model.no={self.model[-1].no}")
        #     raise ValueError(f"Detection head mismatch: expected no={expected_no}, got {self.model[-1].no}")



    # def inference_set_classes(self, imgs, bboxes):
      
    #   from ultralytics.data.augment import LoadVisualPrompt
      
    #   device = next(self.vision_encoder.parameters()).device
      
    #   transform_vision_encoder = transforms.Compose([
    #       transforms.Resize((self.vision_encoder_patch_size, self.vision_encoder_patch_size), antialias=True),
    #       transforms.ToTensor()
    #   ])

    #   transform_visual_prompt = transforms.Compose([
    #       transforms.Resize((640, 640), antialias=True),
    #       transforms.ToTensor()
    #   ])
      
    #   imgs_list = []
    #   if isinstance(imgs, list):
    #     for img in imgs:
    #       if isinstance(img, str):
    #         img = Image.open(img).convert("RGB")
    #       elif isinstance(img, np.ndarray):
    #         img = Image.fromarray(img)
    #       elif not isinstance(img, Image.Image):
    #         raise ValueError("imgs should be List[Image.Image] or np.array")
    #       imgs_list.append(img)
    #   else:
    #     if isinstance(imgs, str):
    #       imgs_list = [Image.open(imgs).convert("RGB")]
    #     elif isinstance(imgs, np.ndarray):
    #       imgs_list = [Image.fromarray(imgs)]
    #     elif isinstance(imgs, Image.Image):
    #       imgs_list = [imgs]
    #     else:
    #       raise ValueError("imgs should be List[Image.Image] or np.array")
      
      
    #   template_img_origin_size = [img.size for img in imgs_list]
    #   img_tensor_list = [transform_vision_encoder(img) for img in imgs_list]
    #   batch_tensor = torch.stack(img_tensor_list)
      
    #   # For Vision Encoder
    #   vision_encoder_inputs = {'pixel_values': batch_tensor.to(device)}
      
    #   # For Visual Prompt
    #   visual_prompt_list = []
    #   for img_idx, img in enumerate(imgs_list):
        
    #     class_idx = torch.tensor([img_idx]).unsqueeze(0)
        
    #     if isinstance(bboxes, list):
    #       bbox = bboxes[img_idx] 
    #     elif isinstance(bboxes, torch.Tensor):
    #       if bboxes.dim() == 2:
    #         bbox = bboxes[img_idx].unsqueeze(0)
    #       elif bboxes.dim() == 1:
    #         bbox = bboxes
    #       else:
    #         raise ValueError("bboxes has some issues.")
    #     else:
    #       raise ValueError("bboxes should be List[Image.Image] or torch.tensor in 2D or 1D")
        
    #     # bbox = bboxes[img_idx] if isinstance(bboxes, list) else bboxes   
        
    #     origin_size = template_img_origin_size[img_idx]
    #     # n_bbox = deepcopy(bbox)
    #     n_bbox = torch.zeros_like(bbox, dtype=torch.float)
    #     n_bbox[0][0] = bbox[0][0]/origin_size[1]
    #     n_bbox[0][1] = bbox[0][1]/origin_size[0]
    #     n_bbox[0][2] = bbox[0][2]/origin_size[1]
    #     n_bbox[0][3] = bbox[0][3]/origin_size[0]
        
    #     visual_prompt_label = {
    #         'img': transform_visual_prompt(img),
    #         'bboxes': torch.tensor(n_bbox),
    #         'cls': torch.tensor([img_idx]).unsqueeze(0),
    #       } 
        
        
    #     # visual_prompt_label = {
    #     #     'img': transform_visual_prompt(img),
    #     #     'bboxes': torch.tensor(bboxes).unsqueeze(0),
    #     #     'cls': torch.arange(0, end=len(imgs), dtype=torch.int8).reshape(-1, 1),
    #     #   } 
    #     load_visual_prompt_func = LoadVisualPrompt()
    #     visual_prompt_label = load_visual_prompt_func(visual_prompt_label)
    #     visual_prompt_list.append(visual_prompt_label['visuals'])
      
    #   visuals_tensor = torch.stack(visual_prompt_list)

    #   if isinstance(self.vision_encoder, pe.VisionTransformer):
        
    #     vision_model_output = self.vision_encoder(vision_encoder_inputs['pixel_values'], output_hidden_list=self.want_layers, strip_cls_token=True)
    #     savpe_input = vision_model_output['hidden_states'].permute(1, 0, 2, 3)
    #     self.vpe = self.patch_emb_savpe(savpe_input, visuals_tensor)
        
    #   else:
    #     vision_model_output = self.vision_encoder(**vision_encoder_inputs, output_hidden_states=True,      return_dict=True)

    #     hidden_states_list = [[] for _ in range(len(self.want_layers))]
    #     for idx, layer_num in enumerate(self.want_layers):
    #       hidden_states_list[idx] = vision_model_output['hidden_states'][layer_num]
      
    #     hidden_states_tensor = torch.stack(tensors=hidden_states_list, dim=1)

    #     self.vpe = self.patch_emb_savpe(hidden_states_tensor, visuals_tensor)
      
    #   self.vpe = self.vpe.permute(1, 0, 2)
    #   self.nc = self.vpe.shape[1]
    #   self.model[-1].nc = self.nc

    def set_classes(self, vpe: torch.Tensor=None, nc=80, imgs=None, batch_cls=None, bboxes=None, batch_idx=None, visuals=None):
      """Set classes in advance so that model could do offline-inference"""
      
      device = self.vision_encoder.device
      # ---- 
      # If vpe (feature of vp and attention with imgs) is given, then use it.
      if vpe is not None:
        self.vpe = vpe
        self.model[-1].nc = vpe.shape[1]
        return
      # ---- 
      
      # ---- 
      # Process image's shape and datatype transform
      if isinstance(imgs, torch.Tensor):
        if imgs.dim() == 5:
          imgs = imgs.view(-1, *imgs.shape[2:])

        vision_encoder_input = imgs

      elif isinstance(imgs, List):
        if isinstance(imgs[0], Image.Image):
          tensor_list = [self.transform(img) for img in imgs]
          vision_encoder_input = torch.stack(tensor_list)
        else:
          raise ValueError("crop_imgs should be List[Image.Image] or torch.Tensor")
      else:
        raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")

      vision_encoder_input = vision_encoder_input.to(device)

      # Process the image through the vision encoder
      # inputs = self.image_processor(vision_encoder_input, return_tensors="pt")
      transform = transforms.Compose([
          transforms.Resize((self.vision_encoder_patch_size, self.vision_encoder_patch_size), antialias=True)
      ])
      inputs = {'pixel_values': transform(vision_encoder_input).to(device)}


      vp_list = []
      for img_idx in range(imgs.shape[0]):
          v = deepcopy(visuals[img_idx]).to(device)
          
          # 找出屬於當前圖像的所有匹配項索引
          matches = (batch_idx == img_idx).nonzero()
          vp_single = torch.zeros(nc, v.shape[-2], v.shape[-1], device=device)  # 在相同設備上創建
          
          # 只有在有匹配項時處理
          if len(matches) > 0:
              # 獲取批次中當前圖像的開始位置和數量
              batch_start = matches[0].item()
              batch_count = len(matches)
              

              img_classes = deepcopy(batch_cls[batch_start:batch_start + batch_count]).to(torch.int32)
              # img_boxes = deepcopy(bboxes[batch_start:batch_start + batch_count])
              
              # get unique classes
              img_classes_unique = torch.unique(img_classes, sorted=True)
              
              # 將視覺特徵分配給相應的類別 
              if v.shape[0] > 0:  # 只在有視覺特徵時處理
                  if len(img_classes_unique) != len(v):
                      vp_single[img_classes_unique] = v[:len(img_classes_unique)]
                  else:
                      vp_single[img_classes_unique] = v
          
          vp_list.append(vp_single)
      
      if vp_list:
          visuals_tensor = torch.stack(tensors=vp_list, dim=0).to(device)
      else:
          # If vp_list is empty, create a tensor of zeros
          visuals_tensor = torch.zeros((0, nc, visuals.shape[-2], visuals.shape[-1]),  device=device)

      # ----

      vision_model_output = self.vision_encoder(**inputs, 
                                    output_hidden_states=True,
                                    return_dict=True)
      
      hidden_states_list = [[] for _ in range(len(self.want_layers))]
      for idx, layer_num in enumerate(self.want_layers):
        hidden_states_list[idx] = vision_model_output['hidden_states'][layer_num]
      
      hidden_states_tensor = torch.stack(tensors=hidden_states_list, dim=1)
      # ----
      
      self.vpe = self.patch_emb_savpe(hidden_states_tensor, visuals_tensor)
      
      self.nc = nc
      self.model[-1].nc = nc

      return self.vpe

    def get_cls_pe(self, vpe):
        """
        Get class positional embeddings.

        Args:
            vpe (torch.Tensor, optional): Visual positional embeddings.

        Returns:
            (torch.Tensor): Class positional embeddings.
        """
        all_pe = []
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(
        self, x, batch=None, profile=False, visualize=False, augment=False, embed=None, vpe=None
    ):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.
            vpe (torch.Tensor, optional): Visual positional embeddings.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        if batch is not None and isinstance(batch, dict) and "nc" in batch:
          nc = batch['nc']
        elif self.training: 
          nc = self.yaml['nc']
        else: 
          nc = self.nc
        
        self.nc = nc
        self.model[-1].nc = nc

        # if batch is not None:
        #   # Check if batch is a dictionary, and having all the components we need.
        #   if isinstance(batch, dict):
        #     if "img" in batch and "visuals" in batch:
        #   else:
        #     raise ValueError("batch should be a dictionary containing 'img' and 'visuals' keys.")
        #   self.set_classes(nc=nc, template_imgs=x, vp=visuals_labels)
        
        if batch is not None:
            if not isinstance(batch, dict):
              raise ValueError("batch should be a dictionary.")
            self.set_classes(nc=nc, imgs=x, batch_cls=batch['cls'], bboxes=batch['bboxes'], batch_idx=batch['batch_idx'], visuals=batch['visuals'])
            
        if vpe is None:
            if hasattr(self, "vpe"):
                vpe = self.vpe
            else:
                vpe = torch.zeros(1, 80, 512)  # features placeholder
        y, dt, embeddings = [], [], []  # outputs
        b = x.shape[0]
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, WorldDetect) or isinstance(m, YOLOEDetect):
                cls_pe = self.get_cls_pe(vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x = m(x, cls_pe)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
      """
      Compute loss.

      Args:
          batch (dict): Batch to compute loss on.
          preds (torch.Tensor | List[torch.Tensor]): Predictions.
      """
      
      if not hasattr(self, "criterion"):
        self.criterion = self.init_criterion()

      if preds is None: 
        preds = self.forward(batch["img"], batch)
      return self.criterion(preds, batch)

    # def loss(self, batch, preds=None):
    #     """
    #     Compute loss.

    #     Args:
    #         batch (dict): Batch to compute loss on.
    #         preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
    #     """
    #     if not hasattr(self, "criterion"):
    #         from ultralytics.utils.loss import TVPDetectLoss

    #         visual_prompt = batch.get("visuals", None) is not None  # TODO
    #         self.criterion = TVPDetectLoss(self) if visual_prompt else self.init_criterion()

    #     if preds is None:
    #         preds = self.forward(batch["img"], vpe=batch.get("template_feats", None))
    #     return self.criterion(preds, batch)

class V2V_With_MultiScale_SAVPE_SigLIP2_B_Model(V2V_With_MultiScale_SAVPE_Model):
    """V2V with SAVPE detection model."""

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.want_layers = self.yaml['want_layers'] if (self.yaml['want_layers'] is not None) else [-1]
    
        LOGGER.info(f"Using layers: {self.want_layers} for multi scale.")
        
        self.image_processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path="google/siglip2-base-patch16-224")
        self.vision_encoder = SiglipVisionModel.from_pretrained(
        pretrained_model_name_or_path="google/siglip2-base-patch16-224") 
        self.patch_emb_savpe = PATCH_EMBEDDING_SAVPE()
        self.vision_encoder_patch_size = 224

        self.vpe = torch.zeros(1, 80, 512)  # features placeholder
        # self.origin_nc = self.nc
        
class V2V_With_MultiScale_SAVPE_SigLIP2_L_Model(V2V_With_MultiScale_SAVPE_Model):
    """V2V with SAVPE detection model."""

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.want_layers = self.yaml['want_layers'] if (self.yaml['want_layers'] is not None) else [-1]
    
        LOGGER.info(f"Using layers: {self.want_layers} for multi scale.")
        
        self.image_processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path="google/siglip2-large-patch16-256")
        self.vision_encoder = SiglipVisionModel.from_pretrained(
        pretrained_model_name_or_path="google/siglip2-large-patch16-256") 
        self.patch_emb_savpe = PATCH_EMBEDDING_SAVPE(embed_dim=1024)
        self.vision_encoder_patch_size = 256

        self.vpe = torch.zeros(1, 80, 512)  # features placeholder
        # self.origin_nc = self.nc
class V2V_With_MultiScale_SAVPE_PE_B16_Model(V2V_With_MultiScale_SAVPE_Model):
    """V2V with SAVPE detection model, with perception_models. (a multi-modaility model proposed by FAIR, )"""

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """

        import v2vdet.v2vdet_ultralytics.perception_models.core.vision_encoder.transforms as pe_transforms
        
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.want_layers = self.yaml['want_layers'] if (self.yaml['want_layers'] is not None) else [-1]
    
        LOGGER.info(f"Using layers: {self.want_layers} for multi scale.")
        
        self.vision_encoder = pe.VisionTransformer.from_config(name="PE-Core-B16-224", pretrained=True)
        # self.image_processor = pe_transforms.get_image_transform(self.vision_encoder.image_size)
        
        self.patch_emb_savpe = PATCH_EMBEDDING_SAVPE(embed_dim=768)

        self.vpe = torch.zeros(1, 80, 512)  # features placeholder
        # self.origin_nc = self.nc
        self.vision_encoder_input_size = 224

    def set_classes(self, vpe: torch.Tensor=None, nc=80, imgs=None, batch_cls=None, bboxes=None, batch_idx=None, visuals=None):
      """Set classes in advance so that model could do offline-inference"""
      
      device = next(self.vision_encoder.parameters()).device
      
      # ---- 
      # If vpe (feature of vp and attention with imgs) is given, then use it.
      if vpe is not None:
        self.vpe = vpe
        self.model[-1].nc = vpe.shape[1]
        return
      # ---- 
      
      # ---- 
      # Process image's shape and datatype transform
      if isinstance(imgs, torch.Tensor):
        if imgs.dim() == 5:
          imgs = imgs.view(-1, *imgs.shape[2:])

        vision_encoder_input = imgs

      elif isinstance(imgs, List):
        if isinstance(imgs[0], Image.Image):
          tensor_list = [self.transform(img) for img in imgs]
          vision_encoder_input = torch.stack(tensor_list)
        else:
          raise ValueError("crop_imgs should be List[Image.Image] or torch.Tensor")
      else:
        raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")
      # self.nc = nc
      # self.model[-1].nc = nc
      # ---- 

      # ----
      # Process the vp to each class
      # vp_list = []
      # for img_idx in range(imgs.shape[0]):
      # # Want to get the range of the class
      #   v = deepcopy(visuals[img_idx]).to('cpu')
      #   matches = (batch_idx == img_idx).nonzero()
      #   vp_single = torch.zeros(nc, v.shape[-2], v.shape[-1])
      #   if len(matches) > 0:
      #     batch_start = (batch_idx == img_idx).nonzero()[0].item()
      #     batch_count = (batch_idx == img_idx).sum().item()
      #     img_classes = deepcopy(batch_cls[batch_start:batch_start + batch_count]).to('cpu').to(torch.int32)
      #     img_boxes = deepcopy(bboxes[batch_start:batch_start + batch_count]).to('cpu')
      #     img_classes_unique, img_classes_lis = torch.unique(img_classes, sorted=True, return_inverse=True)

      #     if visuals[img_idx].shape[0] == 0:
      #       pass
      #     else:
      #       if len(img_classes_unique) != len(v):
      #         vp_single[img_classes_unique] = v[:len(img_classes_unique)]
      #       else:
      #         vp_single[img_classes_unique] = v
        
      #   vp_list.append(vp_single)
      
      # visuals_tensor = torch.stack(tensors=vp_list, dim=0).to(self.vision_encoder.device)
    
      # if vp_list:
      #   visuals_tensor = torch.stack(tensors=vp_list, dim=0).to(self.vision_encoder.device)
      # else:
      #     # If vp_list is empty, create a tensor of zeros
      #     visuals_tensor = torch.zeros((0, nc, visuals.shape[-2], visuals.shape[-1]),  device=self.vision_encoder.device)
      # ----

      vp_list = []
      for img_idx in range(imgs.shape[0]):
          # 獲取當前圖像的視覺特徵副本，不移到 CPU
          v = deepcopy(visuals[img_idx]).to(device)
          
          # 找出屬於當前圖像的所有匹配項索引
          matches = (batch_idx == img_idx).nonzero()
          vp_single = torch.zeros(nc, v.shape[-2], v.shape[-1], device=device)  # 在相同設備上創建
          
          # 只有在有匹配項時處理
          if len(matches) > 0:
              # 獲取批次中當前圖像的開始位置和數量
              batch_start = matches[0].item()
              batch_count = len(matches)
              
              # 獲取當前圖像的類別和邊界框，不移到 CPU
              img_classes = deepcopy(batch_cls[batch_start:batch_start + batch_count]).to(torch.int32)
              # img_boxes = deepcopy(bboxes[batch_start:batch_start + batch_count])  # 注意：這行沒被使用
              
              # 獲取唯一類別
              img_classes_unique = torch.unique(img_classes, sorted=True)
              
              # 將視覺特徵分配給相應的類別
              if v.shape[0] > 0:  # 只在有視覺特徵時處理
                  if len(img_classes_unique) != len(v):
                      vp_single[img_classes_unique] = v[:len(img_classes_unique)]
                  else:
                      vp_single[img_classes_unique] = v
            
          vp_list.append(vp_single)

      # 將處理後的視覺特徵堆疊起來
      if vp_list:
          visuals_tensor = torch.stack(vp_list, dim=0).to(device=device)
      else:
          # 如果 vp_list 為空，創建零張量
          visuals_tensor = torch.zeros((0, nc, visuals.shape[-2], visuals.shape[-1]), device=device)
      
      # ----
      # Process the image through the vision encoder
      # inputs = self.image_processor(vision_encoder_input).unsqueeze(0).to(self.vision_encoder.device)
      
      transform = transforms.Compose([
          transforms.Resize((self.vision_encoder_input_size, self.vision_encoder_input_size), antialias=True)
      ])
      inputs = transform(vision_encoder_input).to(device)
      

      vision_model_output = self.vision_encoder(inputs,
                                                output_hidden_list=self.want_layers,
                                                strip_cls_token=True)
      # inputs = inputs.to(self.vision_encoder.device)
      # vision_model_output = self.vision_encoder(**inputs, 
      #                               output_hidden_states=True,
      #                               return_dict=True)

      # hidden_states_list = vision_model_output['hidden_states']
     
      # for idx, layer_num in enumerate(self.want_layers):
      #   hidden_states_list[idx] = vision_model_output['hidden_states'][layer_num]
      
      # hidden_states_tensor = torch.stack(tensors=hidden_states_list, dim=1)
      # ----
      
      self.vpe = self.patch_emb_savpe(vision_model_output['hidden_states'].permute(1, 0, 2, 3), visuals_tensor)
      
      self.nc = nc
      self.model[-1].nc = nc

      return self.vpe

class V2V_With_MultiScale_SAVPE_PE_L14_Model(V2V_With_MultiScale_SAVPE_PE_B16_Model):
    """V2V with SAVPE detection model, with perception_models. (a multi-modaility model proposed by FAIR, )"""

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.want_layers = self.yaml['want_layers'] if (self.yaml['want_layers'] is not None) else [-1]
    
        LOGGER.info(f"Using layers: {self.want_layers} for multi scale.")
        
        self.vision_encoder = pe.VisionTransformer.from_config(name="PE-Core-L14-336", pretrained=True)
        # self.image_processor = pe_transforms.get_image_transform(self.vision_encoder.image_size)
        
        self.patch_emb_savpe = PATCH_EMBEDDING_SAVPE(embed_dim=1024)

        self.vpe = torch.zeros(1, 80, 512)  # features placeholder
        # self.origin_nc = self.nc
        self.vision_encoder_input_size = 336

# class V2VSegModel(V2V_With_MultiScale_SAVPE_Model, SegmentationModel):
#     """V2V segmentation model."""

#     def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
#         """
#         Initialize YOLOE segmentation model with given config and parameters.

#         Args:
#             cfg (str | dict): Model configuration file path or dictionary.
#             ch (int): Number of input channels.
#             nc (int, optional): Number of classes.
#             verbose (bool): Whether to display model information.
#         """
#         super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

#     def loss(self, batch, preds=None):
#         """
#         Compute loss.

#         Args:
#             batch (dict): Batch to compute loss on.
#             preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
#         """
#         if not hasattr(self, "criterion"):
#             from ultralytics.utils.loss import TVPSegmentLoss

#             visual_prompt = batch.get("visuals", None) is not None  # TODO
#             self.criterion = TVPSegmentLoss(self) if visual_prompt else self.init_criterion()

#         if preds is None:
#             preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
#         return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Generate the YOLO network's final layer.

        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.

        Returns:
            (tuple): Tuple containing the concatenated predictions and None.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output
  

# -------------------------------------------------
# Function

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        (tuple): Tuple containing the PyTorch model and sorted list of output layers.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, ImagePoolingAttn, v10Detect}
        ):
            args.append([ch[x] for x in f])
            if m is Segment or m is YOLOESegment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)

def load_state_dict_layer_by_layer(current_model, checkpoint):
  current_state = current_model.state_dict()
  checkpoint_state = checkpoint.state_dict()

  processed_layers = {}

  for key in checkpoint_state.keys():
    try:
      checkpoint_param = checkpoint_state[key]
      current_param = current_state[key]

      # 檢查數據類型和形狀
      if checkpoint_param.dtype != current_param.dtype or checkpoint_param.shape != current_param.shape:
        # print(f"\nProcessing layer: {key}")
        # print(f"Current model: {current_param.shape}, {current_param.dtype}")
        # print(f"Checkpoint: {checkpoint_param.shape}, {checkpoint_param.dtype}")

        # 處理形狀不匹配
        if checkpoint_param.shape != current_param.shape:
          if checkpoint_param.numel() == 1 and current_param.numel() == 1:
            '''
            for example:
            layer: model.23.cv4.0.bias
              shape in model: torch.Size([1]), value: tensor([-10.])
              shape in ckpt: torch.Size([]), value: tensor(-22.6406, dtype=torch.float16)
            '''
            checkpoint_param = checkpoint_param.reshape(
                current_param.shape)

        # 處理數據類型不匹配
        checkpoint_param = checkpoint_param.to(
            dtype=current_param.dtype)

        # print(f"After processing: {checkpoint_param.shape}, {checkpoint_param.dtype}")

      processed_layers[key] = checkpoint_param

    except Exception as e:
      print(f"\nError processing layer {key}: {str(e)}")
      continue

  # 更新模型
  current_model.load_state_dict(processed_layers)
  return processed_layers
