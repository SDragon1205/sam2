# Ultralytics Eric YOLO 🚀, AGPL-3.0 license

from v2vdet.v2vdet_ultralytics.utils.loss import (v2vDetectionLoss,
                                           v2v_E2EDetectLoss)
from v2vdet.v2vdet_ultralytics.utils.misc import load_images
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
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
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
    v10Detect,
)
from v2vdet.v2vdet_ultralytics.nn.modules import (
    C2fAttn,
    ImagePoolingAttn,
    WorldDetect,
    C2f_v2v_Attn,
    A2C2f_Template_MaxSigmoidAttn,
    TemplateAttentionPooling,
    MultiLevelTemplateAttentionPooling
)
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from transformers import AutoImageProcessor, Dinov2Model
from transformers import (CLIPProcessor, CLIPVisionModel, CLIPImageProcessor, CLIPVisionModelWithProjection, BatchFeature)
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as transforms
from torch.utils.checkpoint import checkpoint
from typing import List, Union
from PIL import Image
import torch.nn as nn
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
import types
import re
import pickle
import contextlib
import os
import sys
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
  import thop
except ImportError:
  thop = None


class DetectionModel(BaseModel):
  """YOLOv8 detection model."""

  # model, input channels, number of classes
  def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
    """Initialize the YOLOv8 detection model with the given config and parameters."""
    super().__init__()
    self.yaml = cfg if isinstance(
        cfg, dict) else yaml_model_load(cfg)  # cfg dict
    if self.yaml["backbone"][0][2] == "Silence":
      LOGGER.warning(
          "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of nn.Identity. "
          "Please delete local *.pt file and re-download the latest model checkpoint."
      )
      self.yaml["backbone"][0][2] = "nn.Identity"

    # Define model
    self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
    ch = self.yaml["ch"]
    if nc and nc != self.yaml["nc"]:
      ''''
          this nc value can be adjusted from 'v2vdet_ultralytics/models/v2vdet/train.py' file
              model = WorldModel(
              cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
              ch=3,
              nc=min(self.data["nc"], 80),
              verbose=verbose and RANK == -1,
              )
          just change the nc value (current is 80) to your desired value.
      '''
      LOGGER.info(
          f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
      self.yaml["nc"] = nc  # override YAML value

    self.model, self.save = parse_model(
        deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
    self.names = {i: f"{i}" for i in range(
        self.yaml["nc"])}  # default names dict
    self.inplace = self.yaml.get("inplace", True)
    self.end2end = getattr(self.model[-1], "end2end", False)
    # Build strides
    m = self.model[-1]  # Detect()
    # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
    if isinstance(m, Detect):
      s = 256  # 2x min stride
      m.inplace = self.inplace

      def _forward(x):
        """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
        if self.end2end:
          return self.forward(x)["one2many"]

        return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

      m.stride = torch.tensor(
          [s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
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
    """Perform augmentations on input image x and return augmented inference and train outputs."""
    if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
      LOGGER.warning(
          "WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
      return self._predict_once(x)
    img_size = x.shape[-2:]  # height, width
    s = [1, 0.83, 0.67]  # scales
    f = [None, 3, None]  # flips (2-ud, 3-lr)
    y = []  # outputs
    for si, fi in zip(s, f):
      xi = scale_img(x.flip(fi) if fi else x, si,
                     gs=int(self.stride.max()))
      yi = super().predict(xi)[0]  # forward
      yi = self._descale_pred(yi, fi, si, img_size)
      y.append(yi)
    y = self._clip_augmented(y)  # clip augmented tails
    return torch.cat(y, -1), None  # augmented inference, train

  @staticmethod
  def _descale_pred(p, flips, scale, img_size, dim=1):
    """De-scale predictions following augmented inference (inverse operation)."""
    p[:, :4] /= scale  # de-scale
    x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
    if flips == 2:
      y = img_size[0] - y  # de-flip ud
    elif flips == 3:
      x = img_size[1] - x  # de-flip lr
    return torch.cat((x, y, wh, cls), dim)

  def _clip_augmented(self, y):
    """Clip YOLO augmented inference tails."""
    nl = self.model[-1].nl  # number of detection layers (P3-P5)
    g = sum(4**x for x in range(nl))  # grid points
    e = 1  # exclude layer count
    i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
    y[0] = y[0][..., :-i]  # large
    i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x)
                                     for x in range(e))  # indices
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

    # if isinstance(crop_imgs, torch.Tensor):

    # elif isinstance(crop_imgs, List):
    #   if not isinstance(crop_imgs[0], Image.Image):
    #     raise ValueError("crop_imgs should be List[Image.Image] or torch.Tensor")
    #   else:
    #     clip_input = self.vision_processor.preprocess(
    #     crop_imgs, return_tensors="pt")

    # else:
    #   raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")

    if (nc>256):
      vision_model_output_batch = []
      for batch_idx in range(0, bs):
        batch_clip_input = clip_input['pixel_values'][nc*batch_idx:nc*(batch_idx+1)]
        batch_clip_input = batch_clip_input.to(self.vision_encoder.device)
        vision_model_output_batch.append(self.vision_encoder(
          batch_clip_input,
          output_hidden_states=False,
          return_dict=True)['image_embeds'])

      vision_model_output = torch.cat(vision_model_output_batch, dim=0)
      # vision_model_output = vision_model_output.reshape(bs, nc, vision_model_output.shape[-1])

    else:
      clip_input = clip_input.to(self.vision_encoder.device)
      vision_model_output = self.vision_encoder(
        **clip_input,
        output_hidden_states=False,
        return_dict=True)['image_embeds']

    cls_token = deepcopy(vision_model_output)
    crop_img_feats = cls_token / cls_token.norm(dim=-1, keepdim=True)
    self.template_feats = crop_img_feats.reshape(bs,
                                            nc, crop_img_feats.shape[-1])
    self.model[-1].nc = nc

  # def dataloader_set_classes(self, crop_img_list_tensor, bs):

  #   if not hasattr(self, 'nc'):
  #     self.nc = 80
  #   # nc = self.model[-1].nc = self.nc
  #   crop_img_list_tensor = crop_img_list_tensor.to('cpu')
  #   nc = crop_img_list_tensor.shape[0]//bs

  #   if (crop_img_list_tensor is None):
  #     crop_img_list_tensor = torch.randn(1, 3, 224, 224)
  #   dataset = torch.utils.data.TensorDataset(crop_img_list_tensor)
  #   dataloader = torch.utils.data.DataLoader(
  #     dataset,
  #     batch_size=bs*8,
  #     pin_memory=True,
  #     num_workers=4
  #   )

  #   if torch.cuda.is_available():
  #     device = "cuda"
  #   # elif torch.backends.mps.is_available():
  #   #   device = "mps"
  #   else:
  #     device = "cpu"

  #   all_cls_tokens = []
  #   all_patches = []

  #   with torch.autocast(device_type=device, dtype=torch.bfloat16):
  #     for batch in dataloader:
  #       batch[0] = batch[0].to('cuda')
  #       vision_model_output = self.vision_encoder(batch[0], output_hidden_states=False, return_dict=True)
  #       cls_token = vision_model_output['image_embeds']
  #       all_cls_tokens.append(cls_token)
  #       # all_patches.append(hidden_states)

  #   cls_token = torch.cat([cls.to("cpu") for cls in all_cls_tokens], dim=0)
  #   crop_img_feats = cls_token / cls_token.norm(p=2, dim=-1, keepdim=True)
  #   self.txt_feats = crop_img_feats.reshape(-1, nc, crop_img_feats.shape[-1])
  #   del all_cls_tokens, dataset, dataloader
  #   self.model[-1].nc = nc

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
    self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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

    self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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
    with torch.inference_mode():
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
  """V2V with multi scaling clip patch. (Take multiple CLIP patch's to do attention pooling)"""

  def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
    """Initialize YOLOv8 world model with given config and parameters."""

    self.template_feats = torch.randn(1, nc or 80, 512)  # features placeholder
    super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model

    self.vision_processor = CLIPImageProcessor().from_pretrained(
        pretrained_model_name_or_path="openai/clip-vit-base-patch32")
    self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path="openai/clip-vit-base-patch32")  # CLIP model placeholder
    self.multi_scale_attn_pooling = MultiLevelTemplateAttentionPooling(hidden_size=768, proj_size=512, num_levels=4)

    self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        tensor_list = [transform(img) for img in crop_imgs]

        nc = len(tensor_list)

        # clip_input = self.vision_processor.preprocess(
        # crop_imgs, return_tensors="pt")

        clip_input = torch.stack(tensor_list)
      else:
        raise ValueError("crop_imgs should be List[Image.Image] or torch.Tensor")
    else:
      raise NotImplementedError("crop_img_list_tensor should be either List[Image.Image] or torch.Tensor")
      # nc = self.model[-1].nc = self.yaml['nc']

    self.nc = nc

    want_layer = [-2, -4, -6, -8]

    if (nc>256):
      for batch_idx in range(0, bs):
        batch_clip_input = clip_input['pixel_values'][nc*batch_idx:nc*(batch_idx+1)]
        batch_clip_input = batch_clip_input.to(self.vision_encoder.device)
        vision_model_output = self.vision_encoder(
          batch_clip_input,
          output_hidden_states=True,
          return_dict=True)

        temp_hidden_states = [vision_model_output['hidden_states'][h_idx] for h_idx in want_layer]
        temp_attn_pooling_result = self.multi_scale_attn_pooling(temp_hidden_states)['pooled_feature_proj']
        if batch_idx == 0:
          attn_pooling_result = temp_attn_pooling_result
        else:
          attn_pooling_result = torch.cat([attn_pooling_result, temp_attn_pooling_result], dim=0)

    else:
      clip_input = clip_input.to(self.vision_encoder.device)
      if isinstance(clip_input, dict) or isinstance(clip_input, BatchFeature):
        vision_model_output = self.vision_encoder(
          **clip_input,
          output_hidden_states=True,
          return_dict=True)
      else:
        vision_model_output = self.vision_encoder(
          clip_input,
          output_hidden_states=True,
          return_dict=True)

      hidden_states = [[] for _ in range(len(want_layer))]
      for idx, layer_num in enumerate(want_layer):
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

    if (nc>256):
      step = 4
      for batch_idx in range(0, bs, step):
        batch_clip_input = crop_imgs[nc*batch_idx:nc*(batch_idx+step)]
        batch_clip_input = batch_clip_input.to(self.device)
        temp_template_feats = self._vision_encoder_forward(batch_clip_input, embed=[9], return_tensor=True)
        if batch_idx == 0:
          template_feats = temp_template_feats
        else:
          template_feats = torch.cat([template_feats, temp_template_feats], dim=0)
    else:
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

    self.template_backbone_model = YOLO("yolov8m-world.pt").model.model[:10]
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
    self.DINO_linear_layer = nn.Linear(in_features=768, out_features=512)
    # self.multi_scale_attn_pooling = MultiLevelTemplateAttentionPooling(hidden_size=768, proj_size=512, num_levels=4)


    self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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

    with torch.inference_mode():
      if (nc>256):
        for batch_idx in range(0, bs):
          batch_vision_enc_input = vision_enc_input['pixel_values'][nc*batch_idx:nc*(batch_idx+1)]
          batch_vision_enc_input = batch_vision_enc_input.to(self.vision_encoder.device)
          vision_model_output = self.vision_encoder(
            batch_vision_enc_input,
            output_hidden_states=False,
            return_dict=True)

          # temp_hidden_states = [vision_model_output['hidden_states'][h_idx] for h_idx in want_layer]
          # temp_attn_pooling_result = self.multi_scale_attn_pooling(temp_hidden_states)['pooled_feature_proj']
          if batch_idx == 0:
            dino_result = vision_model_output['pooler_output']
          else:
            dino_result = torch.cat([dino_result, vision_model_output['pooler_output']], dim=0)

      else:
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
    # self.template_feats = checkpoint(self.DINO_linear_layer, dino_result)
    self.template_feats = self.DINO_linear_layer(dino_result)
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

# -------------------------------------------------
# Function

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
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
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
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
            A2C2f_Template_MaxSigmoidAttn
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
            A2C2f_Template_MaxSigmoidAttn
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
                # A2C2f_Template_MaxSigmoidAttn

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
        elif m in frozenset({Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB}:
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

# def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
#   """Parse a YOLO model.yaml dictionary into a PyTorch model."""
#   import ast

#   # Args
#   legacy = True  # backward compatibility for v3/v5/v8/v9 models
#   max_channels = float("inf")
#   nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
#   depth, width, kpt_shape = (d.get(x, 1.0) for x in (
#       "depth_multiple", "width_multiple", "kpt_shape"))
#   if scales:
#     scale = d.get("scale")
#     if not scale:
#       scale = tuple(scales.keys())[0]
#       LOGGER.warning(
#           f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
#     depth, width, max_channels = scales[scale]

#   if act:
#     # redefine default activation, i.e. Conv.default_act = nn.SiLU()
#     Conv.default_act = eval(act)
#     if verbose:
#       LOGGER.info(f"{colorstr('activation:')} {act}")  # print

#   if verbose:
#     LOGGER.info(
#         f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

#   ch = [ch]
#   layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
#   # from, number, module, args
#   for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
#     m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[
#         m]  # get module
#     for j, a in enumerate(args):
#       if isinstance(a, str):
#         try:
#           args[j] = locals()[a] if a in locals(
#           ) else ast.literal_eval(a)
#         except ValueError:
#           pass
#     n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
#     if m in {
#         Classify,
#         Conv,
#         ConvTranspose,
#         GhostConv,
#         Bottleneck,
#         GhostBottleneck,
#         SPP,
#         SPPF,
#         C2fPSA,
#         C2PSA,
#         DWConv,
#         Focus,
#         BottleneckCSP,
#         C1,
#         C2,
#         C2f,
#         C3k2,
#         RepNCSPELAN4,
#         ELAN1,
#         ADown,
#         AConv,
#         SPPELAN,
#         C2fAttn,
#         C2f_v2v_Attn,
#         C3,
#         C3TR,
#         C3Ghost,
#         nn.ConvTranspose2d,
#         DWConvTranspose2d,
#         C3x,
#         RepC3,
#         PSA,
#         SCDown,
#         C2fCIB,
#     }:
#       c1, c2 = ch[f], args[0]
#       # if c2 not equal to number of classes (i.e. for Classify() output)
#       if c2 != nc:
#         # make number you provide can be divisibled by 8 by closing the nearest number of 8's multiple
#         c2 = make_divisible(min(c2, max_channels) * width, 8)
#       if (m is C2fAttn) or (m is C2f_v2v_Attn):
#         args[1] = make_divisible(
#             min(args[1], max_channels // 2) * width, 8)  # embed channels
#         args[2] = int(
#             max(round(min(args[2], max_channels // 2 // 32))
#                 * width, 1) if args[2] > 1 else args[2]
#         )  # num heads
#       # if m is C2f_v2v_Attn: breakpoint()
#       args = [c1, c2, *args[1:]]

#       if m in {
#           BottleneckCSP,
#           C1,
#           C2,
#           C2f,
#           C3k2,
#           C2fAttn,
#           C2f_v2v_Attn,
#           C3,
#           C3TR,
#           C3Ghost,
#           C3x,
#           RepC3,
#           C2fPSA,
#           C2fCIB,
#           C2PSA,
#       }:
#         args.insert(2, n)  # number of repeats
#         n = 1
#       if m is C3k2:  # for M/L/X sizes
#         legacy = False
#         if scale in "mlx":
#           args[3] = True
#     elif m is AIFI:
#       args = [ch[f], *args]
#     elif m in {HGStem, HGBlock}:
#       c1, cm, c2 = ch[f], args[0], args[1]
#       args = [c1, cm, c2, *args[2:]]
#       if m is HGBlock:
#         args.insert(4, n)  # number of repeats
#         n = 1
#     elif m is ResNetLayer:
#       c2 = args[1] if args[3] else args[1] * 4
#     elif m is nn.BatchNorm2d:
#       args = [ch[f]]
#     elif m is Concat:
#       c2 = sum(ch[x] for x in f)
#     elif m is ImagePoolingAttn:
#       args.insert(1, [ch[x] for x in f])
#     elif m in {Detect, WorldDetect, Segment, Pose, OBB, v10Detect}:
#       args.append([ch[x] for x in f])
#       if m is Segment:
#         args[2] = make_divisible(min(args[2], max_channels) * width, 8)
#       if m in {Detect, Segment, Pose, OBB}:
#         m.legacy = legacy
#     elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
#       args.insert(1, [ch[x] for x in f])
#     elif m is CBLinear:
#       c2 = args[0]
#       c1 = ch[f]
#       args = [c1, c2, *args[1:]]
#     elif m is CBFuse:
#       c2 = ch[f[-1]]
#     else:
#       c2 = ch[f]

#     m_ = nn.Sequential(*(m(*args) for _ in range(n))
#                        ) if n > 1 else m(*args)  # module
#     # to repeat n layers if setting in yaml (some function has been expand in the function in up)

#     t = str(m)[8:-2].replace("__main__.", "")  # module type
#     m_.np = sum(x.numel() for x in m_.parameters())  # number params
#     m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
#     if verbose:
#       LOGGER.info(
#           f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
#     save.extend(x % i for x in ([f] if isinstance(
#         f, int) else f) if x != -1)  # append to savelist
#     layers.append(m_)
#     if i == 0:
#       ch = []
#     ch.append(c2)  # will record each layer's output channel size
#   return nn.Sequential(*layers), sorted(save)


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
