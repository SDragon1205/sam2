from v2vdet.v2vdet_ultralytics.utils.loss import (v2v_ObjectOriented_v8DetectionLoss,
                                           v2v_ObjectOriented_E2EDetectLoss)
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
from PIL import Image, ImageDraw
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
from ultralytics.data.augment import LoadVisualPrompt
from v2vdet.v2vdet_ultralytics.nn.tasks import DetectionModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

class V2V_With_MultiScale_SAVPE_ObjectOriented_Model(DetectionModel):
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

        # if hasattr(self.yaml, want_layers):
        if 'want_layers' in self.yaml:
            self.want_layers = self.yaml['want_layers'] if (self.yaml['want_layers'] is not None) else [-1]
        else:
            self.want_layers = [-1]
    
        LOGGER.info(f"Using layers: {self.want_layers} for multi scale.")
        
        self.image_processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path="google/siglip2-base-patch16-224")
        self.vision_encoder = SiglipVisionModel.from_pretrained(
        pretrained_model_name_or_path="google/siglip2-base-patch16-224") 
        self.vision_encoder_patch_size = 224
        self.patch_emb_savpe = PATCH_EMBEDDING_SAVPE()

        self.vpe = torch.zeros(1, 80, 512)  # features placeholder
        # self.origin_nc = self.nc
    
    # def multi_view_inference_set_classes(self, imgs, bboxes, class_names):
    #     from ultralytics.data.augment import LoadVisualPrompt
        
    #     device = next(self.vision_encoder.parameters()).device
        
    #     transform_vision_encoder = transforms.Compose([
    #         transforms.Resize((self.vision_encoder_patch_size, self.vision_encoder_patch_size), antialias=True),
    #         transforms.ToTensor()
    #     ])

    #     transform_visual_prompt = transforms.Compose([
    #         transforms.Resize((640, 640), antialias=True),
    #         transforms.ToTensor()
    #     ])
        
    #     imgs_list = []
    #     if isinstance(imgs, list):
    #         for img in imgs:
    #             if isinstance(img, str):
    #                 img = Image.open(img).convert("RGB")
    #             elif isinstance(img, np.ndarray):
    #                 img = Image.fromarray(img)
    #             elif not isinstance(img, Image.Image):
    #                 raise ValueError("imgs should be List[Image.Image] or np.array")
    #             imgs_list.append(img)
    #     else:
    #         if isinstance(imgs, str):
    #             imgs_list = [Image.open(imgs).convert("RGB")]
    #         elif isinstance(imgs, np.ndarray):
    #             imgs_list = [Image.fromarray(imgs)]
    #         elif isinstance(imgs, Image.Image):
    #             imgs_list = [imgs]
    #         else:
    #             raise ValueError("imgs should be List[Image.Image] or np.array")
        
    #     template_img_origin_size = [img.size for img in imgs_list]
    #     img_tensor_list = [transform_vision_encoder(img) for img in imgs_list]
    #     batch_tensor = torch.stack(img_tensor_list)
        
    #     # For Vision Encoder
    #     vision_encoder_inputs = {'pixel_values': batch_tensor.to(device)}
    #     logging.info(f"Vision encoder input shape: {batch_tensor.shape}")
        
    #     # For Visual Prompt
    #     visual_prompt_list = []
    #     class_to_idx = {name: idx for idx, name in enumerate(set(class_names))}
    #     logging.info(f"Class to index mapping: {class_to_idx}")
    #     for img_idx, img in enumerate(imgs_list):
    #         class_idx = torch.tensor([class_to_idx[class_names[img_idx]]]).unsqueeze(0)
            
    #         if isinstance(bboxes, list):
    #             bbox = bboxes[img_idx]
    #         elif isinstance(bboxes, torch.Tensor):
    #             if bboxes.dim() == 2:
    #                 bbox = bboxes[img_idx].unsqueeze(0)
    #             elif bboxes.dim() == 1:
    #                 bbox = bboxes
    #             else:
    #                 raise ValueError("bboxes has some issues.")
    #         else:
    #             raise ValueError("bboxes should be List[Image.Image] or torch.tensor in 2D or 1D")
            
    #         origin_size = template_img_origin_size[img_idx]
    #         n_bbox = torch.zeros_like(bbox, dtype=torch.float)
    #         n_bbox[0][0] = bbox[0][0]/origin_size[1]
    #         n_bbox[0][1] = bbox[0][1]/origin_size[0]
    #         n_bbox[0][2] = bbox[0][2]/origin_size[1]
    #         n_bbox[0][3] = bbox[0][3]/origin_size[0]
            
    #         visual_prompt_label = {
    #             'img': transform_visual_prompt(img),
    #             'bboxes': torch.tensor(n_bbox),
    #             'cls': class_idx,
    #         }
            
    #         load_visual_prompt_func = LoadVisualPrompt()
    #         visual_prompt_label = load_visual_prompt_func(visual_prompt_label)
    #         visual_prompt_list.append(visual_prompt_label['visuals'])
        
    #     visuals_tensor = torch.stack(visual_prompt_list)
    #     logging.info(f"Visuals tensor shape: {visuals_tensor.shape}")

    #     if isinstance(self.vision_encoder, pe.VisionTransformer):
    #         vision_model_output = self.vision_encoder(vision_encoder_inputs['pixel_values'], output_hidden_list=self.want_layers, strip_cls_token=True)
    #         savpe_input = vision_model_output['hidden_states'].permute(1, 0, 2, 3)
    #         vpe = self.patch_emb_savpe(savpe_input, visuals_tensor)
    #     else:
    #         vision_model_output = self.vision_encoder(**vision_encoder_inputs, output_hidden_states=True, return_dict=True)
    #         hidden_states_list = [[] for _ in range(len(self.want_layers))]
    #         for idx, layer_num in enumerate(self.want_layers):
    #             hidden_states_list[idx] = vision_model_output['hidden_states'][layer_num]
    #         hidden_states_tensor = torch.stack(tensors=hidden_states_list, dim=1)
    #         vpe = self.patch_emb_savpe(hidden_states_tensor, visuals_tensor)
        
    #     # Correct dimension handling
    #     logging.info(f"Raw VPE shape: {vpe.shape}")
    #     # Squeeze leading dimension if present
    #     if vpe.size(0) == 1:
    #         vpe = vpe.squeeze(0)  # Shape: [num_images, embed_dim]
    #         logging.info(f"VPE shape after squeeze: {vpe.shape}")
    #     else:
    #         vpe = vpe.squeeze(0)  # Shape: [num_images, embed_dim]
    #         logging.info(f"VPE shape after squeeze: {vpe.shape}")
        
    #     # Ensure vpe has shape [num_images, num_patches, embed_dim]
    #     if vpe.dim() == 2:
    #         vpe = vpe.unsqueeze(1)  # Shape: [num_images, 1, embed_dim]
    #     logging.info(f"VPE shape after unsqueeze: {vpe.shape}")
        
    #     batch_size = vpe.size(0)
    #     num_patches = vpe.size(1)
    #     logging.info(f"Batch size: {batch_size}, Num patches: {num_patches}")
        
    #     # Group embeddings by class and compute mean
    #     class_indices = [class_to_idx[name] for name in class_names]
    #     num_classes = len(class_to_idx)
    #     class_embeddings = []
        
    #     for cls_idx in range(num_classes):
    #         cls_mask = torch.tensor([i == cls_idx for i in class_indices], device=vpe.device)
    #         cls_name = list(class_to_idx.keys())[cls_idx]
    #         logging.info(f"Class {cls_idx} ({cls_name}): {cls_mask.sum()} images")
            
    #         if cls_mask.sum() == 0:
    #             logging.warning(f"No valid images for class index {cls_idx} ({cls_name})")
    #             class_embeddings.append(torch.zeros(1, num_patches, vpe.size(2), device=vpe.device))
    #             continue
            
    #         # Select embeddings for this class
    #         cls_vpe = vpe[cls_mask]  # Shape: [num_images_for_class, num_patches, embed_dim]
    #         logging.info(f"cls_vpe shape for class {cls_idx}: {cls_vpe.shape}")
            
    #         # Compute mean across images for this class
    #         cls_mean_vpe = cls_vpe.mean(dim=0, keepdim=True)  # Shape: [1, num_patches, embed_dim]
    #         logging.info(f"cls_mean_vpe shape for class {cls_idx}: {cls_mean_vpe.shape}")
    #         class_embeddings.append(cls_mean_vpe)
        
    #     if not class_embeddings:
    #         raise ValueError("No valid class embeddings computed")
        
    #     # Stack mean embeddings
    #     self.vpe = torch.cat(class_embeddings, dim=0)  # Shape: [num_classes, num_patches, embed_dim]
    #     self.nc = self.vpe.shape[0]
    #     self.model[-1].nc = self.nc
    #     logging.info(f"Final number of classes (self.nc): {self.nc}, VPE final shape: {self.vpe.shape}")
    #     # import pdb; pdb.set_trace()
    #     self.vpe = self.vpe.permute(1, 0, 2)
    #     # Validate self.nc against expected model configuration
    #     # num_anchors = getattr(self.model[-1], 'na', 3)  # Assume 3 anchors if na not set
    #     # expected_no = self.nc * (5 + num_anchors)
    #     # logging.info(f"Expected no (nc * (5 + num_anchors)): {expected_no}")
    #     # if expected_no != getattr(self.model[-1], 'no', expected_no):
    #     #     logging.error(f"Mismatch in detection head: expected no={expected_no}, but model.no={self.model[-1].no}")
    #     #     raise ValueError(f"Detection head mismatch: expected no={expected_no}, got {self.model[-1].no}")


    def multi_view_inference_set_classes(self, imgs, bboxes, class_names):
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

        # Lists to store padded images and adjusted bboxes
        padded_imgs_list = []
        adjusted_bboxes = []

        # Process each image for padding and adjust bboxes
        for img_idx, img in enumerate(imgs_list):
            original_width, original_height = img.size
            max_side = max(original_width, original_height)
            
            # Compute padding
            pad_width = max_side - original_width
            pad_height = max_side - original_height
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            top_pad = pad_height // 2
            bottom_pad = pad_height - top_pad

            # Create a new blank image with zero-padding
            padded_img = Image.new('RGB', (max_side, max_side), (0, 0, 0))
            padded_img.paste(img, (left_pad, top_pad))
            padded_imgs_list.append(padded_img)

            # Adjust bboxes
            if isinstance(bboxes, list):
                bbox = bboxes[img_idx]
            elif isinstance(bboxes, torch.Tensor):
                if bboxes.dim() == 2:
                    bbox = bboxes[img_idx].unsqueeze(0)
                elif bboxes.dim() == 1:
                    bbox = bboxes.unsqueeze(0)
                else:
                    raise ValueError("bboxes has some issues.")
            else:
                raise ValueError("bboxes should be List[Image.Image] or torch.tensor in 2D or 1D")

            # Convert bboxes from [top_x, top_y, width, height] to [x_min, y_min, x_max, y_max]
            converted_bbox = torch.zeros_like(bbox, dtype=torch.float)
            converted_bbox[:, 0] = bbox[:, 0]  # x_min = top_x
            converted_bbox[:, 1] = bbox[:, 1]  # y_min = top_y
            converted_bbox[:, 2] = bbox[:, 0] + bbox[:, 2]  # x_max = top_x + width
            converted_bbox[:, 3] = bbox[:, 1] + bbox[:, 3]  # y_max = top_y + height

            # Adjust bbox coordinates based on padding
            adjusted_bbox = converted_bbox.clone()
            adjusted_bbox[:, 0] += left_pad  # x_min
            adjusted_bbox[:, 2] += left_pad  # x_max
            adjusted_bbox[:, 1] += top_pad   # y_min
            adjusted_bbox[:, 3] += top_pad   # y_max
            adjusted_bboxes.append(adjusted_bbox)

        # Update bboxes to use adjusted ones (in [x_min, y_min, x_max, y_max] format)
        if isinstance(bboxes, list):
            bboxes = adjusted_bboxes
        else:
            bboxes = torch.cat(adjusted_bboxes, dim=0) if bboxes.dim() == 2 else adjusted_bboxes[0]

        # Update template_img_origin_size with padded sizes
        template_img_origin_size = [(max_side, max_side) for _ in padded_imgs_list]
        img_tensor_list = [transform_vision_encoder(img) for img in padded_imgs_list]
        batch_tensor = torch.stack(img_tensor_list)
        
        # For Vision Encoder
        vision_encoder_inputs = {'pixel_values': batch_tensor.to(device)}
        logging.info(f"Vision encoder input shape: {batch_tensor.shape}")
        
        # For Visual Prompt
        visual_prompt_list = []
        class_to_idx = {name: idx for idx, name in enumerate(set(class_names))}
        logging.info(f"Class to index mapping: {class_to_idx}")
        for img_idx, img in enumerate(padded_imgs_list):
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

            origin_size = template_img_origin_size[img_idx]
            n_bbox = torch.zeros_like(bbox, dtype=torch.float)
            # Convert from [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height]
            n_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2  # x_center = (x_min + x_max) / 2
            n_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2  # y_center = (y_min + y_max) / 2
            n_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]       # width = x_max - x_min
            n_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]       # height = y_max - y_min
            # Normalize
            n_bbox[:, 0] /= origin_size[1]  # x_center / width
            n_bbox[:, 1] /= origin_size[0]  # y_center / height
            n_bbox[:, 2] /= origin_size[1]  # width / width
            n_bbox[:, 3] /= origin_size[0]  # height / height
            
            visual_prompt_label = {
                'img': transform_visual_prompt(img),
                'bboxes': n_bbox,
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
        self.vpe = self.vpe.permute(1, 0, 2)

    # def inference_set_classes(self, imgs, bboxes):
    #     device = next(self.vision_encoder.parameters()).device

    #     transform_vision_encoder = transforms.Compose([
    #         transforms.Resize((self.vision_encoder_patch_size, self.vision_encoder_patch_size), antialias=True),
    #         transforms.ToTensor()
    #     ])

    #     transform_visual_prompt = transforms.Compose([
    #         transforms.Resize((640, 640), antialias=True),
    #         transforms.ToTensor()
    #     ])

    #     imgs_list = []
    #     if isinstance(imgs, list):
    #         for img in imgs:
    #             if isinstance(img, str):
    #                 img = Image.open(img).convert("RGB")
    #             elif isinstance(img, np.ndarray):
    #                 img = Image.fromarray(img)
    #             elif not isinstance(img, Image.Image):
    #                 raise ValueError("imgs should be List[Image.Image] or np.array")
    #             imgs_list.append(img)
    #     else:
    #         if isinstance(imgs, str):
    #             imgs_list = [Image.open(imgs).convert("RGB")]
    #         elif isinstance(imgs, np.ndarray):
    #             imgs_list = [Image.fromarray(imgs)]
    #         elif isinstance(imgs, Image.Image):
    #             imgs_list = [imgs]
    #         else:
    #             raise ValueError("imgs should be List[Image.Image] or np.array")

    #     # Lists to store bboxes
    #     adjusted_bboxes = []

    #     # Process each image and adjust bboxes
    #     for img_idx, img in enumerate(imgs_list):
    #         # Adjust bboxes
    #         if isinstance(bboxes, list):
    #             bbox = bboxes[img_idx]
    #         elif isinstance(bboxes, torch.Tensor):
    #             if bboxes.dim() == 2:
    #                 bbox = bboxes[img_idx].unsqueeze(0)
    #             elif bboxes.dim() == 1:
    #                 bbox = bboxes.unsqueeze(0)
    #             else:
    #                 raise ValueError("bboxes has some issues.")
    #         else:
    #             raise ValueError("bboxes should be List[Image.Image] or torch.tensor in 2D or 1D")

    #         # Convert bboxes from [top_x, top_y, width, height] to [x_min, y_min, x_max, y_max]
    #         converted_bbox = torch.zeros_like(bbox, dtype=torch.float)
    #         converted_bbox[:, 0] = bbox[:, 0]  # x_min = top_x
    #         converted_bbox[:, 1] = bbox[:, 1]  # y_min = top_y
    #         converted_bbox[:, 2] = bbox[:, 0] + bbox[:, 2]  # x_max = top_x + width
    #         converted_bbox[:, 3] = bbox[:, 1] + bbox[:, 3]  # y_max = top_y + height

    #         adjusted_bboxes.append(converted_bbox)

    #     # Update bboxes to use adjusted ones (in [x_min, y_min, x_max, y_max] format)
    #     if isinstance(bboxes, list):
    #         bboxes = adjusted_bboxes
    #     else:
    #         bboxes = torch.cat(adjusted_bboxes, dim=0) if bboxes.dim() == 2 else adjusted_bboxes[0]

    #     # Use original image sizes
    #     template_img_origin_size = [img.size for img in imgs_list]
    #     img_tensor_list = [transform_vision_encoder(img) for img in imgs_list]
    #     batch_tensor = torch.stack(img_tensor_list)

    #     # For Vision Encoder
    #     vision_encoder_inputs = {'pixel_values': batch_tensor.to(device)}

    #     # For Visual Prompt
    #     visual_prompt_list = []
    #     for img_idx, img in enumerate(imgs_list):
    #         class_idx = torch.tensor([img_idx]).unsqueeze(0)

    #         if isinstance(bboxes, list):
    #             bbox = bboxes[img_idx]
    #         elif isinstance(bboxes, torch.Tensor):
    #             if bboxes.dim() == 2:
    #                 bbox = bboxes[img_idx].unsqueeze(0)
    #             elif bboxes.dim() == 1:
    #                 bbox = bboxes
    #             else:
    #                 raise ValueError("bboxes has some issues.")

    #         origin_size = template_img_origin_size[img_idx]
    #         n_bbox = torch.zeros_like(bbox, dtype=torch.float)
    #         # Convert from [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height]
    #         n_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2  # x_center = (x_min + x_max) / 2
    #         n_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2  # y_center = (y_min + y_max) / 2
    #         n_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]       # width = x_max - x_min
    #         n_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]       # height = y_max - y_min
    #         # Normalize
    #         n_bbox[:, 0] /= origin_size[1]  # x_center / width
    #         n_bbox[:, 1] /= origin_size[0]  # y_center / height
    #         n_bbox[:, 2] /= origin_size[1]  # width / width
    #         n_bbox[:, 3] /= origin_size[0]  # height / height

    #         visual_prompt_label = {
    #             'img': transform_visual_prompt(img),
    #             'bboxes': n_bbox,
    #             'cls': class_idx,
    #         }

    #         load_visual_prompt_func = LoadVisualPrompt()
    #         visual_prompt_label = load_visual_prompt_func(visual_prompt_label)
    #         visual_prompt_list.append(visual_prompt_label['visuals'])

    #     visuals_tensor = torch.stack(visual_prompt_list)

    #     if isinstance(self.vision_encoder, pe.VisionTransformer):
    #         vision_model_output = self.vision_encoder(vision_encoder_inputs['pixel_values'], output_hidden_list=self.want_layers, strip_cls_token=True)
    #         savpe_input = vision_model_output['hidden_states'].permute(1, 0, 2, 3)
    #         self.vpe = self.patch_emb_savpe(savpe_input, visuals_tensor)
    #     else:
    #         vision_model_output = self.vision_encoder(**vision_encoder_inputs, output_hidden_states=True, return_dict=True)
    #         hidden_states_list = [[] for _ in range(len(self.want_layers))]
    #         for idx, layer_num in enumerate(self.want_layers):
    #             hidden_states_list[idx] = vision_model_output['hidden_states'][layer_num]
    #         hidden_states_tensor = torch.stack(tensors=hidden_states_list, dim=1)
    #         self.vpe = self.patch_emb_savpe(hidden_states_tensor, visuals_tensor)

    #     self.vpe = self.vpe.permute(1, 0, 2)
    #     self.nc = self.vpe.shape[1]
    #     self.model[-1].nc = self.nc

    def inference_set_classes(self, imgs, bboxes):
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

        # Lists to store padded images and adjusted bboxes
        padded_imgs_list = []
        adjusted_bboxes = []

        # Process each image for padding and adjust bboxes
        for img_idx, img in enumerate(imgs_list):
            original_width, original_height = img.size
            max_side = max(original_width, original_height)
            
            # Compute padding
            pad_width = max_side - original_width
            pad_height = max_side - original_height
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            top_pad = pad_height // 2
            bottom_pad = pad_height - top_pad

            # Create a new blank image with zero-padding
            padded_img = Image.new('RGB', (max_side, max_side), (0, 0, 0))
            padded_img.paste(img, (left_pad, top_pad))
            padded_imgs_list.append(padded_img)

            # Adjust bboxes
            if isinstance(bboxes, list):
                bbox = bboxes[img_idx]
            elif isinstance(bboxes, torch.Tensor):
                if bboxes.dim() == 2:
                    bbox = bboxes[img_idx].unsqueeze(0)
                elif bboxes.dim() == 1:
                    bbox = bboxes.unsqueeze(0)
                else:
                    raise ValueError("bboxes has some issues.")
            else:
                raise ValueError("bboxes should be List[Image.Image] or torch.tensor in 2D or 1D")

            # Convert bboxes from [top_x, top_y, width, height] to [x_min, y_min, x_max, y_max]
            converted_bbox = torch.zeros_like(bbox, dtype=torch.float)
            converted_bbox[:, 0] = bbox[:, 0]  # x_min = top_x
            converted_bbox[:, 1] = bbox[:, 1]  # y_min = top_y
            converted_bbox[:, 2] = bbox[:, 0] + bbox[:, 2]  # x_max = top_x + width
            converted_bbox[:, 3] = bbox[:, 1] + bbox[:, 3]  # y_max = top_y + height

            # Adjust bbox coordinates based on padding
            adjusted_bbox = converted_bbox.clone()
            adjusted_bbox[:, 0] += left_pad  # x_min
            adjusted_bbox[:, 2] += left_pad  # x_max
            adjusted_bbox[:, 1] += top_pad   # y_min
            adjusted_bbox[:, 3] += top_pad   # y_max
            adjusted_bboxes.append(adjusted_bbox)

        # Update bboxes to use adjusted ones (in [x_min, y_min, x_max, y_max] format)
        if isinstance(bboxes, list):
            bboxes = adjusted_bboxes
        else:
            bboxes = torch.cat(adjusted_bboxes, dim=0) if bboxes.dim() == 2 else adjusted_bboxes[0]

        # Update template_img_origin_size with padded sizes
        template_img_origin_size = [(max_side, max_side) for _ in padded_imgs_list]
        img_tensor_list = [transform_vision_encoder(img) for img in padded_imgs_list]
        batch_tensor = torch.stack(img_tensor_list)

        # For Vision Encoder
        vision_encoder_inputs = {'pixel_values': batch_tensor.to(device)}

        # For Visual Prompt
        visual_prompt_list = []
        for img_idx, img in enumerate(padded_imgs_list):
            class_idx = torch.tensor([img_idx]).unsqueeze(0)

            if isinstance(bboxes, list):
                bbox = bboxes[img_idx]
            elif isinstance(bboxes, torch.Tensor):
                if bboxes.dim() == 2:
                    bbox = bboxes[img_idx].unsqueeze(0)
                elif bboxes.dim() == 1:
                    bbox = bboxes
                else:
                    raise ValueError("bboxes has some issues.")

            origin_size = template_img_origin_size[img_idx]
            n_bbox = torch.zeros_like(bbox, dtype=torch.float)
            # Convert from [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height]
            n_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2  # x_center = (x_min + x_max) / 2
            n_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2  # y_center = (y_min + y_max) / 2
            n_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]       # width = x_max - x_min
            n_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]       # height = y_max - y_min
            # Normalize
            n_bbox[:, 0] /= origin_size[1]  # x_center / width
            n_bbox[:, 1] /= origin_size[0]  # y_center / height
            n_bbox[:, 2] /= origin_size[1]  # width / width
            n_bbox[:, 3] /= origin_size[0]  # height / height

            visual_prompt_label = {
                'img': transform_visual_prompt(img),
                'bboxes': n_bbox,
                'cls': class_idx,
            }

            load_visual_prompt_func = LoadVisualPrompt()
            visual_prompt_label = load_visual_prompt_func(visual_prompt_label)
            visual_prompt_list.append(visual_prompt_label['visuals'])

        visuals_tensor = torch.stack(visual_prompt_list)

        if isinstance(self.vision_encoder, pe.VisionTransformer):
            vision_model_output = self.vision_encoder(vision_encoder_inputs['pixel_values'], output_hidden_list=self.want_layers, strip_cls_token=True)
            savpe_input = vision_model_output['hidden_states'].permute(1, 0, 2, 3)
            self.vpe = self.patch_emb_savpe(savpe_input, visuals_tensor)
        else:
            vision_model_output = self.vision_encoder(**vision_encoder_inputs, output_hidden_states=True, return_dict=True)
            hidden_states_list = [[] for _ in range(len(self.want_layers))]
            for idx, layer_num in enumerate(self.want_layers):
                hidden_states_list[idx] = vision_model_output['hidden_states'][layer_num]
            hidden_states_tensor = torch.stack(tensors=hidden_states_list, dim=1)
            self.vpe = self.patch_emb_savpe(hidden_states_tensor, visuals_tensor)

        self.vpe = self.vpe.permute(1, 0, 2)
        self.nc = self.vpe.shape[1]
        self.model[-1].nc = self.nc

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
    #   import pdb; pdb.set_trace()

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
    #   transform = transforms.Compose([
    #       transforms.Resize((self.vision_encoder_patch_size, self.vision_encoder_patch_size), antialias=True)
    #   ])
    #   inputs = {'pixel_values': transform(vision_encoder_input).to(device)}
      inputs = {'pixel_values': vision_encoder_input}


      vp_list = []
      for img_idx in range(imgs.shape[0]):
          v = visuals[img_idx].to(device)
          
          # 找出屬於當前圖像的所有匹配項索引
          matches = (batch_idx == img_idx).nonzero()
          vp_single = torch.zeros(nc, v.shape[-2], v.shape[-1], device=device)  # 在相同設備上創建
          
          # 只有在有匹配項時處理
          if len(matches) > 0:
              # 獲取批次中當前圖像的開始位置和數量
              batch_start = matches[0].item()
              batch_count = len(matches)
              

              img_classes = batch_cls[batch_start:batch_start + batch_count].to(torch.int32)
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
            self.set_classes(nc=nc, imgs=batch['t']['img'], batch_cls=batch['t']['cls'], bboxes=batch['t']['bboxes'], batch_idx=batch['t']['batch_idx'], visuals=batch['t']['visuals'])
            
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
        preds = self.forward(batch['i']["img"], batch)
      return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v2v_ObjectOriented_E2EDetectLoss(self) if getattr(self, "end2end", False) else v2v_ObjectOriented_v8DetectionLoss(self)
    
class V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented_Model(V2V_With_MultiScale_SAVPE_ObjectOriented_Model):
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

class V2V_With_MultiScale_SAVPE_SigLIP2_L_ObjectOriented_Model(V2V_With_MultiScale_SAVPE_ObjectOriented_Model):
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

class V2V_With_MultiScale_SAVPE_PE_B16_ObjectOriented_Model(V2V_With_MultiScale_SAVPE_ObjectOriented_Model):
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
      
    #   transform = transforms.Compose([
    #       transforms.Resize((self.vision_encoder_input_size, self.vision_encoder_input_size), antialias=True)
    #   ])
    #   inputs = transform(vision_encoder_input).to(device)
      
      inputs = vision_encoder_input.to(device)
      vision_model_output = self.vision_encoder(inputs,
                                                output_hidden_list=self.want_layers,
                                                strip_cls_token=True)
      
      self.vpe = self.patch_emb_savpe(vision_model_output['hidden_states'].permute(1, 0, 2, 3), visuals_tensor)
      
      self.nc = nc
      self.model[-1].nc = nc

      return self.vpe

class V2V_With_MultiScale_SAVPE_PE_L14_ObjectOriented_Model(V2V_With_MultiScale_SAVPE_PE_B16_ObjectOriented_Model):
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
        
        self.vision_encoder = pe.VisionTransformer.from_config(name="PE-Core-L14-336", pretrained=True)
        # self.image_processor = pe_transforms.get_image_transform(self.vision_encoder.image_size)
        
        self.patch_emb_savpe = PATCH_EMBEDDING_SAVPE(embed_dim=1024)

        self.vpe = torch.zeros(1, 80, 512)  # features placeholder
        # self.origin_nc = self.nc
        self.vision_encoder_input_size = 336
        
class V2V_With_MultiScale_SAVPE_DINOv2_B_ObjectOriented_Model(V2V_With_MultiScale_SAVPE_ObjectOriented_Model):
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
        pretrained_model_name_or_path='facebook/dinov2-base')
        self.vision_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path='facebook/dinov2-base') 
        self.patch_emb_savpe = PATCH_EMBEDDING_SAVPE()
        self.vision_encoder_patch_size = 256

        self.vpe = torch.zeros(1, 80, 512)  # features placeholder
        # self.origin_nc = self.nc
        
class V2V_With_MultiScale_SAVPE_DINOv2_L_ObjectOriented_Model(V2V_With_MultiScale_SAVPE_ObjectOriented_Model):
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
        pretrained_model_name_or_path='facebook/dinov2-large')
        self.vision_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path='facebook/dinov2-large') 
        self.patch_emb_savpe = PATCH_EMBEDDING_SAVPE(embed_dim=1024)
        self.vision_encoder_patch_size = 256

        self.vpe = torch.zeros(1, 80, 512)  # features placeholder
        # self.origin_nc = self.nc
        
class V2V_With_MultiScale_SAVPE_YOLOE_ObjectOriented_Model(V2V_With_MultiScale_SAVPE_ObjectOriented_Model):        
    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters."""
        from ultralytics.nn.modules.block import SAVPE

        self.template_feats = torch.randn(1, nc or 80, 512)
        # features placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # here will parse the model
        del self.image_processor
        del self.vision_encoder
        self.template_backbone_model = self.model 
        self.patch_emb_savpe = self.model[-1].savpe
        self.vpe = torch.zeros(1, 80, 512)  # features placeholder

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
            if isinstance(m, YOLOEDetect):
                return x
        
            # if hasattr(self.args, 'gradient_checkpointing'):  
            #     if self.args.gradient_checkpointing:
            #         x = checkpoint(m, x)
            #     else:
            #         x = m(x)    
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
    
    def set_classes(self, vpe: torch.Tensor=None, nc=80, imgs=None, batch_cls=None, bboxes=None, batch_idx=None, visuals=None):
      """Set classes in advance so that model could do offline-inference"""
      
      # device = self.vision_encoder.device
      device = next(self.model.parameters()).device
      # ---- 
      # If vpe (feature of vp and attention with imgs) is given, then use it.
      if vpe is not None:
        self.vpe = vpe
        self.model[-1].nc = vpe.shape[1]
        return
      # ---- 
      
      vp_list = []
      for img_idx in range(imgs.shape[0]):
          v = visuals[img_idx].to(device)
          
          # 找出屬於當前圖像的所有匹配項索引
          matches = (batch_idx == img_idx).nonzero()
          vp_single = torch.zeros(nc, v.shape[-2], v.shape[-1], device=device)  # 在相同設備上創建
          
          # 只有在有匹配項時處理
          if len(matches) > 0:
              # 獲取批次中當前圖像的開始位置和數量
              batch_start = matches[0].item()
              batch_count = len(matches)
              
              img_classes = batch_cls[batch_start:batch_start + batch_count].to(torch.int32)
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
      
      hidden_states_tensor = self._vision_encoder_forward(x=imgs)
      
      self.vpe = self.patch_emb_savpe(hidden_states_tensor, visuals_tensor)
      
      self.nc = nc
      self.model[-1].nc = nc

      return self.vpe