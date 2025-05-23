# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transforms and data augmentation for both image + bbox.
"""

import logging
import math
import random
from typing import Iterable
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch.nn.functional as NF
import torchvision.transforms.v2.functional as Fv2
from PIL import Image as PILImage
from torchvision.transforms import InterpolationMode

from training.utils.data_utils import VideoDatapoint, VideoDatapoint_yolo
import sys

def hflip(datapoint, index):

    datapoint.frames[index].data = F.hflip(datapoint.frames[index].data)
    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.hflip(obj.segment)

    return datapoint


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = int(round(size))
        oh = int(round(size * h / w))
    else:
        oh = int(round(size))
        ow = int(round(size * w / h))

    return (oh, ow)


def resize(datapoint, index, size, max_size=None, square=False, v2=False):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if square:
        size = size, size
    else:
        cur_size = (
            datapoint.frames[index].data.size()[-2:][::-1]
            if v2
            else datapoint.frames[index].data.size
        )
        size = get_size(cur_size, size, max_size)

    old_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )
    if v2:
        datapoint.frames[index].data = Fv2.resize(
            datapoint.frames[index].data, size, antialias=True
        )
    else:
        datapoint.frames[index].data = F.resize(datapoint.frames[index].data, size)

    new_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.resize(obj.segment[None, None], size).squeeze()

    h, w = size
    datapoint.frames[index].size = (h, w)
    return datapoint


def pad(datapoint, index, padding, v2=False):
    old_h, old_w = datapoint.frames[index].size
    h, w = old_h, old_w
    if len(padding) == 2:
        # assumes that we only pad on the bottom right corners
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data, (0, 0, padding[0], padding[1])
        )
        h += padding[1]
        w += padding[0]
    else:
        # left, top, right, bottom
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data,
            (padding[0], padding[1], padding[2], padding[3]),
        )
        h += padding[1] + padding[3]
        w += padding[0] + padding[2]

    datapoint.frames[index].size = (h, w)

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            if v2:
                if len(padding) == 2:
                    obj.segment = Fv2.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = Fv2.pad(obj.segment, tuple(padding))
            else:
                if len(padding) == 2:
                    obj.segment = F.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = F.pad(obj.segment, tuple(padding))
    return datapoint


class RandomHorizontalFlip:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.frames)):
                    datapoint = hflip(datapoint, i)
            return datapoint
        for i in range(len(datapoint.frames)):
            if random.random() < self.p:
                datapoint = hflip(datapoint, i)
        return datapoint


class RandomResizeAPI:
    def __init__(
        self, sizes, consistent_transform, max_size=None, square=False, v2=False
    ):
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform
        self.v2 = v2

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            size = random.choice(self.sizes)
            # print("RandomResizeAPI datapoint:", datapoint)
            for i in range(len(datapoint.frames)):
                datapoint = resize(
                    datapoint, i, size, self.max_size, square=self.square, v2=self.v2
                )
            return datapoint
        for i in range(len(datapoint.frames)):
            size = random.choice(self.sizes)
            datapoint = resize(
                datapoint, i, size, self.max_size, square=self.square, v2=self.v2
            )
        return datapoint


class ToTensorAPI:
    def __init__(self, v2=False):
        self.v2 = v2

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.to_image_tensor(img.data)
            else:
                img.data = F.to_tensor(img.data)
        return datapoint


class NormalizeAPI:
    def __init__(self, mean, std, v2=False):
        self.mean = mean
        self.std = std
        self.v2 = v2

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.convert_image_dtype(img.data, torch.float32)
                img.data = Fv2.normalize(img.data, mean=self.mean, std=self.std)
            else:
                img.data = F.normalize(img.data, mean=self.mean, std=self.std)

        return datapoint


class ComposeAPI:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datapoint, **kwargs):
        for t in self.transforms:
            datapoint = t(datapoint, **kwargs)
        return datapoint

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomGrayscale:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform
        self.Grayscale = T.Grayscale(num_output_channels=3)

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for img in datapoint.frames:
                    img.data = self.Grayscale(img.data)
            return datapoint
        for img in datapoint.frames:
            if random.random() < self.p:
                img.data = self.Grayscale(img.data)
        return datapoint


class ColorJitter:
    def __init__(self, consistent_transform, brightness, contrast, saturation, hue):
        self.consistent_transform = consistent_transform
        self.brightness = (
            brightness
            if isinstance(brightness, list)
            else [max(0, 1 - brightness), 1 + brightness]
        )
        self.contrast = (
            contrast
            if isinstance(contrast, list)
            else [max(0, 1 - contrast), 1 + contrast]
        )
        self.saturation = (
            saturation
            if isinstance(saturation, list)
            else [max(0, 1 - saturation), 1 + saturation]
        )
        self.hue = hue if isinstance(hue, list) or hue is None else ([-hue, hue])

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        if self.consistent_transform:
            # Create a color jitter transformation params
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = T.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        for img in datapoint.frames:
            if not self.consistent_transform:
                (
                    fn_idx,
                    brightness_factor,
                    contrast_factor,
                    saturation_factor,
                    hue_factor,
                ) = T.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue
                )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img.data = F.adjust_brightness(img.data, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img.data = F.adjust_contrast(img.data, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img.data = F.adjust_saturation(img.data, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img.data = F.adjust_hue(img.data, hue_factor)
        return datapoint


class RandomAffine:
    def __init__(
        self,
        degrees,
        consistent_transform,
        scale=None,
        translate=None,
        shear=None,
        image_mean=(123, 116, 103),
        log_warning=True,
        num_tentatives=1,
        image_interpolation="bicubic",
    ):
        """
        The mask is required for this transform.
        if consistent_transform if True, then the same random affine is applied to all frames and masks.
        """
        self.degrees = degrees if isinstance(degrees, list) else ([-degrees, degrees])
        self.scale = scale
        self.shear = (
            shear if isinstance(shear, list) else ([-shear, shear] if shear else None)
        )
        self.translate = translate
        self.fill_img = image_mean
        self.consistent_transform = consistent_transform
        self.log_warning = log_warning
        self.num_tentatives = num_tentatives

        if image_interpolation == "bicubic":
            self.image_interpolation = InterpolationMode.BICUBIC
        elif image_interpolation == "bilinear":
            self.image_interpolation = InterpolationMode.BILINEAR
        else:
            raise NotImplementedError

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for _tentative in range(self.num_tentatives):
            res = self.transform_datapoint(datapoint)
            if res is not None:
                return res

        if self.log_warning:
            logging.warning(
                f"Skip RandomAffine for zero-area mask in first frame after {self.num_tentatives} tentatives"
            )
        return datapoint

    def transform_datapoint(self, datapoint: VideoDatapoint):
        _, height, width = F.get_dimensions(datapoint.frames[0].data)
        img_size = [width, height]

        if self.consistent_transform:
            # Create a random affine transformation
            affine_params = T.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shear,
                img_size=img_size,
            )

        for img_idx, img in enumerate(datapoint.frames):
            this_masks = [
                obj.segment.unsqueeze(0) if obj.segment is not None else None
                for obj in img.objects
            ]
            if not self.consistent_transform:
                # if not consistent we create a new affine params for every frame&mask pair Create a random affine transformation
                affine_params = T.RandomAffine.get_params(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale_ranges=self.scale,
                    shears=self.shear,
                    img_size=img_size,
                )

            transformed_bboxes, transformed_masks = [], []
            for i in range(len(img.objects)):
                if this_masks[i] is None:
                    transformed_masks.append(None)
                    # Dummy bbox for a dummy target
                    transformed_bboxes.append(torch.tensor([[0, 0, 1, 1]]))
                else:
                    transformed_mask = F.affine(
                        this_masks[i],
                        *affine_params,
                        interpolation=InterpolationMode.NEAREST,
                        fill=0.0,
                    )
                    if img_idx == 0 and transformed_mask.max() == 0:
                        # We are dealing with a video and the object is not visible in the first frame
                        # Return the datapoint without transformation
                        return None
                    transformed_masks.append(transformed_mask.squeeze())

            for i in range(len(img.objects)):
                img.objects[i].segment = transformed_masks[i]

            img.data = F.affine(
                img.data,
                *affine_params,
                interpolation=self.image_interpolation,
                fill=self.fill_img,
            )
        return datapoint


def random_mosaic_frame(
    datapoint,
    index,
    grid_h,
    grid_w,
    target_grid_y,
    target_grid_x,
    should_hflip,
):
    # Step 1: downsize the images and paste them into a mosaic
    image_data = datapoint.frames[index].data
    is_pil = isinstance(image_data, PILImage.Image)
    if is_pil:
        H_im = image_data.height
        W_im = image_data.width
        image_data_output = PILImage.new("RGB", (W_im, H_im))
    else:
        H_im = image_data.size(-2)
        W_im = image_data.size(-1)
        image_data_output = torch.zeros_like(image_data)

    downsize_cache = {}
    for grid_y in range(grid_h):
        for grid_x in range(grid_w):
            y_offset_b = grid_y * H_im // grid_h
            x_offset_b = grid_x * W_im // grid_w
            y_offset_e = (grid_y + 1) * H_im // grid_h
            x_offset_e = (grid_x + 1) * W_im // grid_w
            H_im_downsize = y_offset_e - y_offset_b
            W_im_downsize = x_offset_e - x_offset_b

            if (H_im_downsize, W_im_downsize) in downsize_cache:
                image_data_downsize = downsize_cache[(H_im_downsize, W_im_downsize)]
            else:
                image_data_downsize = F.resize(
                    image_data,
                    size=(H_im_downsize, W_im_downsize),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,  # antialiasing for downsizing
                )
                downsize_cache[(H_im_downsize, W_im_downsize)] = image_data_downsize
            if should_hflip[grid_y, grid_x].item():
                image_data_downsize = F.hflip(image_data_downsize)

            if is_pil:
                image_data_output.paste(image_data_downsize, (x_offset_b, y_offset_b))
            else:
                image_data_output[:, y_offset_b:y_offset_e, x_offset_b:x_offset_e] = (
                    image_data_downsize
                )

    datapoint.frames[index].data = image_data_output

    # Step 2: downsize the masks and paste them into the target grid of the mosaic
    for obj in datapoint.frames[index].objects:
        if obj.segment is None:
            continue
        assert obj.segment.shape == (H_im, W_im) and obj.segment.dtype == torch.uint8
        segment_output = torch.zeros_like(obj.segment)

        target_y_offset_b = target_grid_y * H_im // grid_h
        target_x_offset_b = target_grid_x * W_im // grid_w
        target_y_offset_e = (target_grid_y + 1) * H_im // grid_h
        target_x_offset_e = (target_grid_x + 1) * W_im // grid_w
        target_H_im_downsize = target_y_offset_e - target_y_offset_b
        target_W_im_downsize = target_x_offset_e - target_x_offset_b

        segment_downsize = F.resize(
            obj.segment[None, None],
            size=(target_H_im_downsize, target_W_im_downsize),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,  # antialiasing for downsizing
        )[0, 0]
        if should_hflip[target_grid_y, target_grid_x].item():
            segment_downsize = F.hflip(segment_downsize[None, None])[0, 0]

        segment_output[
            target_y_offset_b:target_y_offset_e, target_x_offset_b:target_x_offset_e
        ] = segment_downsize
        obj.segment = segment_output

    return datapoint


class RandomMosaicVideoAPI:
    def __init__(self, prob=0.15, grid_h=2, grid_w=2, use_random_hflip=False):
        self.prob = prob
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_random_hflip = use_random_hflip

    def __call__(self, datapoint, **kwargs):
        if random.random() > self.prob:
            return datapoint

        # select a random location to place the target mask in the mosaic
        target_grid_y = random.randint(0, self.grid_h - 1)
        target_grid_x = random.randint(0, self.grid_w - 1)
        # whether to flip each grid in the mosaic horizontally
        if self.use_random_hflip:
            should_hflip = torch.rand(self.grid_h, self.grid_w) < 0.5
        else:
            should_hflip = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        for i in range(len(datapoint.frames)):
            datapoint = random_mosaic_frame(
                datapoint,
                i,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                target_grid_y=target_grid_y,
                target_grid_x=target_grid_x,
                should_hflip=should_hflip,
            )

        return datapoint

######################################################################################################
class RandomHorizontalFlip_yolo:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint: VideoDatapoint_yolo, **kwargs):
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("before datapoint:", datapoint)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.frames)):
                    datapoint = self.hflip_yolo(datapoint, i)
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print("after datapoint:", datapoint)
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return datapoint
        for i in range(len(datapoint.frames)):
            if random.random() < self.p:
                datapoint = self.hflip_yolo(datapoint, i)
        return datapoint
    
    def hflip_yolo(self, datapoint, index):
        datapoint.frames[index].data = F.hflip(datapoint.frames[index].data)

        # 更新 bboxes
        for i, bbox in enumerate(datapoint.frames[index].bboxes):
            x_center, y_center, width, height = bbox
            # 水平翻轉時，x_center 改變位置
            datapoint.frames[index].bboxes[i] = (
                1 - x_center,  # 翻轉 x_center
                y_center,
                width,
                height,
            )
        return datapoint
    
class RandomResizeAPI_yolo:
    def __init__(
        self, sizes, consistent_transform, max_size=None, square=False, v2=False
    ):
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform
        self.v2 = v2

    def __call__(self, datapoint: VideoDatapoint_yolo, **kwargs):
        if self.consistent_transform:
            size = random.choice(self.sizes)
            # print("datapoint:", datapoint)
            for i in range(len(datapoint.frames)):
                datapoint = self.resize_yolo(
                    datapoint, i, size, self.max_size, square=self.square, v2=self.v2
                )
            return datapoint
        for i in range(len(datapoint.frames)):
            size = random.choice(self.sizes)
            datapoint = self.resize_yolo(
                datapoint, i, size, self.max_size, square=self.square, v2=self.v2
            )
        return datapoint
    
    def resize_yolo(self, datapoint, index, size, max_size=None, square=False, v2=False):
        # size can be min_size (scalar) or (w, h) tuple

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size[::-1]
            else:
                return get_size_with_aspect_ratio(image_size, size, max_size)

        if square:
            size = size, size
        else:
            cur_size = (
                datapoint.frames[index].data.size()[-2:][::-1]
                if v2
                else datapoint.frames[index].data.size
            )
            size = get_size(cur_size, size, max_size)

        old_size = (
            datapoint.frames[index].data.size()[-2:][::-1]
            if v2
            else datapoint.frames[index].data.size
        )
        if v2:
            datapoint.frames[index].data = Fv2.resize(
                datapoint.frames[index].data, size, antialias=True
            )
        else:
            datapoint.frames[index].data = F.resize(datapoint.frames[index].data, size)

        new_size = (
            datapoint.frames[index].data.size()[-2:][::-1]
            if v2
            else datapoint.frames[index].data.size
        )

        # for obj in datapoint.frames[index].objects:
        #     if obj.segment is not None:
        #         obj.segment = F.resize(obj.segment[None, None], size).squeeze()

        h, w = size
        datapoint.frames[index].size = (h, w)

        # if self.consistent_transform:
        #     datapoint.size = (h, w)

        return datapoint

class RandomGrayscale_yolo:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform
        self.Grayscale = T.Grayscale(num_output_channels=3)

    def __call__(self, datapoint: VideoDatapoint_yolo, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for img in datapoint.frames:
                    img.data = self.Grayscale(img.data)
            return datapoint
        for img in datapoint.frames:
            if random.random() < self.p:
                img.data = self.Grayscale(img.data)
        return datapoint

class ColorJitter_yolo:
    def __init__(self, consistent_transform, brightness, contrast, saturation, hue):
        self.consistent_transform = consistent_transform
        self.brightness = (
            brightness
            if isinstance(brightness, list)
            else [max(0, 1 - brightness), 1 + brightness]
        )
        self.contrast = (
            contrast
            if isinstance(contrast, list)
            else [max(0, 1 - contrast), 1 + contrast]
        )
        self.saturation = (
            saturation
            if isinstance(saturation, list)
            else [max(0, 1 - saturation), 1 + saturation]
        )
        self.hue = hue if isinstance(hue, list) or hue is None else ([-hue, hue])

    def __call__(self, datapoint: VideoDatapoint_yolo, **kwargs):
        if self.consistent_transform:
            # Create a color jitter transformation params
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = T.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        for img in datapoint.frames:
            if not self.consistent_transform:
                (
                    fn_idx,
                    brightness_factor,
                    contrast_factor,
                    saturation_factor,
                    hue_factor,
                ) = T.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue
                )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img.data = F.adjust_brightness(img.data, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img.data = F.adjust_contrast(img.data, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img.data = F.adjust_saturation(img.data, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img.data = F.adjust_hue(img.data, hue_factor)
        return datapoint
    
class ToTensorAPI_yolo:
    def __init__(self, v2=False):
        self.v2 = v2

    def __call__(self, datapoint: VideoDatapoint_yolo, **kwargs):
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.to_image_tensor(img.data)
            else:
                img.data = F.to_tensor(img.data)
            # print("max:", torch.max(img.data))
            # print("min:", torch.min(img.data))
        # sys.exit()
        return datapoint

class NormalizeAPI_yolo:
    def __init__(self, mean, std, v2=False):
        self.mean = mean
        self.std = std
        self.v2 = v2

    def __call__(self, datapoint: VideoDatapoint_yolo, **kwargs):
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.convert_image_dtype(img.data, torch.float32)
                img.data = Fv2.normalize(img.data, mean=self.mean, std=self.std)
            else:
                img.data = F.normalize(img.data, mean=self.mean, std=self.std)
            # print("NormalizeAPI_yolo max:", torch.max(img.data))
            # print("NormalizeAPI_yolo min:", torch.min(img.data))
        return datapoint

class RandomAffine_yolo:
    def __init__(self, p=0.5, degrees=10, translate=0.1, scale=0.1, shear=10, border=0, consistent_transform=True):
        """
        YOLO 格式 (x_center, y_center, width, height) 的仿射變換:
        - `degrees`：旋轉角度範圍
        - `translate`：最大平移比例 (相對於影像大小)
        - `scale`：最大縮放比例
        - `shear`：最大錯切角度
        - `border`：影像邊界填充
        - `consistent_transform`：所有影格是否使用相同變換 (適用於影片)
        """
        self.p = p
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.border = border
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint: VideoDatapoint_yolo, **kwargs):
        """
        Args:
            datapoint (VideoDatapoint_yolo): 包含 frames，每個 frame 有 `data` (影像) 和 `bboxes` (YOLO 格式 BBox)
        Returns:
            transformed datapoint
        """
        _, height, width = F.get_dimensions(datapoint.frames[0].data)
        if self.consistent_transform:
            affine_params = self.get_affine_params((height, width))
            apply_affine =  random.random() < self.p

        for i in range(len(datapoint.frames)):
            if not self.consistent_transform:
                affine_params = self.get_affine_params((height, width))
                apply_affine =  random.random() < self.p
            if apply_affine:
                datapoint.frames[i] = self.transform_frame(datapoint.frames[i], affine_params)

        return datapoint

    def get_affine_params(self, img_size):
        """隨機生成仿射變換參數"""
        height, width = img_size

        # 旋轉角度 (-degrees, +degrees)
        angle = random.uniform(-self.degrees, self.degrees)
        # print("angle:", angle)
        # 縮放範圍 (1 - scale, 1 + scale)
        scale = random.uniform(1 - self.scale, 1 + self.scale)
        # 平移範圍 (translate * width, translate * height)
        tx = random.uniform(-self.translate, self.translate) * width
        ty = random.uniform(-self.translate, self.translate) * height
        # 錯切角度 (-shear, +shear)
        shear_x = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        shear_y = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

        return angle, scale, tx, ty, shear_x, shear_y, width, height

    def transform_frame(self, frame, affine_params):
        """
        變換單個 frame，包括影像與 YOLO BBox
        """
        angle, scale, tx, ty, shear_x, shear_y, width, height = affine_params
        M = self.get_affine_matrix(angle, scale, tx, ty, shear_x, shear_y, width, height)

        if isinstance(frame.data, PILImage.Image):  # 如果是 PIL 影像
            frame.data = np.array(frame.data)  # 轉換為 NumPy 陣列
        # print("frame.data.shape:", frame.data.shape)
        # print("type(frame.data):", type(frame.data))
        # # 變換影像
        # if frame.data.ndim == 3 and frame.data.shape[0] in [1, 3]:  # PyTorch 格式 (C, H, W)
        #     frame.data = frame.data.transpose(1, 2, 0)
        # # frame.data = frame.data.astype(np.uint8)
        # print("after frame.data.shape:", frame.data.shape)
        frame.data = cv2.warpAffine(frame.data, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        frame.data = PILImage.fromarray(frame.data)

        # 變換 BBox
        frame.bboxes = self.transform_bboxes(frame.bboxes, M, width, height)

        return frame

    def get_affine_matrix(self, angle, scale, tx, ty, shear_x, shear_y, width, height):
        """計算完整的仿射變換矩陣"""
        R = np.eye(3)  # 旋轉 & 縮放
        R[:2] = cv2.getRotationMatrix2D(center=(width / 2, height / 2), angle=angle, scale=scale)

        T = np.eye(3)  # 平移
        T[0, 2] = tx
        T[1, 2] = ty

        S = np.eye(3)  # 錯切
        S[0, 1] = shear_x
        S[1, 0] = shear_y

        return S @ T @ R  # 順序: 錯切 → 平移 → 旋轉/縮放

    def transform_bboxes(self, bboxes, M, width, height):
        """
        變換 YOLO BBox (x_center, y_center, w, h) 格式
        """
        if len(bboxes) == 0:
            return []

        transformed_bboxes = []
        for bbox in bboxes:
            x_center, y_center, w, h = bbox
            x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height

            # 轉換 BBox 為 corner (x1, y1, x2, y2)
            x1, y1 = x_center - w / 2, y_center - h / 2
            x2, y2 = x_center + w / 2, y_center + h / 2

            # 轉換為齊次座標
            corners = np.array([
                [x1, y1, 1],
                [x2, y1, 1],
                [x1, y2, 1],
                [x2, y2, 1]
            ]).T

            # 應用仿射變換
            transformed_corners = M @ corners
            transformed_corners = transformed_corners[:2, :].T  # 轉回 2D

            # 取得新 BBox
            x_min, y_min = transformed_corners[:, 0].min(), transformed_corners[:, 1].min()
            x_max, y_max = transformed_corners[:, 0].max(), transformed_corners[:, 1].max()

            # 限制範圍，防止超出邊界
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(width, x_max), min(height, y_max)

            # 計算新 BBox
            new_x_center = (x_min + x_max) / 2 / width
            new_y_center = (y_min + y_max) / 2 / height
            new_w = (x_max - x_min) / width
            new_h = (y_max - y_min) / height

            # 檢查 BBox 是否有效
            if new_w > 0.01 and new_h > 0.01:
                transformed_bboxes.append((new_x_center, new_y_center, new_w, new_h))

        return transformed_bboxes

# class RandomAffine_yolo:
#     def __init__(
#         self,
#         degrees,
#         consistent_transform,
#         scale=None,
#         translate=None,
#         shear=None,
#         image_mean=(123, 116, 103),
#         log_warning=True,
#         num_tentatives=1,
#         image_interpolation="bicubic",
#     ):
#         """
#         YOLO BBox 版的 RandomAffine
#         如果 consistent_transform=True，則所有 frames 共享相同的隨機仿射變換。
#         """
#         self.degrees = degrees if isinstance(degrees, list) else ([-degrees, degrees])
#         self.scale = scale
#         self.shear = (
#             shear if isinstance(shear, list) else ([-shear, shear] if shear else None)
#         )
#         self.translate = translate
#         self.fill_img = image_mean
#         self.consistent_transform = consistent_transform
#         self.log_warning = log_warning
#         self.num_tentatives = num_tentatives

#         if image_interpolation == "bicubic":
#             self.image_interpolation = InterpolationMode.BICUBIC
#         elif image_interpolation == "bilinear":
#             self.image_interpolation = InterpolationMode.BILINEAR
#         else:
#             raise NotImplementedError

#     def __call__(self, datapoint: VideoDatapoint_yolo, **kwargs):
#         """
#         對 YOLO dataset 進行仿射變換
#         """
#         for _tentative in range(self.num_tentatives):
#             res = self.transform_datapoint(datapoint)
#             if res is not None:
#                 return res

#         if self.log_warning:
#             logging.warning(
#                 f"Skip RandomAffine for zero-area bbox in first frame after {self.num_tentatives} attempts"
#             )
#         return datapoint

#     def transform_datapoint(self, datapoint: VideoDatapoint_yolo):
#         _, height, width = F.get_dimensions(datapoint.frames[0].data)
#         img_size = [width, height]

#         if self.consistent_transform:
#             # 產生全局隨機仿射變換參數
#             affine_params = T.RandomAffine.get_params(
#                 degrees=self.degrees,
#                 translate=self.translate,
#                 scale_ranges=self.scale,
#                 shears=self.shear,
#                 img_size=img_size,
#             )

#         for img_idx, img in enumerate(datapoint.frames):
#             if not self.consistent_transform:
#                 # 為每個 frame 單獨產生仿射變換參數
#                 affine_params = T.RandomAffine.get_params(
#                     degrees=self.degrees,
#                     translate=self.translate,
#                     scale_ranges=self.scale,
#                     shears=self.shear,
#                     img_size=img_size,
#                 )
#             print("affine_params:", affine_params)
#             # 轉換 BBox
#             transformed_bboxes = self._transform_bboxes(img.bboxes, affine_params, width, height)
            
#             # 轉換影像
#             img.data = F.affine(
#                 img.data,
#                 *affine_params,
#                 interpolation=self.image_interpolation,
#                 fill=self.fill_img,
#             )

#             # 更新 BBoxes
#             img.bboxes = transformed_bboxes

#         return datapoint

#     def _transform_bboxes(self, bboxes, affine_params, img_w, img_h):
#         """
#         使用仿射變換更新 YOLO 格式的 BBox
#         YOLO 格式：(x_center, y_center, width, height) ∈ [0, 1]
#         """
#         if len(bboxes) == 0:
#             return []

#         angle, translations, scale, shear = affine_params
#         tx, ty = translations  # 平移
#         sx, sy = shear  # 錯切

#         # 轉換 BBox 為 (x1, y1, x2, y2) 格式
#         new_bboxes = []
#         for bbox in bboxes:
#             x_center, y_center, width, height = bbox

#             # 轉換到 pixel space
#             x_center *= img_w
#             y_center *= img_h
#             width *= img_w
#             height *= img_h

#             # 取得 BBox 角點
#             x1, y1 = x_center - width / 2, y_center - height / 2
#             x2, y2 = x_center + width / 2, y_center + height / 2

#             # 構建 corner points
#             corners = torch.tensor([
#                 [x1, y1], [x2, y1], [x1, y2], [x2, y2]
#             ], dtype=torch.float)

#             # 建立仿射變換矩陣
#             theta = torch.tensor([
#                 [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), tx * img_w],
#                 [math.sin(math.radians(angle)),  math.cos(math.radians(angle)), ty * img_h]
#             ])

#             # 應用仿射變換
#             ones = torch.ones((corners.shape[0], 1))
#             corners_homo = torch.cat([corners, ones], dim=1).T  # [3, N]
#             transformed_corners = theta @ corners_homo  # [2, N]

#             # 取得變換後的新 BBox
#             x_min, y_min = transformed_corners.min(dim=1).values
#             x_max, y_max = transformed_corners.max(dim=1).values

#             # 轉回 YOLO 格式
#             new_x_center = (x_min + x_max) / 2 / img_w
#             new_y_center = (y_min + y_max) / 2 / img_h
#             new_width = (x_max - x_min) / img_w
#             new_height = (y_max - y_min) / img_h

#             # 限制 BBox 在 [0,1] 範圍內
#             new_bboxes.append((
#                 max(0, min(1, new_x_center)),
#                 max(0, min(1, new_y_center)),
#                 max(0, min(1, new_width)),
#                 max(0, min(1, new_height))
#             ))

#         return new_bboxes

def get_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3):
    # 產生 1D Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss /= gauss.sum()
    # 擴展為 2D kernel
    kernel2d = torch.outer(gauss, gauss)
    kernel2d = kernel2d.expand(channels, 1, kernel_size, kernel_size)
    return kernel2d

def gaussian_blur_batch(images, kernel_size=5, sigma=1.0, frame_transform_rate=0):
    b, c, h, w = images.shape
    kernel = get_gaussian_kernel(kernel_size, sigma, c).to(images.device)
    padding = kernel_size // 2

    if frame_transform_rate == 0:
        blurred = NF.conv2d(images, kernel, padding=padding, groups=c)
        return blurred

    result = []
    for i in range(b):
        frame = images[i:i+1]  # shape = (1, c, h, w)
        if frame_transform_rate == -1:
            # 只對最後一張圖做模糊
            if i == b - 1:
                blurred = NF.conv2d(frame, kernel, padding=padding, groups=c)
                result.append(blurred)
            else:
                result.append(frame)

        elif (i + 1) % frame_transform_rate == 0:
            blurred = NF.conv2d(frame, kernel, padding=padding, groups=c)
            result.append(blurred)
        else:
            result.append(frame)

    return torch.cat(result, dim=0)  # 回傳 shape (b, c, h, w)