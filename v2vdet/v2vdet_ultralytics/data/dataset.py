import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
import math
import random
import pickle

import cv2
from copy import deepcopy
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import time
import os
from multiprocessing import Pool
import torch.multiprocessing as mp
from functools import partial

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import TORCHVISION_0_18
from ultralytics.utils.patches import imread

from ultralytics.data.augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
    LoadVisualPrompt,
)
from ultralytics.data.base import BaseDataset
from ultralytics.data.utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

from v2vdet.v2vdet_ultralytics.data.augment import RandomLoadClass, v2vFormat
from v2vdet.v2vdet_ultralytics.utils.misc import (
    process_segmentation_mask,
    process_segmentation_mask_np,
    extract_and_sample_images,
    random_sample_picture
)
from v2vdet.v2vdet_ultralytics.data.base import BaseDatasetV2
from v2vdet.v2vdet_ultralytics.utils import DEFAULT_CFG
from v2vdet.v2vdet_ultralytics.utils.dataset_misc import (extract_and_resize_masked_region)

from v2vdet.v2vdet_ultralytics.data.augment import v2v_transforms

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"

# from ultralytics.data.dataset import GroundingDataset, YOLODataset


def get_random_indices(numbers: list) -> list:
    unique_numbers = torch.unique(torch.tensor(numbers)).tolist()

    random_indices = {}
    for num in unique_numbers:
        indices = [idx for idx, val in enumerate(numbers) if val == num]
        random_indices[num] = random.choice(indices)

    return random_indices


class YOLODataset(BaseDatasetV2):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, hyp=DEFAULT_CFG, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, hyp=hyp, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path("./labels.cache").

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            v2v_transforms
            # transforms = v8_transforms(self, self.imgsz, hyp)
            transforms = v2v_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """Add texts information for multi-modal model training."""
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp=None):
        """Enhances data transformations with optional text augmentation for multi-modal training."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(-1, RandomLoadText(max_samples=min(self.data["nc"], 80), padding=True))
        return transforms
    


# def process_image(args):
#     """
#     單個圖像的處理函數，供多進程調用
#     args: (img_idx, self_instance) 包含圖像索引和類實例
#     """
#     img_idx, self_instance = args
#     image_and_label = self_instance.get_image_and_label(img_idx)
#     width, height, _ = image_and_label['img'].shape
#     temp_area = [None] * self_instance.data["nc"]
#     templates = [None] * self_instance.data["nc"]

#     for instance_idx, instance in enumerate(image_and_label['instances']):
#         object_cls = int(image_and_label['cls'][instance_idx].item())
#         area = instance.bbox_areas.item()

#         if temp_area[object_cls] is None or area > temp_area[object_cls]:
#             temp_area[object_cls] = area
#             scaled_points = []
#             for s_x, s_y in np.array(instance.segments):
#                 px = int(s_x * width)
#                 py = int(s_y * height)
#                 scaled_points.append([px, py])

#             template_h, template_w = self_instance.template_size
#             background = (
#                 np.zeros((template_h, template_w, 3), dtype=np.uint8)
#                 if random.random() < self_instance.hyp.template_background_ratio
#                 else image_and_label['img']
#             )
#             crop_image = self_instance.extract_and_resize_masked_region(
#                 background, np.array(instance.segments), self_instance.template_size
#             )
#             templates[object_cls] = {
#                 'ori_shape': (width, height),
#                 'bboxes': instance.bboxes,
#                 'normalized': instance.normalized,
#                 'segments': np.array(instance.segments),
#                 'crop_image': crop_image
#             }

#     return img_idx, templates

class V2V_Dataset(YOLODataset):
    
    def __init__(self, *args, 
                 hyp=DEFAULT_CFG, 
                 data=None, 
                 task="detect", 
                 template_size=(224, 224), 
                 template_cache=False,
                 **kwargs):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        super().__init__(*args, hyp=hyp, data=data, task=task, **kwargs)
        
        self.hyp = hyp
        self.template_size = template_size
        # self.create_template()

        self.template_transforms = self.build_template_transforms(hyp=hyp)
        self.build_template_list()
        self.create_template()
        
        # self.template_npy_files = [str(Path(f).with_suffix('')) + "_template.npy" for f in self.im_files]
        # self.image_and_label = self.get_image_and_label(0)
        # self.template_npy_files = [[] for _ in range(len(self.labels))]
        # self.template_cache = template_cache.lower() if isinstance(template_cache, str) else "ram" if template_cache is True else None
        # if self.template_cache == "ram" and self.check_cache_ram():
        #     if hyp.deterministic:
        #         LOGGER.warning(
        #             "WARNING ⚠️ cache='ram' may produce non-deterministic training results. "
        #             "Consider cache='disk' as a deterministic alternative if your disk space allows."
        #         )
        #     self.cache_templates()
        # elif self.template_cache == "disk" and self.check_cache_disk():
        #     self.cache_templates()

    def update_labels_info(self, label):
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        # labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        labels["sampled_cls"] = np.array([n for n in range(self.data["nc"])], dtype=int)
        labels["label2ids"] = [None]
        labels["nc"] = self.data["nc"] if not self.augment else 80
        return labels

    def build_transforms(self, hyp=None):
        """Enhances data transformations with optional text augmentation for multi-modal training."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(-1, RandomLoadClass(max_samples=min(self.data["nc"], 80), padding=True))
        return transforms

    def build_template_transforms(self, hyp=None):
        """Augumentation on template images.
            I think it is easier to do augumentation first, than crop into template images. 
        """
        
        if hyp is None:
            hyp = self.hyp
            
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.copy_paste = 0.0

        if self.augment:
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            self.hyp.template_background_ratio = hyp.template_background_ratio = 0.0
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

        transforms.append(
            v2vFormat(
                bbox_format="xywh",
                normalize=True,
                return_mask=True,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=1,
                mask_overlap=False,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )

        return transforms

    def create_template(self, each_cls_amt=10):
        # Create a template for each class, we will choice the maximum area of the bounding box
        
        total_cls = self.data["nc"] if self.augment is False else 80
        self.templates = [[] for _ in range(total_cls)]
        
        cache_dir = os.path.dirname(self.im_files[0])

        # Read data
        cache_exists_flag = False
        train_or_val = "train" if self.augment else "val"
        cache_name = f"{cache_dir}_{train_or_val}_template.pkl"
        if os.path.exists(cache_name):
            with open(cache_name, 'rb') as file:
                templates = pickle.load(file)
            
            for cls_templates in templates:
                if not cls_templates:
                    continue
            
            self.templates = templates
            cache_exists_flag = True
            LOGGER.info(f"{self.prefix}Templates loaded from {cache_name}")
        
        if cache_exists_flag is False:
            LOGGER.info(f"{self.prefix}Template cache no exists or there were some issues, creating new templates.")
            for img_idx in TQDM(range(total_cls), desc="Creating templates", disable=LOCAL_RANK > 0):
                class_idx = random.randint(0, self.data["nc"]-1) if self.augment else img_idx
                cls_template_list = self.template_list[class_idx]
                sample_size = min(each_cls_amt, len(cls_template_list))
                sampled_templates = random.sample(cls_template_list, sample_size)
                for sampled_template in sampled_templates:
                    im_files_idx = sampled_template['im_files_idx']
                    single_template = self.load_template(im_files_idx)
                    # self.templates[img_idx].append(single_template[class_idx][im_files_idx]['crop_image'])
                    if class_idx in single_template['class_idx']:
                        class_info_idx = single_template['class_idx'].index(class_idx)
                        self.templates[img_idx].append(single_template['class_info'][class_info_idx]['crop_image'])
                    else:
                        raise ValueError(f"Class {class_idx} not found in the template list for image index {im_files_idx}.")

            with open(cache_name, 'wb') as file:
                pickle.dump(self.templates, file)
            
            LOGGER.info(f"{self.prefix}Templates created and saved to {cache_name}")

        return self.templates
            
    def build_template_list(self):
        self.template_list = [[] for _ in range(self.data['nc'])]
        for im_files_idx, label in TQDM(enumerate(self.labels), desc="Building template list", disable=LOCAL_RANK > 0):
            
            unique_classes = np.unique(label['cls'])
            
            file_path = f"{os.path.dirname(label['im_file'])}/template_np"
            if os.path.exists(file_path) is False:
                os.makedirs(file_path)
            basename = os.path.basename(label['im_file'])
            # file_name, file_ext = os.path.splitext(basename)
            
            for class_idx in unique_classes:
                int_class_idx = int(class_idx)
                # np_folder_path = f"{file_path}/{int_class_idx}"
                # if os.path.exists(np_folder_path) is False:
                #     os.makedirs(np_folder_path)
                info_dict = {
                    'im_file': label['im_file'],
                    # 'np_file': f"{np_folder_path}/{file_name}.npy",
                    'im_files_idx': im_files_idx,
                }
                self.template_list[int(class_idx)].append(info_dict)
            
        
        return self.template_list
    
    def load_template(self, img_idx):
   
        image_and_label = self.get_image_and_label(img_idx)
        # _ = self.transforms(image_and_label)
        width, height, _ = image_and_label['img'].shape
        # templates = [[] for _ in range(self.data["nc"])]
        templates={
            'img_idx': img_idx,
            'file_name': self.im_files[img_idx],
            'ori_shape': (width, height),
            'class_idx': [],
            'class_info': []
        }
        
        for instance_idx, instance in enumerate(image_and_label['instances']):
            temp_area = [None] * self.data["nc"]
            object_cls = int(image_and_label['cls'][instance_idx].item())
            area = instance.bbox_areas.item()
            
            # We only save the largest template for each class
            if temp_area[object_cls] is None or area > temp_area[object_cls]:
                temp_area[object_cls] = area
                scaled_points = []
                for s_x, s_y in np.array(instance.segments):
                    px = int(s_x * width)
                    py = int(s_y * height)
                    scaled_points.append([px, py])
                
                # * Adjust the background is all black or original image's background
                template_h, template_w = self.template_size
                
                # Select the background of the pasted template crop image is black background of detected image's
                if random.random() < self.hyp.template_background_ratio:
                    background = np.zeros((template_h, template_w, 3), dtype=np.uint8) 
                else:
                    background = deepcopy(image_and_label['img'])
                
                crop_image = extract_and_resize_masked_region(background, np.array(instance.segments), self.template_size)
                
                if object_cls in templates['class_idx']:
                    templates_idx = templates['class_idx'].index(object_cls)
                    templates['class_info'][templates_idx] = {
                        # 'img_idx': img_idx,
                        # 'file_name': self.im_files[img_idx],
                        # 'ori_shape': (width, height),
                        'bboxes': instance.bboxes,
                        'normalized': instance.normalized,
                        'segments': np.array(instance.segments),
                        'crop_image': crop_image
                    }
                else:
                    templates['class_idx'].append(int(object_cls))
                    templates['class_info'].append({
                        'bboxes': instance.bboxes,
                        'normalized': instance.normalized,
                        'segments': np.array(instance.segments),
                        'crop_image': crop_image
                    })

                # templates[object_cls]={
                #     'img_idx': img_idx,
                #     'file_name': self.im_files[img_idx],
                #     'ori_shape': (width, height),
                #     'bboxes': instance.bboxes,
                #     'normalized': instance.normalized,
                #     'segments': np.array(instance.segments),
                #     'crop_image': crop_image
                # }
        
        return templates
    
    # def cache_templates(self):
    #     """Cache templates to memory or disk for faster training."""
    #     b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
    #     fcn, storage = (self.cache_templates_to_disk, "Disk") if self.template_cache == "disk" else (self.load_template, "RAM")
    #     self.templates = [{} for _ in range(self.data['nc'])]
    #     with ThreadPool(NUM_THREADS) as pool:
    #         results = pool.imap(fcn, range(len(self.im_files)))
    #         pbar = TQDM(enumerate(results), total=len(self.im_files), disable=LOCAL_RANK > 0)
    #         for i, x in pbar:
    #             if self.template_cache == "disk":
    #                 for npy_files in self.template_npy_files:
    #                     for npy_file in npy_files:
    #                         b += Path(npy_file).stat().st_size
    #             else:  # 'ram'
    #                 for each_cls_idx, each_cls in enumerate(iterable=x):
    #                     if each_cls is not None:
    #                         self.templates[each_cls_idx].update(each_cls)
    #                 b += np.array(self.templates).nbytes
    #             pbar.desc = f"{self.prefix}Caching templates ({b / gb:.1f}GB {storage})"
    #         pbar.close()

    # def cache_templates_to_disk(self, img_idx):
    #     """Save template list as an *.npy file for faster loading."""
    #     f = self.im_files[img_idx]
    #     file_path = f'{os.path.dirname(f)}/template_np'
    #     if os.path.exists(file_path) is False:
    #         os.makedirs(file_path)
    #     basename = os.path.basename(f)
    #     file_name, file_ext = os.path.splitext(basename)
        
    #     templates = self.load_template(img_idx)
    #     for template_idx, template in enumerate(templates):
    #         if template != {}:
                
    #             cls_file_path = f"{file_path}/{template_idx}" 
    #             if os.path.exists(cls_file_path) is False:
    #                 os.makedirs(cls_file_path)
                    
    #             f = Path(f"{cls_file_path}/{file_name}.npy")

    #             crop_img = templates[template_idx][img_idx]['crop_image']
    #             if not f.exists():
    #                 np.save(f.as_posix(), crop_img, allow_pickle=False)

    #             self.template_npy_files[img_idx].append(f.as_posix())
    #     del templates
                
    def __getitem__(self, index):
        """Return transformed label information for given index."""
        get_image_and_label = self.get_image_and_label(index)
        class_np = np.unique(get_image_and_label['cls'])
        image_and_label = self.transforms(get_image_and_label)
        
        template = self.load_template(index)
        
        image_and_label['train'] = True if self.augment else False
        image_and_label['template_list'] = deepcopy(self.template_list)
        image_and_label['nc'] = self.data['nc'] if not self.augment else 80
        # image_and_label['cls'] = sorted(list(int(g) for g in class_np)) if not self.augment else image_and_label['pos_labels']
        # image_and_label['cls'] = 
        image_and_label['class'] =  sorted(list(int(g) for g in class_np)) if not self.augment else image_and_label['pos_labels']
        image_and_label['template'] = template
        image_and_label['templates_crop_images'] = self.templates
        
        return image_and_label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        
        template_list = batch[0]['template_list']
        nc = batch[0]['nc']
        # for i in range(len(template_list)):
        all_templates_crop_images = batch[0]['templates_crop_images']
        
        templates_crop_images = []
        '''
         all_templates_crop_images is a list that stored each class's crop image, up to 10 crop images for each class.
        '''
        for templates_crop_image in all_templates_crop_images:
            if templates_crop_image:
                random_element = random.choice(templates_crop_image)
                templates_crop_images.append(random_element)
            else:
                '''
                In some really small dataset (especially those only for testing if the training/val code can work), there may happened that there is no template crop image for this class.
                In this case, we need to add a dummy image to the list, which is all black.
                This is to avoid the error when we do the collate_fn, which will cause the batch size not equal.
                '''
                templates_crop_images.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img"}:
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        
        template_feats_list = []
        for b_idx, b in enumerate(batch):
            each_sample_template = deepcopy(templates_crop_images)
            # class_list = b['class_list']
            template = batch[b_idx]['template']
            for c_idx in list(b['class']):
                
                # Since we will do augumentation in training, which means the class will reduce (e.g. 1203->80). Therefore, we need to map the class index to the template class index.
                if b['train']:
                    class_idx = b['label2ids'][c_idx]
                # No need to do that in eval mode.
                else: 
                    class_idx = c_idx
                
                if c_idx in template['class_idx']:
                    info_idx = template['class_idx'].index(c_idx)
                    each_sample_template[class_idx] = template['class_info'][info_idx]['crop_image']

            template_feats_list.extend(each_sample_template)
        
        template_feats_np = np.stack(template_feats_list, axis=0)
        new_batch['template_feats']  = torch.from_numpy(template_feats_np).permute(0, 3, 1, 2)
        
        return new_batch
    
    def paste_image_on_background(self, foreground, background, position=None, resized=(224, 224)):
        """
        將前景圖片貼到背景圖片上
        
        參數:
            foreground: numpy array，前景圖片（已經調整大小的物體，含透明/黑色背景）
            background: numpy array，背景圖片，尺寸為 224x224
            position: (x, y) tuple，貼上位置的左上角座標，默認為居中
            resized: (x, y) tuple，調整後的大小，默認為 (224, 224)
            
        返回:
            合成後的圖片
        """
        
        input_is_torch = False
        if isinstance(foreground, torch.Tensor):
            input_is_torch = True
            # 保存原始設備，用於最終轉換回 torch
            device = foreground.device
            # 如果是 torch.tensor，先轉換為 numpy
            if foreground.dim() == 4:  # 批次張量 [B,C,H,W]
                foreground = foreground.squeeze(0)  # 移除批次維度
            if foreground.dim() == 3:  # [C,H,W] 格式
                # 轉換為 HWC 格式
                foreground = foreground.permute(1, 2, 0)
            # 將 torch tensor 轉換為 numpy
            foreground = foreground.detach().cpu().numpy()
        
        if isinstance(background, torch.Tensor):
            input_is_torch = True
            device = background.device
            if background.dim() == 4:
                background = background.squeeze(0)
            if background.dim() == 3:
                background = background.permute(1, 2, 0)
            background = background.detach().cpu().numpy()
        
        # 處理浮點數範圍
        if foreground.dtype == np.float32 or foreground.dtype == np.float64:
            if foreground.max() <= 1.0:
                foreground = (foreground * 255).astype(np.uint8)
        
        if background.dtype == np.float32 or background.dtype == np.float64:
            if background.max() <= 1.0:
                background = (background * 255).astype(np.uint8)
    
        
        # 確保背景圖片的大小為 224x224
        if background.shape[:2] != resized:
            background = cv2.resize(background, resized)
        
        # 確保前景圖片的大小為 224x224
        if foreground.shape[:2] != resized:
            foreground = cv2.resize(foreground, resized)
        
        # 創建前景的遮罩 (非黑色部分)
        # 假設背景是黑色 (0, 0, 0)
        mask = np.any(foreground > 0, axis=2).astype(np.uint8) * 255
        
        # 如果沒有指定位置，則居中放置
        if position is None:
            # 前景已經是 224x224，所以位置為 (0, 0)
            position = (0, 0)
        
        # 複製背景以創建結果圖片
        result = background.copy()
        
        # 計算放置區域
        x, y = position
        h, w = foreground.shape[:2]
        
        # 確保不超出範圍
        if x + w > background.shape[1] or y + h > background.shape[0]:
            w = min(w, background.shape[1] - x)
            h = min(h, background.shape[0] - y)
        
        # 提取相應區域
        roi = result[y:y+h, x:x+w]
        
        # 創建前景和背景的遮罩
        mask_roi = mask[0:h, 0:w]
        mask_inv = cv2.bitwise_not(mask_roi)
        
        # 將前景和背景合併
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(foreground[0:h, 0:w], foreground[0:h, 0:w], mask=mask_roi)
        
        # 組合前景和背景
        result[y:y+h, x:x+w] = cv2.add(bg, fg)
        
        return result
    


class backup_V2V_COCO_Dataset(YOLODataset):

    """
    For vision2vision template matching dataset.
    """

    def __init__(
        self, *args, data=None, task="detect", hyp=DEFAULT_CFG, **kwargs
    ):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        super().__init__(*args, data=data, task=task, **kwargs)
        self.nc = min(self.data["nc"], 80) if self.augment else self.data["nc"]
        self.template_transforms = self.build_template_transforms(hyp)

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        get_image_and_label = self.get_image_and_label(index)
        batch = self.transforms(get_image_and_label)

        w, h = batch["resized_shape"]

        # Got random indices for each class, for cropping template images
        crop_template_lists = [-1] * self.nc

        unique_values = torch.unique(batch["cls"])
        for value in unique_values:
            indices = torch.where(batch["cls"].squeeze() == value)[0]
            # chosen_index
            crop_template_lists[int(value.item())] = indices[
                torch.randint(len(indices), (1,))
            ].item()
        batch["crop_template_lists"] = crop_template_lists

        random_img_file = random.choice(self.im_files)
        random_img = cv2.imread(random_img_file)
        random_img_tensor = torch.from_numpy(random_img).float() / 255.0
        random_img_tensor = random_img_tensor.permute(2, 0, 1)
        batch["random_img"] = random_img_tensor

        return {"batch": batch}

    @staticmethod
    def collate_fn(batch_dict: list):
        """Collates data samples into batches."""
        batch = [b["batch"] for b in batch_dict]
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
            if k == "crop_template_lists":
                template_feats_tensor = []
                for batch_idx, vs in enumerate(value):
                    cropped_images = []
                    for v in vs:
                        if v == -1:

                            random_img = batch[random.randint(0, len(batch) - 1)][
                                "random_img"
                            ]
                            h, w = random_img.shape[1], random_img.shape[2]
                            bbox_x = max(0, random.randint(
                                0, random_img.shape[1] - 90))
                            bbox_y = max(0, random.randint(
                                0, random_img.shape[2] - 90))
                            bbox_h = 80
                            bbox_w = 80

                            # bbox_h = min(bbox_h, h-30)
                            # bbox_w = min(bbox_w, w-30)
                            bbox_x = min(bbox_x, w - bbox_w - 1)
                            bbox_y = min(bbox_y, h - bbox_h - 1)
                            # y1, x1 = max(0, random.randint(0, random_img.shape[1]-90)), max(0, random.randint(0, random_img.shape[2]-90))
                            # y2, x2 = min(y1+80, random_img.shape[1]), min(x1+80, random_img.shape[2])
                            # cropped_tensor = random_img[y1:y2, x1:x2]
                            cropped_images.append(
                                F.crop(
                                    random_img,
                                    top=bbox_y,
                                    left=bbox_x,
                                    height=bbox_w,
                                    width=bbox_h,
                                )
                            )

                        else:
                            w, h = batch[batch_idx]["resized_shape"]
                            box = batch[batch_idx]["bboxes"][v].tolist()
                            bbox_x = int((box[0] - box[2] / 2) * w)
                            bbox_y = int((box[1] - box[3] / 2) * h)
                            bbox_h = int(box[2] * w)
                            bbox_w = int(box[3] * h)

                            bbox_h = min(bbox_h, h - 30)
                            bbox_w = min(bbox_w, w - 30)
                            bbox_x = min(bbox_x, w - bbox_w)
                            bbox_y = min(bbox_y, h - bbox_h)

                            cropped_images.append(
                                F.crop(
                                    batch[batch_idx]["img"],
                                    top=bbox_y,
                                    left=bbox_x,
                                    height=bbox_w,
                                    width=bbox_h,
                                )
                            )

                    template_feats_tensor.append(
                        torch.stack(
                            [
                                torch.nn.functional.interpolate(
                                    input=p.unsqueeze(0),
                                    size=(224, 224),
                                    mode="bilinear",
                                    align_corners=False,
                                ).squeeze(0)
                                for p in cropped_images
                            ]
                        )
                    )

                new_batch["template_feats"] = torch.stack(
                    template_feats_tensor, 0)

        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            # add target image index for build_targets()
            new_batch["batch_idx"][i] += i

        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)

        return new_batch

    def _load_template_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(
                        f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}"
                    )
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                im = cv2.imread(f)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (
                        min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz),
                    )
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (
                h0 == w0 == self.imgsz
            ):  # resize by stretching image to square imgsz
                im = cv2.resize(
                    im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
                )

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def get_template_image_and_label(self, index):
        """Get and return label information from the dataset. For template sample prepare."""
        label = deepcopy(
            self.labels[index]
            # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        )
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = (
            self._load_template_image(index)
        )
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label=label)

    def build_template_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""

        transforms = Compose(
            [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
        )
        transforms.append(
            # Format(
            #     bbox_format="xywh",
            #     normalize=True,
            #     return_mask=self.use_segments,
            #     return_keypoint=self.use_keypoints,
            #     return_obb=self.use_obb,
            #     batch_idx=True,
            #     mask_ratio=hyp.mask_ratio,
            #     mask_overlap=hyp.overlap_mask,
            #     bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            # )
            transform=Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=0.0,  # only affect training.
            )
        )
        return transforms

    def build_transforms(self, hyp=None):
        """Enhances data transformations with optional text augmentation for multi-modal training."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(
                -1, RandomLoadClass(max_samples=min(self.data["nc"], 80), padding=True)
            )
        return transforms


class SA_V_V2VDataset(V2V_Dataset):
    """
    For SA_V dataset's vision2vision dataset.
    """

    random.seed(878787)

    def __init__(
        self,
        *args,
        data=None,
        task="detect",
        train=False,
        num_search_range=10,
        **kwargs,
    ):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        self.train = train
        self.num_search_range = num_search_range
        self.nc = data["nc"]
        self.target_size = 224

        if train:
            ann_file = f"DATASET/SA_V/annotation/sav_train_annotation.json"
        else:
            ann_file = f"DATASET/SA_V/annotation/sav_val_annotation.json"
        with open(ann_file, "r", encoding="utf-8") as f:
            self.coco_annotation = json.load(f)

        # self.image_dict = {f'{item['file_name'].split("/")[-2]}/{item['file_name'].split("/")[-1]}': item for item in self.coco_annotation['images']}
        self.image_dict = {
            f"{item['file_name'].split('/')[-2]}/{item['file_name'].split('/')[-1]}": item
            for item in self.coco_annotation["images"]
        }

        # self.ann_dict = {item['image_id']: item for item in self.coco_annotation['annotations']}
        self.ann_dict = defaultdict(list)
        for item in self.coco_annotation["annotations"]:
            self.ann_dict[item["image_id"]].append(item)

        super().__init__(*args, data=data, task=task, **kwargs)

    def __getitem__(self, index):
        """
        返回指定索引的轉換後標籤信息。

        這個方法從資料集中獲取查詢圖像和模板圖像，處理它們的標籤信息，
        並應用轉換以生成可用於訓練或評估的批次。

        參數:
            index (int): 要獲取的元素索引

        返回:
            dict: 包含批次數據、模板數據和元數據的字典
        """
        # 獲取查詢圖像和標籤
        query_image_and_label = self.get_image_and_label(index)

        # 從查詢圖像和標籤中獲取模板圖像和標籤
        template_image_and_label, template_ann_info, query_video_name = self._find_template_with_annotations(
            index, query_image_and_label
        )

        # 按類別隨機選擇一個目標，確保每個類別都有代表
        template_ann_info = self._select_one_instance_per_category(template_ann_info)

        # 處理分割遮罩並裁剪圖像
        cropped_images = self._process_segmentation_masks(template_image_and_label, template_ann_info)

        # 獲取隨機圖像（用於對比或數據增強）
        random_image_and_label, random_video_name = self._get_random_image(index)

        # 應用轉換
        batch = self.transforms(query_image_and_label)
        template = self.template_transforms(template_image_and_label)

        # 將隨機圖像和裁剪圖像添加到模板中
        template["random_img"] = {
            "video_name": random_video_name,
            "img": random_image_and_label["img"],
        }

        template["cropped_images"] = np.stack(cropped_images, axis=0)
        batch["cropped_images"] = torch.from_numpy(
            np.stack(cropped_images, axis=0)
        ).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        if len(cropped_images) != len(batch['cls']):
            breakpoint()

        return {
            "batch": batch,
            "template": template,
            "video_name": int(query_video_name),
            "nc": self.nc,
            "target_size": self.target_size,
            "is_train": self.train,
        }

    def _extract_video_name(self, image_path, is_train=None):
        """
        從圖像路徑中提取視頻名稱。

        參數:
            image_path (str): 圖像文件路徑
            is_train (bool, optional): 是否為訓練模式，如果為None則使用self.train

        返回:
            str: 視頻名稱
        """
        is_train = self.train if is_train is None else is_train
        temp_list = image_path.split("/")[-1].split("_")

        if is_train and len(temp_list) > 1:  # 正常訓練
            return temp_list[1]
        else:  # 驗證或測試
            return image_path.split("/")[-2].split("_")[1]

    def _find_template_with_annotations(self, index, query_image_and_label):
        """
        從同一視頻的其他幀中找到模板圖像和標籤。

        參數:
            index (int): 查詢圖像的索引
            query_image_and_label (dict): 查詢圖像和標籤

        返回:
            tuple: (template_image_and_label, template_ann_info, query_video_name)
        """
        template_ann_info = None
        query_video_name = self._extract_video_name(query_image_and_label["im_file"])

        # 嘗試找到具有有效註釋的模板
        max_attempts = 10  # 設定最大嘗試次數以避免無限循環
        attempts = 0

        while template_ann_info is None and attempts < max_attempts:
            attempts += 1

            # 在搜索範圍內隨機選擇索引
            min_index = max(0, index - self.num_search_range)
            max_index = min(len(self.labels) - 1, index + self.num_search_range)
            template_image_and_label = self.get_template_image_and_label(
                random.randint(min_index, max_index)
            )

            # 確保模板和查詢來自同一視頻
            template_video_name = self._extract_video_name(template_image_and_label["im_file"])

            if query_video_name == template_video_name:
                # 構建模板圖像的文件名
                template_file_name = f"{template_image_and_label['im_file'].split('/')[-2]}/{template_image_and_label['im_file'].split('/')[-1]}"
                image_id_info = self.image_dict.get(template_file_name)

                if image_id_info is None:
                    raise KeyError(f"找不到圖像ID信息。可能與__init__類中的ann_file相關。")

                # 嘗試獲取註釋信息
                template_ann_info = self.ann_dict.get(image_id_info["id"])

                # 如果找不到註釋，嘗試在相同的1000幀序列中查找
                if template_ann_info is None:
                    first_frame_idx = image_id_info["id"] // 1000 * 1000

                    for i in range(first_frame_idx, first_frame_idx + 1000):
                        template_ann_info = self.ann_dict.get(i)
                        if template_ann_info is not None:
                            break

        # 如果仍然找不到，使用查詢圖像本身作為模板（備選方案）
        if template_ann_info is None:
            template_image_and_label = query_image_and_label
            template_file_name = f"{query_image_and_label['im_file'].split('/')[-2]}/{query_image_and_label['im_file'].split('/')[-1]}"
            image_id_info = self.image_dict.get(template_file_name)
            template_ann_info = self.ann_dict.get(image_id_info["id"])

            if template_ann_info is None:
                raise ValueError(f"無法找到索引 {index} 的有效模板註釋。")

        return template_image_and_label, template_ann_info, query_video_name

    def _select_one_instance_per_category(self, annotations):
        """
        從每個類別中隨機選擇一個實例。

        參數:
            annotations (list): 註釋信息列表

        返回:
            list: 選擇後的註釋信息列表
        """
        # 按類別ID分組
        category_dict = {}
        for item in annotations:
            category_id = item["category_id"]
            if category_id not in category_dict:
                category_dict[category_id] = []
            category_dict[category_id].append(item)

        # 從每個類別中隨機選擇一個實例
        selected_annotations = [random.choice(items) for items in category_dict.values()]

        # 按類別ID排序，確保輸出順序一致
        return sorted(selected_annotations, key=lambda x: x["category_id"])

    def _process_segmentation_masks(self, template_image_and_label, template_ann_info):
        """
        處理分割遮罩並裁剪圖像。

        參數:
            template_image_and_label (dict): 模板圖像和標籤
            template_ann_info (list): 模板註釋信息

        返回:
            list: 裁剪的圖像列表
        """
        cropped_images = []

        for seg in template_ann_info:
            _, seg_cropped_image = process_segmentation_mask_np(
                seg["segmentation"], original_image=template_image_and_label["img"]
            )
            cropped_images.append(seg_cropped_image)

        return cropped_images

    def _get_random_image(self, index):
        """
        獲取隨機圖像和視頻名稱。

        參數:
            index (int): 當前索引

        返回:
            tuple: (random_image_and_label, random_video_name)
        """
        min_index = max(0, index - self.num_search_range)
        max_index = min(len(self.labels) - 1, index + self.num_search_range)

        random_image_and_label = self.get_template_image_and_label(
            random.randint(min_index, max_index)
        )

        random_video_name = int(self._extract_video_name(random_image_and_label["im_file"]))

        return random_image_and_label, random_video_name

    @staticmethod
    def collate_fn(batch_dict: list):
        """
        將數據樣本整合到批次中。

        參數:
            batch_dict (list): 包含樣本數據的字典列表

        返回:
            dict: 處理後的批次數據
        """
        # 1. 收集批次中的基本數據
        batch = [b["batch"] for b in batch_dict]
        template = [b["template"] for b in batch_dict]
        video_name = [b["video_name"] for b in batch_dict]
        video_name_unique = np.unique(np.array(video_name)).tolist()
        nc = batch_dict[0]["nc"]
        random_image = [b["template"]["random_img"] for b in batch_dict]
        target_size = batch_dict[0]["target_size"]
        # bbox 格式: xywh 已標準化, x 和 y 是框的中心 (YOLO 格式)

        # 2. 從重複樣本中隨機選擇樣本
        video_name_idx_pair_dict = get_random_indices(video_name)

        # 3. 處理類別數量
        template_nc = 0
        init_origin_class_list = []
        init_video_name_idx_list = []

        # 收集所有模板類別和對應的視訊名稱
        for key, value in video_name_idx_pair_dict.items():
            template_nc += len(torch.unique(template[value]["cls"]))
            current_classes = [int(cls) for cls in template[value]["cls"]]
            init_origin_class_list.append(current_classes)
            init_video_name_idx_list.append([key] * len(current_classes))

        # 展平列表
        origin_class_list = sum(init_origin_class_list, [])
        video_name_idx_list = sum(init_video_name_idx_list, [])

        # 處理類別數量超出或不足的情況
        if template_nc <= nc:
            # 類別數量不足時，用 -1 填充
            origin_class_list.extend([-1] * (nc - len(origin_class_list)))
            video_name_idx_list.extend([-1] * (nc - len(video_name_idx_list)))
        else:
            # 類別數量超出時，截取前 nc 個
            origin_class_list = origin_class_list[:nc]
            video_name_idx_list = video_name_idx_list[:nc]

        # 4. 隨機打亂類別索引
        zipped = list(zip(origin_class_list, video_name_idx_list))
        random.shuffle(zipped)
        s1, s2 = zip(*zipped)
        shuffled_origin_class_list = np.array(list(s1))
        shuffled_video_name_idx_list = np.array(list(s2))
        shuffled_result = [
            list(pair)
            for pair in zip(shuffled_video_name_idx_list, shuffled_origin_class_list)
        ]

        # 5. 處理隨機圖像
        random_image_list = []
        for rnd_img in random_image:
            if rnd_img["video_name"] not in video_name_unique:
                random_image_list.append(rnd_img)

        # 如果沒有合適的隨機圖像，使用原始隨機圖像列表
        if len(random_image_list) == 0:
            random_image_list = random_image

        # 6. 處理查詢圖像並構建新批次
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))

        # 初始化 valid_mask 為 None
        valid_mask = None

        for i, k in enumerate(keys):
            value = values[i]

            # 堆疊圖像
            if k == "img":
                value = torch.stack(value, 0)

            # 處理類別信息
            if k == "cls":
                value_cls_list = []
                for v_idx, v in enumerate(value):
                    valid_indices = np.where(
                        shuffled_video_name_idx_list == video_name[v_idx]
                    )[0]
                    temp_origin_class_list = np.array(
                        [int(vv[0]) for vv in v.tolist()])
                    cls_temp = shuffled_origin_class_list[valid_indices]

                    mapped_indices = []
                    for cls in temp_origin_class_list:
                        matches = np.where(cls_temp == cls)[0]
                        if len(matches) > 0:
                            mapped_indices.append(valid_indices[matches[0]])
                        else:
                            # 類別不在模板中，標記為 -1
                            mapped_indices.append(-1)

                    value_cls_list.append(
                        torch.tensor(mapped_indices,
                                     dtype=torch.float32).unsqueeze(-1)
                    )

                value = value_cls_list

            # 處理特殊類型的數據
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
                if k == "cls":  # 移除不存在於模板中的類別 (-1)
                    valid_mask = value != -1
                    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
                    valid_mask = valid_mask.reshape(-1)
                    value = value[valid_mask]
                elif valid_mask is not None:
                    # 僅當 valid_mask 已被定義時才應用
                    value = value[valid_mask]

            new_batch[k] = value

        # 更新批次索引
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # 為 build_targets() 添加目標圖像索引

        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        if valid_mask is not None:
            new_batch["batch_idx"] = new_batch["batch_idx"][valid_mask]

        # 7. 創建裁剪的模板圖像
        template_feats = []

        for video_name_cls_pair in shuffled_result:
            video_name_id, origin_cls = video_name_cls_pair

            if video_name_id == -1:  # 隨機裁剪，非模板
                # 隨機選擇圖像並裁剪
                random_image_pick = random.choice(random_image_list)
                height, width = 80, 80

                # 確保裁剪區域在有效範圍內
                max_x = min(random_image_pick["img"].shape[0] - height, height)
                max_y = min(random_image_pick["img"].shape[1] - width, width)
                min_x = random.randint(0, max(0, max_x))
                min_y = random.randint(0, max(0, max_y))

                is_train = batch_dict[0]["is_train"]

                '''
                if is_train:
                    # 在訓練模式下裁剪實際圖像
                    random_image_tensor = torch.from_numpy(
                        random_image_pick["img"])
                    random_image_tensor = random_image_tensor.permute(2, 0, 1)
                    cropped_tensor = F.crop(
                        random_image_tensor,
                        top=min_y,
                        left=min_x,
                        height=height,
                        width=width,
                    )

                    # 檢查裁剪結果
                    if cropped_tensor.shape[1] < 5 or cropped_tensor.shape[2] < 5:
                        LOGGER.warning(
                            f"裁剪圖像過小: {cropped_tensor.shape}，使用零張量替代"
                        )
                        cropped_tensor = torch.zeros(
                            (3, height, width), dtype=torch.uint8
                        )
                else:
                    # 測試模式使用零張量
                    cropped_tensor = torch.zeros(
                        (3, 224, 224), dtype=torch.uint8)
                '''
                # We use zero tensor for all non-template class (black image)
                cropped_tensor = torch.zeros(
                    (3, 224, 224), dtype=torch.uint8)


                template_feats.append(cropped_tensor)

            else:  # 使用模板
                template_idx = video_name_idx_pair_dict[video_name_id]
                pick_template = template[template_idx]
                pick_template_class_list = [
                    int(pp[0]) for pp in pick_template["cls"]]

                try:
                    class_idx = pick_template_class_list.index(origin_cls)
                except ValueError:
                    class_idx = -1
                    LOGGER.warning(
                        f"模板中找不到類別 {origin_cls}，使用第一個模板圖像替代"
                    )

                if class_idx == -1:
                    # 無法找到匹配的類別時，使用第一個模板圖像
                    if len(pick_template["cropped_images"]) > 0:
                        template_feats.append(
                            pick_template["cropped_images"][0])
                    else:
                        LOGGER.warning(
                            f"模板 {template_idx} 沒有裁剪圖像，使用零張量替代"
                        )
                        template_feats.append(
                            torch.zeros((3, 224, 224), dtype=torch.uint8)
                        )
                else:
                    # 檢查裁剪圖像
                    current_image = pick_template["cropped_images"][class_idx]
                    if isinstance(current_image, torch.Tensor) and (
                        current_image.shape[1] < 5 or current_image.shape[2] < 5
                    ):
                        LOGGER.warning(
                            f"模板圖像 {class_idx} 過小: {current_image.shape}，使用零張量替代"
                        )
                        template_feats.append(
                            torch.zeros((3, 224, 224), dtype=torch.uint8)
                        )
                    else:
                        template_feats.append(current_image)

        # 8. 確保所有模板圖像大小一致
        resize_transform = transforms.Resize((target_size, target_size))
        resized_template_feats = []

        for idx, template_feat in enumerate(template_feats):
            try:
                if isinstance(template_feat, torch.Tensor):
                    resized_template_feats.append(
                        resize_transform(template_feat))
                elif isinstance(template_feat, np.ndarray):
                    tensor_feat = torch.tensor(
                        template_feat, dtype=torch.uint8
                    ).permute(2, 0, 1)
                    resized_template_feats.append(
                        resize_transform(tensor_feat))
                else:
                    #raise ValueError(f"未知的模板特徵類型: {type(template_feat)}")
                    raise ValueError(f"Unknown template feats type: {type(template_feat)}")
            except Exception as e:
                # LOGGER.error(f"調整模板圖像 {idx} 大小時出錯: {e}")
                LOGGER.error(f"Error: {e}, when adjust no.{idx} ")
                # 提供默認替代方案
                resized_template_feats.append(
                    torch.zeros((3, target_size, target_size),
                                dtype=torch.uint8)
                )

        # 將模板特徵添加到批次
        new_batch["template_feats"] = torch.stack(resized_template_feats, 0)

        return new_batch

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            # hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            # hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.mosaic = 0.0
            hyp.mixup = 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = (
            0,
            0,
            0,
            0,
            [],
        )  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for (
                im_file,
                lb,
                shape,
                segments,
                keypoint,
                nm_f,
                nf_f,
                ne_f,
                nc_f,
                msg,
            ) in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:

                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                            "segments_rle": None,
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(
                f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}"
            )
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x=x,
                                version=DATASET_CACHE_VERSION)
        return x


class Each_Picture_Each_Class_SA_V_V2VDataset(SA_V_V2VDataset):
    """
    For SA_V dataset's vision2vision dataset.
    """

    random.seed(878787)

    def __init__(
        self,
        *args,
        data=None,
        task="detect",
        train=False,
        num_search_range=10,
        **kwargs,
    ):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        super().__init__(
            *args,
            data=data,
            task=task,
            train=train,
            num_search_range=num_search_range,
            **kwargs,
        )

    @staticmethod
    def collate_fn(batch_dict: list):
        """
        將數據樣本整合到批次中。

        參數:
            batch_dict (list): 包含樣本數據的字典列表

        返回:
            dict: 處理後的批次數據
        """
        # 1. 收集批次中的基本數據
        batch = [b["batch"] for b in batch_dict]
        template = [b["template"] for b in batch_dict]
        video_name = [b["video_name"] for b in batch_dict]
        video_name_unique = np.unique(np.array(video_name)).tolist()
        nc = batch_dict[0]["nc"]
        random_image = [b["template"]["random_img"] for b in batch_dict]
        target_size = batch_dict[0]["target_size"]
        # bbox 格式: xywh 已標準化, x 和 y 是框的中心 (YOLO 格式)

        # 2. 從重複樣本中隨機選擇樣本
        video_name_idx_pair_dict = get_random_indices(video_name)

        # 3. 處理類別數量
        template_nc = 0
        init_origin_class_list = []
        init_video_name_idx_list = []

        # 收集所有模板類別和對應的視訊名稱
        for key, value in video_name_idx_pair_dict.items():
            template_nc += len(torch.unique(template[value]["cls"]))
            current_classes = [int(cls) for cls in template[value]["cls"]]
            init_origin_class_list.append(current_classes)
            init_video_name_idx_list.append([key] * len(current_classes))

        # 展平列表
        origin_class_list = sum(init_origin_class_list, [])
        video_name_idx_list = sum(init_video_name_idx_list, [])

        # 處理類別數量超出或不足的情況
        if template_nc <= nc:
            # 類別數量不足時，用 -1 填充
            origin_class_list.extend([-1] * (nc - len(origin_class_list)))
            video_name_idx_list.extend([-1] * (nc - len(video_name_idx_list)))
        else:
            # 類別數量超出時，截取前 nc 個
            origin_class_list = origin_class_list[:nc]
            video_name_idx_list = video_name_idx_list[:nc]

        # 4. 隨機打亂類別索引
        zipped = list(zip(origin_class_list, video_name_idx_list))
        random.shuffle(zipped)
        s1, s2 = zip(*zipped)
        shuffled_origin_class_list = np.array(list(s1))
        shuffled_video_name_idx_list = np.array(list(s2))
        shuffled_result = [
            list(pair)
            for pair in zip(shuffled_video_name_idx_list, shuffled_origin_class_list)
        ]

        # 5. 處理隨機圖像
        random_image_list = []
        for rnd_img in random_image:
            if rnd_img["video_name"] not in video_name_unique:
                random_image_list.append(rnd_img)

        # 如果沒有合適的隨機圖像，使用原始隨機圖像列表
        if len(random_image_list) == 0:
            random_image_list = random_image

        # 6. 處理查詢圖像並構建新批次
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))

        # 初始化 valid_mask 為 None
        valid_mask = None

        for i, k in enumerate(keys):
            value = values[i]

            # 堆疊圖像
            if k == "img":
                value = torch.stack(value, 0)

            # 處理類別信息
            if k == "cls":
                value_cls_list = []
                for v_idx, v in enumerate(value):
                    valid_indices = np.where(
                        shuffled_video_name_idx_list == video_name[v_idx]
                    )[0]
                    temp_origin_class_list = np.array(
                        [int(vv[0]) for vv in v.tolist()])
                    cls_temp = shuffled_origin_class_list[valid_indices]

                    mapped_indices = []
                    for cls in temp_origin_class_list:
                        matches = np.where(cls_temp == cls)[0]
                        if len(matches) > 0:
                            mapped_indices.append(valid_indices[matches[0]])
                        else:
                            # 類別不在模板中，標記為 -1
                            mapped_indices.append(-1)

                    value_cls_list.append(
                        torch.tensor(mapped_indices,
                                     dtype=torch.float32).unsqueeze(-1)
                    )

                value = value_cls_list

            # 處理特殊類型的數據
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
                if k == "cls":  # 移除不存在於模板中的類別 (-1)
                    valid_mask = value != -1
                    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
                    valid_mask = valid_mask.reshape(-1)
                    value = value[valid_mask]
                elif valid_mask is not None:
                    # 僅當 valid_mask 已被定義時才應用
                    value = value[valid_mask]

            new_batch[k] = value

        # 更新批次索引
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # 為 build_targets() 添加目標圖像索引

        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        if valid_mask is not None:
            new_batch["batch_idx"] = new_batch["batch_idx"][valid_mask]

        # 7. 創建裁剪的模板圖像
        template_feats = []
        resize_transform = transforms.Resize((target_size, target_size))

        for one_batch in batch:
            class_list = sum(one_batch['cls'].int().tolist(), [])
            crop_size_w, crop_size_h = one_batch['cropped_images'].shape[2], one_batch['cropped_images'].shape[3]
            # one_batch_template_feats = extract_and_sample_images(data_list=random_image, sample_size=nc, crop_size=(crop_size_h, crop_size_w))
            one_batch_template_feats = random_sample_picture(data_list=random_image, sample_size=nc)
            for i, class_idx in enumerate(class_list):
                one_batch_template_feats[class_idx] = one_batch['cropped_images'][i]
            for sample in one_batch_template_feats:
                template_feats.append(resize_transform(sample))

            # template_feats.extend(one_batch_template_feats)
        

        new_batch["template_feats"] = torch.stack(template_feats, 0)

        return new_batch

        for video_name_cls_pair in shuffled_result:
            video_name_id, origin_cls = video_name_cls_pair

            if video_name_id == -1:  # 隨機裁剪，非模板
                # 隨機選擇圖像並裁剪
                random_image_pick = random.choice(random_image_list)
                height, width = 80, 80

                # 確保裁剪區域在有效範圍內
                max_x = min(random_image_pick["img"].shape[0] - height, height)
                max_y = min(random_image_pick["img"].shape[1] - width, width)
                min_x = random.randint(0, max(0, max_x))
                min_y = random.randint(0, max(0, max_y))

                is_train = batch_dict[0]["is_train"]

                '''
                if is_train:
                    # 在訓練模式下裁剪實際圖像
                    random_image_tensor = torch.from_numpy(
                        random_image_pick["img"])
                    random_image_tensor = random_image_tensor.permute(2, 0, 1)
                    cropped_tensor = F.crop(
                        random_image_tensor,
                        top=min_y,
                        left=min_x,
                        height=height,
                        width=width,
                    )

                    # 檢查裁剪結果
                    if cropped_tensor.shape[1] < 5 or cropped_tensor.shape[2] < 5:
                        LOGGER.warning(
                            f"裁剪圖像過小: {cropped_tensor.shape}，使用零張量替代"
                        )
                        cropped_tensor = torch.zeros(
                            (3, height, width), dtype=torch.uint8
                        )
                else:
                    # 測試模式使用零張量
                    cropped_tensor = torch.zeros(
                        (3, 224, 224), dtype=torch.uint8)
                '''
                # We use zero tensor for all non-template class (black image)
                cropped_tensor = torch.zeros(
                    (3, 224, 224), dtype=torch.uint8)

                template_feats.append(cropped_tensor)

            else:  # 使用模板
                template_idx = video_name_idx_pair_dict[video_name_id]
                pick_template = template[template_idx]
                pick_template_class_list = [
                    int(pp[0]) for pp in pick_template["cls"]]

                try:
                    class_idx = pick_template_class_list.index(origin_cls)
                except ValueError:
                    class_idx = -1
                    LOGGER.warning(
                        f"模板中找不到類別 {origin_cls}，使用第一個模板圖像替代"
                    )

                if class_idx == -1:
                    # 無法找到匹配的類別時，使用第一個模板圖像
                    if len(pick_template["cropped_images"]) > 0:
                        template_feats.append(
                            pick_template["cropped_images"][0])
                    else:
                        LOGGER.warning(
                            f"模板 {template_idx} 沒有裁剪圖像，使用零張量替代"
                        )
                        template_feats.append(
                            torch.zeros((3, 224, 224), dtype=torch.uint8)
                        )
                else:
                    # 檢查裁剪圖像
                    current_image = pick_template["cropped_images"][class_idx]
                    if isinstance(current_image, torch.Tensor) and (
                        current_image.shape[1] < 5 or current_image.shape[2] < 5
                    ):
                        LOGGER.warning(
                            f"模板圖像 {class_idx} 過小: {current_image.shape}，使用零張量替代"
                        )
                        template_feats.append(
                            torch.zeros((3, 224, 224), dtype=torch.uint8)
                        )
                    else:
                        template_feats.append(current_image)

        # 8. 確保所有模板圖像大小一致
        resize_transform = transforms.Resize((target_size, target_size))
        resized_template_feats = []

        for idx, template_feat in enumerate(template_feats):
            try:
                if isinstance(template_feat, torch.Tensor):
                    resized_template_feats.append(
                        resize_transform(template_feat))
                elif isinstance(template_feat, np.ndarray):
                    tensor_feat = torch.tensor(
                        template_feat, dtype=torch.uint8
                    ).permute(2, 0, 1)
                    resized_template_feats.append(
                        resize_transform(tensor_feat))
                else:
                    #raise ValueError(f"未知的模板特徵類型: {type(template_feat)}")
                    raise ValueError(f"Unknown template feats type: {type(template_feat)}")
            except Exception as e:
                # LOGGER.error(f"調整模板圖像 {idx} 大小時出錯: {e}")
                LOGGER.error(f"Error: {e}, when adjust no.{idx} ")
                # 提供默認替代方案
                resized_template_feats.append(
                    torch.zeros((3, target_size, target_size),
                                dtype=torch.uint8)
                )

        # 將模板特徵添加到批次
        new_batch["template_feats"] = torch.stack(resized_template_feats, 0)

        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    for v2v model.
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """Add texts information for multi-modal model training."""
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp=None):
        """Enhances data transformations with optional text augmentation for multi-modal training."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(
                -1, RandomLoadText(max_samples=min(self.data["nc"], 80), padding=True)
            )
        return transforms

class ObjectOrientedYOLODataset(YOLODataset):
    """
    加載成對的圖像與標註的資料集

    這個類將圖像和標註配對加載，基於檔名前綴將 *I* 和 *T* 配對在一起
    """
    
    # def __init__(self, *args, **kwargs):
    def __init__(self,
                img_path,
                imgsz=640,
                cache=False,
                augment=True,
                hyp=DEFAULT_CFG,
                prefix="",
                rect=False,
                batch_size=16,
                stride=32,
                pad=0.5,
                single_cls=False,
                classes=None,
                fraction=1.0,
                channels=3,
                data=None, 
                task="detect",
                vision_encoder_input_size=224,):
        # self.use_segments = kwargs.get("task", "detect") == "segment"
        # self.use_keypoints = kwargs.get("task", "detect") == "pose"
        # self.use_obb = kwargs.get("task", "detect") == "obb"
        # self.data = kwargs.get("data", None)
        self.use_segments = (task == "segment")
        self.use_keypoints = (task == "pose")
        self.use_obb = (task == "obb")
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Cannot use segments and keypoints at the same time"
        
        # super().__init__(*args, **kwargs)
        # super().__init__()
        Dataset.__init__(self)
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        
        self.npy_files = []
        for i, t in self.im_files:
            self.npy_files.append(Path(i).with_suffix(".npy"))
            self.npy_files.append(Path(t).with_suffix(".npy"))
        # self.npy_files = [Path(i).with_suffix(".npy") for i, t in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.vision_encoder_input_size = vision_encoder_input_size
        self.transforms = self.build_transforms(hyp=hyp)
        self.__update__im_files__()
        
    def pair_im_files(self):
        """將 'I' 和 'T' 類型的圖片配對"""
        # 建立配對索引
        self.pair_indices = {}
        self.pair_keys = []
        self.paired_im_files = []
        self.original_im_files = self.im_files.copy()  # 保存原始列表
        
        for im_file in self.im_files:
            # 提取檔名前綴與類型（I 或 T）
            basename = Path(im_file).stem
            # 檢查最後一個字元是否為 I 或 T
            if basename[-2] not in ('I', 'T'):
                continue
            
            # basen
            prefix = basename[:-3]  # 移除最後一個字元（I或T） 
            suffix = basename[-2]   # 取得最後一個字元（I或T）
            
            if prefix not in self.pair_indices:
                self.pair_indices[prefix] = {}
            
            self.pair_indices[prefix][suffix] = im_file
            
            # 如果這個前綴已經有I和T，則添加到配對鍵列表中
            if len(self.pair_indices[prefix]) == 2 and prefix not in self.pair_keys:
                self.pair_keys.append(prefix)
                self.paired_im_files.append((
                    self.pair_indices[prefix].get('I'),
                    self.pair_indices[prefix].get('T')
                ))
        
        LOGGER.info(f"Found {len(self.pair_keys)} pairs of matched images")
        
    def cache_labels(self, path=Path("./labels.cache")):
        """
        緩存資料集標註，檢查圖片並讀取形狀
        
        Args:
            path (Path): 緩存文件保存路徑
            
        Returns:
            dict: 包含緩存標註的字典
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # 缺失、發現、空、損壞、訊息的數量
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.original_im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            # raise ValueError(
            #     "'kpt_shape' 在 data.yaml 中缺失或不正確。應該是包含 [關鍵點數量, 維度數量 (2表示x,y或3表示x,y,可見性)] 的列表"
            # )
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.original_im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                # pbar.desc = f"{desc} {nf} 圖片, {nm + ne} 背景, {nc} 損壞"
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            # LOGGER.warning(f"{self.prefix}沒有在 {path} 中找到標註. {HELP_URL}")
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.original_im_files)
        x["results"] = nf, nm, ne, nc, len(self.original_im_files)
        x["msgs"] = msgs  # 警告
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x
    
    def __process_prefix__(self, prefix, pair_indices, all_labels_dict):
        i_file = pair_indices[prefix].get('I')
        t_file = pair_indices[prefix].get('T')
        if i_file is None or t_file is None:
            return None
        
        i_label = all_labels_dict.get(i_file)
        t_label = all_labels_dict.get(t_file)
        
        if i_label is None or t_label is None:
            return None
        
        return {
            "prefix": prefix,
            "i_label": i_label,
            "t_label": t_label
        }
        
    def get_labels(self):
        """
        獲取標註並處理成對的圖片和標註
        
        Returns:
            (List[dict]): 包含配對圖片和標註的列表
        """
        self.original_im_files = self.im_files.copy()
        self.label_files = img2label_paths(self.original_im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.original_im_files)
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False
            
        # 顯示緩存
        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))
                
        # 處理標註
        self.all_labels = cache["labels"]
        if not self.all_labels:
            raise RuntimeError(f"Did not fine effective image {cache_path}. {HELP_URL}")
                
        # Get pair of input and template images

        self.get_pair_labels()
        # self.pair_im_files()
            
        # # 將標註按照 I 和 T 分類
        # # 我們會創建一個新的標註列表，每個元素包含一對圖片和標註
        # paired_labels = []
    
        # for prefix in TQDM(self.pair_keys, desc='Pairing labels'):
        #     i_file = self.pair_indices[prefix].get('I')
        #     t_file = self.pair_indices[prefix].get('T')
            
        #     if i_file is None or t_file is None:
        #         continue
                
        #     # 在所有標註中尋找對應的標註
        #     i_label = None
        #     t_label = None
            
        #     for label in all_labels:
        #         if label["im_file"] == i_file:
        #             i_label = label
        #         elif label["im_file"] == t_file:
        #             t_label = label
                    
        #         if i_label is not None and t_label is not None:
        #             break
                    
        #     if i_label is None or t_label is None:
        #         continue
                
        #     # 建立一個包含兩組標註的對
        #     paired_labels.append({
        #         "prefix": prefix,
        #         "i_label": i_label,
        #         "t_label": t_label
        #     })
        
        # # Pair data cache
            
        # # 更新 im_files 為配對後的文件列表
        # self.im_files = [
        #     (self.pair_indices[label["prefix"]].get('I'), self.pair_indices[label["prefix"]].get('T'))
        #     for label in paired_labels
        # ]
        
        return self.paired_labels

    def get_pair_labels(self):
        # cache_path = Path(f"{self.label_files[0]}_pair").parent.with_suffix(".cache")
        cache_path = Path(self.label_files[0]).parent / "pair_labels.cache"
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.original_im_files)
            
            self.im_files = cache["im_files"]
            self.pair_indices = cache["pair_indices"]
            self.paired_im_files = cache["paired_im_files"]
            self.original_im_files = cache["original_im_files"]
            self.paired_labels = cache["paired_labels"]
        
        except (FileNotFoundError, AssertionError, AttributeError):

            self.pair_im_files()
                
            # 將標註按照 I 和 T 分類
            # 我們會創建一個新的標註列表，每個元素包含一對圖片和標註
            paired_labels = []
        
            for prefix in TQDM(self.pair_keys, desc='Pairing labels'):
                i_file = self.pair_indices[prefix].get('I')
                t_file = self.pair_indices[prefix].get('T')
                
                if i_file is None or t_file is None:
                    continue
                    
                # 在所有標註中尋找對應的標註
                i_label = None
                t_label = None
                
                for label in self.all_labels:
                    if label["im_file"] == i_file:
                        i_label = label
                    elif label["im_file"] == t_file:
                        t_label = label
                        
                    if i_label is not None and t_label is not None:
                        break
                        
                if i_label is None or t_label is None:
                    continue
                    
                # 建立一個包含兩組標註的對
                paired_labels.append({
                    "prefix": prefix,
                    "i_label": i_label,
                    "t_label": t_label
                })
            
            # Pair data cache
                
            # 更新 im_files 為配對後的文件列表
            self.im_files = [
                (self.pair_indices[label["prefix"]].get('I'), self.pair_indices[label["prefix"]].get('T'))
                for label in paired_labels
            ]

            self.paired_labels = paired_labels

            # 緩存配對標註
            cache, exists = self.cache_pair_labels(cache_path), False

        return cache, exists
    
    def cache_pair_labels(self, cache_path=None):
        cache_path = cache_path = Path(self.label_files[0]).parent / "pair_labels.cache" if None else cache_path
        
        pair_data_cache = {}
        pair_data_cache["hash"] = get_hash(self.label_files + self.original_im_files)

        pair_data_cache["im_files"] = self.im_files
        pair_data_cache["pair_indices"] = self.pair_indices
        pair_data_cache["paired_im_files"] = self.paired_im_files
        pair_data_cache["original_im_files"] = self.original_im_files
        pair_data_cache["paired_labels"] = self.paired_labels
        # pair_data_cache["results"] = nf, nm, ne, nc, len(self.original_im_files)
        save_dataset_cache_file(self.prefix, cache_path, pair_data_cache, DATASET_CACHE_VERSION)

        return pair_data_cache

    def __update__im_files__(self):
        self.im_files = [
            (self.pair_indices[label["prefix"]].get('I'), self.pair_indices[label["prefix"]].get('T'))
            for label in self.labels
        ]
        
    def __len__(self):
        """返回配對數量"""
        return len(self.im_files)
    
    def __getitem__(self, index):
        """返回一對圖片和標註"""
        paired_label = self.labels[index]
        i_label = paired_label["i_label"]
        t_label = paired_label["t_label"]
        
        # 處理 I 標籤
        i_data = self.get_label_data(i_label)
        # 處理 T 標籤
        t_data = self.get_label_data(t_label)
        
        # 創建一個包含兩組資料的字典
        paired_data = {
            'i_data': i_data,
            't_data': t_data,
        }
        
        # 應用轉換
        if self.transforms:
            paired_data = self.transforms(paired_data)
            
        return paired_data
    
    def get_label_data(self, label):
        """處理單個標註數據"""
        label = deepcopy(label)  # 需要深拷貝 https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape 用於 rect，移除它
        img, ori_shape, resized_shape = self.load_image_by_path(label["im_file"])
        label["img"] = img
        label["ori_shape"] = ori_shape
        label["resized_shape"] = resized_shape
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # 用於評估
        if self.rect:
            # 找到這個文件的批次索引
            for i, (img1, img2) in enumerate(self.im_files):
                if img1 == label["im_file"] or img2 == label["im_file"]:
                    label["rect_shape"] = self.batch_shapes[self.batch[i]]
                    break
        return self.update_labels_info(label)
    
    def load_image_by_path(self, im_path):
        """根據路徑加載圖片"""
        img = None
        h0, w0 = None, None
        
        # 檢查圖片是否已緩存
        for i, (img1, img2) in enumerate(self.im_files):
            if im_path == img1 or im_path == img2:
                # 如果已緩存，直接返回
                if self.ims[i] is not None:
                    return self.ims[i], self.im_hw0[i], self.im_hw[i]
                break
        
        # 如果未緩存，加載圖片
        img = imread(im_path, flags=self.cv2_flag)
        if img is None:
            raise FileNotFoundError(f"找不到圖片 {im_path}")
            
        h0, w0 = img.shape[:2]  # 原始高寬
        
        # 根據是否使用 rect 模式調整圖片大小
        if self.rect:
            # 調整圖片大小，保持長寬比
            r = self.imgsz / max(h0, w0)
            if r != 1:
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        elif h0 != w0 or h0 != self.imgsz:
            # 拉伸調整為正方形
            img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            
        # 確保單通道圖片有正確的維度
        if img.ndim == 2:
            img = img[..., None]
            
        return img, (h0, w0), img.shape[:2]
    
    @staticmethod
    def collate_fn(batch):
        """
        將批次的資料合併成可用於訓練的格式
        
        Args:
            batch (List[dict]): 每個元素包含 'i_data' 和 't_data' 兩個字典
            
        Returns:
            dict: 用於訓練的批次資料
        """
        i_batch = [x['i_data'] for x in batch]
        t_batch = [x['t_data'] for x in batch]
        
        # 整理 i_batch 和 t_batch
        i_collated = {}
        t_collated = {}
        
        keys = i_batch[0].keys()
        # 處理一般欄位（img, cls, bboxes等）
        # for key in ['im_file', 'ori_shape', 'resized_shape', 'img', 'cls', 'bboxes', 'masks', 'keypoints', 'visuals']:
        for key in keys:
            if key in i_batch[0]:
                i_values = [x[key] for x in i_batch]
                t_values = [x[key] for x in t_batch]
                
                # if key == 'img':
                #     i_collated[key] = torch.stack(i_values, 0)
                #     t_collated[key] = torch.stack(t_values, 0)
                # elif key in ['masks', 'keypoints', 'bboxes', 'cls']:
                #     i_collated[key] = torch.cat(i_values, 0)
                #     t_collated[key] = torch.cat(t_values, 0)
                # elif key == "visuals":
                #     # refer from ultralytics/data/dataset.py YOLODataset.collate_fn
                #     i_collated[key] = torch.nn.utils.rnn.pad_sequence(i_values, batch_first=True)
                #     t_collated[key] = torch.nn.utils.rnn.pad_sequence(t_values, batch_first=True)
                if key == 'img':
                    i_values = torch.stack(i_values, 0)
                    t_values = torch.stack(t_values, 0)
                elif key in ['masks', 'keypoints', 'bboxes', 'cls']:
                    i_values = torch.cat(i_values, 0)
                    t_values = torch.cat(t_values, 0)
                elif key == "visuals":
                    # refer from ultralytics/data/dataset.py YOLODataset.collate_fn
                    i_values = torch.nn.utils.rnn.pad_sequence(i_values, batch_first=True)
                    t_values = torch.nn.utils.rnn.pad_sequence(t_values, batch_first=True)
                
                i_collated[key] = i_values
                t_collated[key] = t_values
        
        # 處理 batch_idx
        if 'batch_idx' in i_batch[0]:
            i_batch_idx = []
            t_batch_idx = []
            
            for i in range(len(i_batch)):
                i_batch_idx.append(i_batch[i]['batch_idx'] + i)
                t_batch_idx.append(t_batch[i]['batch_idx'] + i)
                
            i_collated['batch_idx'] = torch.cat(i_batch_idx, 0)
            t_collated['batch_idx'] = torch.cat(t_batch_idx, 0)
        
        # 合併結果
        result = {
            'i': i_collated,
            't': t_collated
        }
        
        return result
    
    def build_transforms(self, hyp=None):
        """
        構建資料轉換管道
        
        需要修改以處理成對資料
        """
        if self.augment:
            # hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            # hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            # hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            
            base_hyp = deepcopy(hyp)
            hyp.mosaic = 0.0  # 禁用 Mosaic
            hyp.mixup = 0.0   # 禁用 MixUp
            hyp.cutmix = 0.0  # 禁用 CutMix
            
            # 獲取原始的轉換
            single_transforms = v2v_transforms(self, self.imgsz, hyp)
            single_transforms.append(RandomLoadClass(max_samples=min(self.data["nc"], 80), padding=True))

            template_transforms = v2v_transforms(self, self.vision_encoder_input_size, hyp)
            template_transforms.append(RandomLoadClass(max_samples=min(self.data["nc"], 80), padding=True))
        else:
            single_transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
            template_transforms = Compose([LetterBox(new_shape=(self.vision_encoder_input_size, self.vision_encoder_input_size), scaleup=False)])
            
        # 添加格式化轉換
        format_transform = Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=self.use_segments,
            return_keypoint=self.use_keypoints,
            return_obb=self.use_obb,
            batch_idx=True,
            mask_ratio=hyp.mask_ratio if hyp else 4,
            mask_overlap=hyp.overlap_mask if hyp else True,
        )
        single_transforms.append(format_transform)
        single_transforms.append(LoadVisualPrompt())
        template_transforms.append(format_transform)
        template_transforms.append(LoadVisualPrompt(80/256))
        
        # # 創建一個新的轉換函數來處理成對資料
        # def transform_pair(paired_data):
        #     paired_data['i_data'] = single_transforms(paired_data['i_data'])
        #     paired_data['t_data'] = single_transforms(paired_data['t_data'])
        #     return paired_data
        LOGGER.info(f"Warning: We have now set template_transforms to size 224, LoadVisualPrompt(80/224) in file v2vdet_ultralytics/data/dataset.py line 2620.")
        
        def transform_pair(paired_data):
            # 使用相同的隨機種子處理兩張圖片，確保一致的隨機變換
            seed = np.random.randint(2147483647)
            
            # 處理 I 圖片
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            paired_data['i_data'] = single_transforms(paired_data['i_data'])
            
            # 處理 T 圖片，使用相同的隨機種子
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # paired_data['t_data'] = single_transforms(paired_data['t_data'])
            paired_data['t_data'] = template_transforms(paired_data['t_data'])
            
            return paired_data
        
        return transform_pair

class MultiClass_ObjectOrientedYOLODataset(ObjectOrientedYOLODataset):

    def __init__(self, *args, n_extra=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_extra = n_extra  # 額外模板的數量

    def __getitem__(self, index):
        """返回一對圖片和多個模板標註"""
        # 獲取原始配對
        paired_label = self.labels[index]
        i_label = paired_label["i_label"]
        t_label = paired_label["t_label"]
        
        # 處理 I 標籤
        i_data = self.get_label_data(i_label)
        # 處理 T 標籤
        t_data = self.get_label_data(t_label)
        
        # 額外選擇 N 個隨機模板
        n_extra = self.n_extra  # 您可以調整這個數字，決定要添加多少個額外模板
        extra_t_data = []
        
        # 從數據集中隨機選擇 n_extra 個不同的索引
        indices = random.sample(range(len(self.labels)), min(n_extra + 1, len(self.labels)))
        # 確保不會選到當前索引
        if index in indices:
            indices.remove(index)
        indices = indices[:n_extra]  # 限制為 n_extra 個
        
        # 獲取這些索引的 T 標籤
        for idx in indices:
            extra_t_label = self.labels[idx]["t_label"]
            extra_t_data.append(self.get_label_data(extra_t_label))
        
        # 創建一個包含一個輸入和多個模板的字典
        paired_data = {
            'i_data': i_data,
            't_data': t_data,  # 原始配對的模板
            'extra_t_data': extra_t_data  # 額外的模板列表
        }
        
        # 應用轉換
        if self.transforms:
            paired_data = self.transforms(paired_data)
            
        return paired_data
    
    @staticmethod
    def collate_fn(batch):
        """
        將批次的資料合併成可用於訓練的格式，支援多個模板
        
        Args:
            batch (List[dict]): 每個元素包含 'i_data'、't_data' 和 'extra_t_data'
                
        Returns:
            dict: 用於訓練的批次資料
        """
        i_batch = [x['i_data'] for x in batch]
        t_batch = [x['t_data'] for x in batch]
        
        # 整理 i_batch 和 t_batch
        i_collated = {}
        t_collated = {}
        
        keys = i_batch[0].keys()
        # 處理一般欄位
        for key in keys:
            if key in i_batch[0]:
                i_values = [x[key] for x in i_batch]
                t_values = [x[key] for x in t_batch]
                
                if key == 'img':
                    i_values = torch.stack(i_values, 0)
                    t_values = torch.stack(t_values, 0)
                elif key in ['masks', 'keypoints', 'bboxes', 'cls']:
                    i_values = torch.cat(i_values, 0)
                    t_values = torch.cat(t_values, 0)
                elif key == "visuals":
                    i_values = torch.nn.utils.rnn.pad_sequence(i_values, batch_first=True)
                    t_values = torch.nn.utils.rnn.pad_sequence(t_values, batch_first=True)
                
                i_collated[key] = i_values
                t_collated[key] = t_values
        
        # 處理 batch_idx
        if 'batch_idx' in i_batch[0]:
            i_batch_idx = []
            t_batch_idx = []
            
            for i in range(len(i_batch)):
                i_batch_idx.append(i_batch[i]['batch_idx'] + i)
                t_batch_idx.append(t_batch[i]['batch_idx'] + i)
                
            i_collated['batch_idx'] = torch.cat(i_batch_idx, 0)
            t_collated['batch_idx'] = torch.cat(t_batch_idx, 0)
        
        # 處理額外的模板數據
        # 將所有 extra_t_data 合併為一個列表
        all_extra_t_data = []
        for x in batch:
            all_extra_t_data.extend(x['extra_t_data'])
        
        # 如果有額外的模板，則處理它們
        extra_t_collated = {}
        if all_extra_t_data:
            for key in all_extra_t_data[0].keys():
                if key in all_extra_t_data[0]:
                    extra_t_values = [x[key] for x in all_extra_t_data]
                    
                    if key == 'img':
                        extra_t_values = torch.stack(extra_t_values, 0)
                    elif key in ['masks', 'keypoints', 'bboxes', 'cls']:
                        extra_t_values = torch.cat(extra_t_values, 0)
                    elif key == "visuals":
                        extra_t_values = torch.nn.utils.rnn.pad_sequence(extra_t_values, batch_first=True)
                    
                    extra_t_collated[key] = extra_t_values
        
        # 合併結果
        result = {
            'i': i_collated,
            't': t_collated,
            'extra_t': extra_t_collated
        }
        
        return result
    
    def build_transforms(self, hyp=None):
        """
        構建資料轉換管道，處理帶有額外模板的資料
        """
        # 獲取原始的轉換函數
        original_transform_pair = super().build_transforms(hyp)
        
        # 創建新的轉換函數
        def enhanced_transform_pair(paired_data):
            # 提取資料
            i_data = paired_data['i_data']
            t_data = paired_data['t_data']
            extra_t_data = paired_data.get('extra_t_data', [])
            
            # 將主要數據送入原始轉換函數
            temp_result = original_transform_pair({
                'i_data': i_data,
                't_data': t_data
            })
            
            # 取出轉換後的結果
            transformed_i_data = temp_result['i_data']
            transformed_t_data = temp_result['t_data']
            
            # 單獨處理每個額外的模板
            transformed_extra_t_data = []
            for extra_t in extra_t_data:
                # 為每個額外模板設置不同的隨機種子
                seed = np.random.randint(2147483647)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                # 使用與主數據相同的轉換
                temp_extra_result = original_transform_pair({
                    'i_data': None,  # 不需要轉換 i_data
                    't_data': extra_t
                })
                
                transformed_extra_t_data.append(temp_extra_result['t_data'])
            
            # 組合結果
            return {
                'i_data': transformed_i_data,
                't_data': transformed_t_data,
                'extra_t_data': transformed_extra_t_data,
                'num_extra': len(transformed_extra_t_data)
            }
        
        return enhanced_transform_pair