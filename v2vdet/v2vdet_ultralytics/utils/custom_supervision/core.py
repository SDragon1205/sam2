from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from supervision.classification.core import Classifications
from supervision.dataset.formats.coco import (
    load_coco_annotations,
    save_coco_annotations,
)
from supervision.dataset.formats.pascal_voc import (
    detections_to_pascal_voc,
    load_pascal_voc_annotations,
)
from supervision.dataset.formats.yolo import (
    load_yolo_annotations,
    save_data_yaml,
    save_yolo_annotations,
)
from supervision.dataset.utils import (
    build_class_index_mapping,
    map_detections_class_id,
    merge_class_lists,
    save_dataset_images,
    train_test_split,
)
from supervision.dataset.core import (
  BaseDataset, 
  DetectionDataset
)
from .lvis import (
  load_lvis_annotations,
  save_lvis_annotations,
)
from supervision.detection.core import Detections
from supervision.utils.internal import deprecated, warn_deprecated
from supervision.utils.iterables import find_duplicates

import requests

class LVIS_DetectionDataset(DetectionDataset):
    """
    Contains information about a detection dataset. Handles lazy image loading
    and annotation retrieval, dataset splitting, conversions into multiple
    formats.

    Attributes:
        classes (List[str]): List containing dataset class names.
        images (Union[List[str], Dict[str, np.ndarray]]):
            Accepts a list of image paths, or dictionaries of loaded cv2 images
            with paths as keys. If you pass a list of paths, the dataset will
            lazily load images on demand, which is much more memory-efficient.
        annotations (Dict[str, Detections]): Dictionary mapping
            image path to annotations. The dictionary keys match
            match the keys in `images` or entries in the list of
            image paths.
    """

    def __init__(
        self,
        classes: List[str],
        images: Union[List[str], Dict[str, np.ndarray]],
        annotations: Dict[str, Detections],
    ) -> None:
      super().__init__(classes=classes, images=images, annotations=annotations)
    
  
    def _get_image(self, image_path: str) -> np.ndarray:
      """Assumes that image is in dataset"""
      if self._images_in_memory:
          return self._images_in_memory[image_path]
      return self._from_url_to_cv2(image_path)

    @property
    @deprecated(
        "`DetectionDataset.images` property is deprecated and will be removed in "
        "`supervision-0.26.0`. Iterate with `for path, image, annotation in dataset:` "
        "instead."
    )
    def images(self) -> Dict[str, np.ndarray]:
      """
      Load all images to memory and return them as a dictionary.

      !!! warning

          Only use this when you need all images at once.
          It is much more memory-efficient to initialize dataset with
          image paths and use `for path, image, annotation in dataset:`.
      """
      if self._images_in_memory:
          return self._images_in_memory

      # images = {image_path: cv2.imread(image_path) for image_path in self.image_paths}
      images = {image_path: self._from_url_to_cv2(image_path) for image_path in self.image_paths}
      
      return images
      
    @classmethod
    def from_coco(
        cls,
        images_directory_path: str,
        annotations_path: str,
        force_masks: bool = False,
    ) -> DetectionDataset:
      """
      Creates a Dataset instance from COCO formatted data.

      Args:
          images_directory_path (str): The path to the
              directory containing the images.
          annotations_path (str): The path to the json annotation files.
          force_masks (bool): If True,
              forces masks to be loaded for all annotations,
              regardless of whether they are present.

      Returns:
          DetectionDataset: A DetectionDataset instance containing
              the loaded images and annotations.

      Examples:
          ```python
          import roboflow
          from roboflow import Roboflow
          import supervision as sv

          roboflow.login()
          rf = Roboflow()

          project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
          dataset = project.version(PROJECT_VERSION).download("coco")

          ds = sv.DetectionDataset.from_coco(
              images_directory_path=f"{dataset.location}/train",
              annotations_path=f"{dataset.location}/train/_annotations.coco.json",
          )

          ds.classes
          # ['dog', 'person']
          ```
      """
      classes, images, annotations = load_lvis_annotations(
          images_directory_path=images_directory_path,
          annotations_path=annotations_path,
          force_masks=force_masks,
      )
      return LVIS_DetectionDataset(classes=classes, images=images, annotations=annotations)

    def as_coco(
        self,
        images_directory_path: Optional[str] = None,
        annotations_path: Optional[str] = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
    ) -> None:
      """
      Exports the dataset to COCO format. This method saves the
      images and their corresponding annotations in COCO format.

      !!! tip

          The format of the mask is determined automatically based on its structure:

          - If a mask contains multiple disconnected components or holes, it will be
          saved using the Run-Length Encoding (RLE) format for efficient storage and
          processing.
          - If a mask consists of a single, contiguous region without any holes, it
          will be encoded as a polygon, preserving the outline of the object.

          This automatic selection ensures that the masks are stored in the most
          appropriate and space-efficient format, complying with COCO dataset
          standards.

      Args:
          images_directory_path (Optional[str]): The path to the directory
              where the images should be saved.
              If not provided, images will not be saved.
          annotations_path (Optional[str]): The path to COCO annotation file.
          min_image_area_percentage (float): The minimum percentage of
              detection area relative to
              the image area for a detection to be included.
              Argument is used only for segmentation datasets.
          max_image_area_percentage (float): The maximum percentage of
              detection area relative to
              the image area for a detection to be included.
              Argument is used only for segmentation datasets.
          approximation_percentage (float): The percentage of polygon points
              to be removed from the input polygon,
              in the range [0, 1). This is useful for simplifying the annotations.
              Argument is used only for segmentation datasets.
      """
      if images_directory_path is not None:
          save_dataset_images(
              dataset=self, images_directory_path=images_directory_path
          )
      if annotations_path is not None:
          save_coco_annotations(
              dataset=self,
              annotation_path=annotations_path,
              min_image_area_percentage=min_image_area_percentage,
              max_image_area_percentage=max_image_area_percentage,
              approximation_percentage=approximation_percentage,
          )
    
    def _from_url_to_numpy(self, url: str) -> np.ndarray:
      response = requests.get(url)
      image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
      return image_array
    
    def _from_url_to_cv2(self, url: str) -> np.ndarray:
      image_array = self._from_url_to_numpy(url)
      return cv2.imdecode(image_array, cv2.IMREAD_COLOR)