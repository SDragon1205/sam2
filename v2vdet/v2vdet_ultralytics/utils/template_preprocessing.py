from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
import cv2
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoImageProcessor, Dinov2Model, BatchFeature
import random
import torchvision.transforms as T
import torch
import torchvision.transforms.functional as F
import supervision as sv
# from supervision.utils.conversion import pillow_to_cv2
import matplotlib.pyplot as plt
from typing import TypeVar

import copy
from datetime import datetime
from typing import Callable, Dict, Optional, Union

from packaging import version

try:
    import dill as pickle
except ImportError:
    import pickle

from ultralytics.models import YOLO
from torchvision import transforms
import random
from copy import deepcopy
from pycocotools import mask as mask_utils

ImageType = TypeVar("ImageType", np.ndarray, Image.Image)

class template_preprocessing():
  
  def __init__(self, image, boxes, classes, size=(224, 224), augment=False, aug_params=None):
    
    pass

  def __call__(self, *args, **kwargs):
    return None