import math
import random
from copy import deepcopy
from typing import Tuple, Union, List

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13
from ultralytics.data.augment import Format
from ultralytics.data.augment import (Mosaic, 
                                      Albumentations,
                                      RandomPerspective,
                                      LetterBox,
                                      Compose,
                                      CopyPaste,
                                      MixUp,
                                      CutMix,
                                      RandomHSV,
                                      RandomFlip,
                                      )

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0

# class RandomLoadClass:
#     """
#     Randomly samples positive and negative class and updates class indices accordingly.

#     This class is responsible for sampling texts from a given set of class texts, including both positive
#     (present in the image) and negative (not present in the image) samples. It updates the class indices
#     to reflect the sampled texts and can optionally pad the text list to a fixed length.

#     Attributes:
#         prompt_format (str): Format string for text prompts.
#         neg_samples (Tuple[int, int]): Range for randomly sampling negative texts.
#         max_samples (int): Maximum number of different text samples in one image.
#         padding (bool): Whether to pad texts to max_samples.
#         padding_value (str): The text used for padding when padding is True.

#     Methods:
#         __call__: Processes the input labels and returns updated classes and texts.

#     Examples:
#         >>> loader = RandomLoadClass(prompt_format="Object: {}", neg_samples=(5, 10), max_samples=20)
#         >>> labels = {"cls": [0, 1, 2], "texts": [["cat"], ["dog"], ["bird"]], "instances": [...]}
#         >>> updated_labels = loader(labels)
#         >>> print(updated_labels["texts"])
#         ['Object: cat', 'Object: dog', 'Object: bird', 'Object: elephant', 'Object: car']
#     """

#     def __init__(
#         self,
#         neg_samples: Tuple[int, int] = (80, 80),
#         max_samples: int = 80,
#         padding: bool = False,
#         padding_value: str = "",
#     ) -> None:
#         """
#         Initializes the RandomLoadClass class for randomly sampling positive and negative texts.

#         This class is designed to randomly sample positive texts and negative texts, and update the class
#         indices accordingly to the number of samples. It can be used for text-based object detection tasks.

#         Args:
#             prompt_format (str): Format string for the prompt. Default is '{}'. The format string should
#                 contain a single pair of curly braces {} where the text will be inserted.
#             neg_samples (Tuple[int, int]): A range to randomly sample negative texts. The first integer
#                 specifies the minimum number of negative samples, and the second integer specifies the
#                 maximum. Default is (80, 80).
#             max_samples (int): The maximum number of different text samples in one image. Default is 80.
#             padding (bool): Whether to pad texts to max_samples. If True, the number of texts will always
#                 be equal to max_samples. Default is False.
#             padding_value (str): The padding text to use when padding is True. Default is an empty string.

#         Attributes:
#             prompt_format (str): The format string for the prompt.
#             neg_samples (Tuple[int, int]): The range for sampling negative texts.
#             max_samples (int): The maximum number of text samples.
#             padding (bool): Whether padding is enabled.
#             padding_value (str): The value used for padding.

#         Examples:
#             >>> random_load_text = RandomLoadText(prompt_format="Object: {}", neg_samples=(50, 100), max_samples=120)
#             >>> random_load_text.prompt_format
#             'Object: {}'
#             >>> random_load_text.neg_samples
#             (50, 100)
#             >>> random_load_text.max_samples
#             120
#         """
#         self.neg_samples = neg_samples
#         self.max_samples = max_samples
#         self.padding = padding
#         self.padding_value = padding_value

#     def __call__(self, labels: dict) -> dict:
#         """
#         Randomly samples positive and negative texts and updates class indices accordingly.

#         This method samples positive texts based on the existing class labels in the image, and randomly
#         selects negative texts from the remaining classes. It then updates the class indices to match the
#         new sampled text order.

#         Args:
#             labels (Dict): A dictionary containing image labels and metadata. Must include 'texts' and 'cls' keys.

#         Returns:
#             (Dict): Updated labels dictionary with new 'cls' and 'texts' entries.

#         Examples:
#             >>> loader = RandomLoadText(prompt_format="A photo of {}", neg_samples=(5, 10), max_samples=20)
#             >>> labels = {"cls": np.array([[0], [1], [2]]), "texts": [["dog"], ["cat"], ["bird"]]}
#             >>> updated_labels = loader(labels)
#         """
#         # assert "texts" in labels, "No texts found in labels."
#         class_texts = labels["cls"]
#         num_classes = len(class_texts)
#         cls = np.asarray(labels.pop("cls"), dtype=int)
#         pos_labels = np.unique(cls).tolist()

#         if len(pos_labels) > self.max_samples:
#             pos_labels = random.sample(pos_labels, k=self.max_samples)

#         neg_samples = min(min(num_classes, self.max_samples) - len(pos_labels), random.randint(*self.neg_samples))
#         neg_labels = [i for i in range(num_classes) if i not in pos_labels]
#         neg_labels = random.sample(neg_labels, k=neg_samples)

#         sampled_labels = pos_labels + neg_labels
#         random.shuffle(sampled_labels)

#         label2ids = {label: i for i, label in enumerate(sampled_labels)}
#         valid_idx = np.zeros(len(labels["instances"]), dtype=bool)
#         new_cls = []
#         for i, label in enumerate(cls.squeeze(-1).tolist()):
#             if label not in label2ids:
#                 continue
#             valid_idx[i] = True
#             new_cls.append([label2ids[label]])
#         labels["instances"] = labels["instances"][valid_idx]
#         labels["cls"] = np.array(new_cls)

#         # Randomly select one prompt when there's more than one prompts
#         # texts = []
#         # for label in sampled_labels:
#         #     prompts = class_texts[label]
#         #     assert len(prompts) > 0
#             # prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])
#             # texts.append(prompt)

#         # if self.padding:
#         #     valid_labels = len(pos_labels) + len(neg_labels)
#         #     num_padding = self.max_samples - valid_labels
#         #     if num_padding > 0:
#         #         texts += [self.padding_value] * num_padding

#         # labels["texts"] = texts
#         return labels

class RandomLoadClass:
    """
    Randomly samples positive and negative texts and updates class indices accordingly.

    This class is responsible for sampling texts from a given set of class texts, including both positive
    (present in the image) and negative (not present in the image) samples. It updates the class indices
    to reflect the sampled texts and can optionally pad the text list to a fixed length.

    Attributes:
        prompt_format (str): Format string for text prompts.
        neg_samples (Tuple[int, int]): Range for randomly sampling negative texts.
        max_samples (int): Maximum number of different text samples in one image.
        padding (bool): Whether to pad texts to max_samples.
        padding_value (str): The text used for padding when padding is True.

    Methods:
        __call__: Processes the input labels and returns updated classes and texts.

    Examples:
        >>> loader = RandomLoadText(prompt_format="Object: {}", neg_samples=(5, 10), max_samples=20)
        >>> labels = {"cls": [0, 1, 2], "texts": [["cat"], ["dog"], ["bird"]], "instances": [...]}
        >>> updated_labels = loader(labels)
        >>> print(updated_labels["texts"])
        ['Object: cat', 'Object: dog', 'Object: bird', 'Object: elephant', 'Object: car']
    """

    def __init__(
        self,
        prompt_format: str = "{}",
        neg_samples: Tuple[int, int] = (80, 80),
        max_samples: int = 80,
        padding: bool = False,
        padding_value: List[str] = [""],
    ) -> None:
        """
        Initializes the RandomLoadText class for randomly sampling positive and negative texts.

        This class is designed to randomly sample positive texts and negative texts, and update the class
        indices accordingly to the number of samples. It can be used for text-based object detection tasks.

        Args:
            prompt_format (str): Format string for the prompt. Default is '{}'. The format string should
                contain a single pair of curly braces {} where the text will be inserted.
            neg_samples (Tuple[int, int]): A range to randomly sample negative texts. The first integer
                specifies the minimum number of negative samples, and the second integer specifies the
                maximum. Default is (80, 80).
            max_samples (int): The maximum number of different text samples in one image. Default is 80.
            padding (bool): Whether to pad texts to max_samples. If True, the number of texts will always
                be equal to max_samples. Default is False.
            padding_value (str): The padding text to use when padding is True. Default is an empty string.

        Attributes:
            prompt_format (str): The format string for the prompt.
            neg_samples (Tuple[int, int]): The range for sampling negative texts.
            max_samples (int): The maximum number of text samples.
            padding (bool): Whether padding is enabled.
            padding_value (str): The value used for padding.

        Examples:
            >>> random_load_text = RandomLoadText(prompt_format="Object: {}", neg_samples=(50, 100), max_samples=120)
            >>> random_load_text.prompt_format
            'Object: {}'
            >>> random_load_text.neg_samples
            (50, 100)
            >>> random_load_text.max_samples
            120
        """
        self.prompt_format = prompt_format
        self.neg_samples = neg_samples
        self.max_samples = max_samples
        self.padding = padding
        self.padding_value = padding_value

    def __call__(self, labels: dict) -> dict:
        """
        Randomly samples positive and negative texts and updates class indices accordingly.

        This method samples positive texts based on the existing class labels in the image, and randomly
        selects negative texts from the remaining classes. It then updates the class indices to match the
        new sampled text order.

        Args:
            labels (dict): A dictionary containing image labels and metadata. Must include 'texts' and 'cls' keys.

        Returns:
            (dict): Updated labels dictionary with new 'cls' and 'texts' entries.

        Examples:
            >>> loader = RandomLoadText(prompt_format="A photo of {}", neg_samples=(5, 10), max_samples=20)
            >>> labels = {"cls": np.array([[0], [1], [2]]), "texts": [["dog"], ["cat"], ["bird"]]}
            >>> updated_labels = loader(labels)
        """
        # assert "texts" in labels, "No texts found in labels."
        # class_texts = labels["texts"]
        num_classes = self.max_samples
        class_cls = np.asarray(labels.pop("cls"), dtype=int)
        pos_labels = np.unique(class_cls).tolist()

        if len(pos_labels) > self.max_samples:
            pos_labels = random.sample(pos_labels, k=self.max_samples)

        neg_samples = min(min(num_classes, self.max_samples) - len(pos_labels), random.randint(*self.neg_samples))
        neg_labels = [i for i in range(num_classes) if i not in pos_labels]
        neg_labels = random.sample(neg_labels, k=neg_samples)

        sampled_labels = pos_labels + neg_labels
        # Randomness
        # random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}
        valid_idx = np.zeros(len(labels["instances"]), dtype=bool)
        new_cls = []
        for i, label in enumerate(class_cls.squeeze(-1).tolist()):
            if label not in label2ids:
                continue
            valid_idx[i] = True
            new_cls.append([label2ids[label]])
        labels["instances"] = labels["instances"][valid_idx]
        labels["cls"] = np.array(object=new_cls)
        

        # Randomly select one prompt when there's more than one prompts
        # texts = []
        # for label in sampled_labels:
        #     prompts = class_texts[label]
        #     assert len(prompts) > 0
        #     prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])
        #     texts.append(prompt)

        if self.padding:
            valid_labels = len(pos_labels) + len(neg_labels)
            num_padding = self.max_samples - valid_labels
            # if num_padding > 0:
            #     texts += random.choices(self.padding_value, k=num_padding)

        # assert len(texts) == self.max_samples
        # labels["texts"] = texts
        labels["sampled_cls"] = np.array(sampled_labels, dtype=int)
        labels["pos_labels"] = np.array(pos_labels, dtype=int)
        labels["label2ids"] = label2ids
        
        return labels

class v2vFormat(Format):
    def __init__(self, *args,
                 return_mask = True, 
                 mask_ratio=1, 
                 mask_overlap = False,
                 **kwargs):
        
        super().__init__(*args, 
                         return_mask = True, 
                         mask_ratio=1,
                         mask_overlap = False,
                         **kwargs)
    
    def __call__(self, labels):
        labels = super().__call__(labels)
        cls_list = list(int(gg) for gg in labels['cls'])
        positions_list = [i for i, x in enumerate(cls_list) if x == labels['wanted_cls']]
        
        max_area = 0
        for positions in positions_list:
            temp_area = labels['bboxes'][positions][2] * labels['bboxes'][positions][3]
            if temp_area > max_area:
                max_area = temp_area
                max_position = positions
        
        labels['maximum_pos'] = max_position
        labels['wanted_cls_maximum_bbox'] = labels['bboxes'][max_position]
        labels['wanted_cls_maximum_mask'] = labels['masks'][max_position]

        return labels

class PasteTemplateIntoRandomImage:
    """
    Paste template into random image.
    """

    def __init__(self,
                 paste_ratio: float = 0.5,
                 paste_overlap: bool = False,
                 paste_random: bool = True,
                 paste_template: bool = True,
                 paste_template_ratio: float = 1.0,
                 paste_template_overlap: bool = False,
                 paste_template_random: bool = True):
        """
        Args:
            paste_ratio (float): The ratio of the template to be pasted into the image.
            paste_overlap (bool): Whether to allow overlap between the pasted template and the image.
            paste_random (bool): Whether to randomly select the position to paste the template.
            paste_template (bool): Whether to paste the template into the image.
            paste_template_ratio (float): The ratio of the template to be pasted into the image.
            paste_template_overlap (bool): Whether to allow overlap between the pasted template and the image.
            paste_template_random (bool): Whether to randomly select the position to paste the template.
        """
        self.paste_ratio = paste_ratio
        self.paste_overlap = paste_overlap
        self.paste_random = paste_random
        self.paste_template = paste_template
        self.paste_template_ratio = paste_template_ratio
        self.paste_template_overlap = paste_template_overlap
        self.paste_template_random = paste_template_random
    
    def __call__(self, labels: dict) -> dict:
        """
        Paste template into random image.

        Args:
            labels (dict): The labels of the image.

        Returns:
            dict: The labels of the image with the template pasted into it.
        """
        pass

class new_Albumentations(Albumentations):
    def __init__(self, p=1.0, T=None):
        """
        Initialize the Albumentations transform object for YOLO bbox formatted parameters.

        This class applies various image augmentations using the Albumentations library, including Blur, Median Blur,
        conversion to grayscale, Contrast Limited Adaptive Histogram Equalization, random changes of brightness and
        contrast, RandomGamma, and image quality reduction through compression.

        Args:
            p (float): Probability of applying the augmentations. Must be between 0 and 1.

        Attributes:
            p (float): Probability of applying the augmentations.
            transform (albumentations.Compose): Composed Albumentations transforms.
            contains_spatial (bool): Indicates if the transforms include spatial transformations.

        Raises:
            ImportError: If the Albumentations package is not installed.
            Exception: For any other errors during initialization.

        Examples:
            >>> transform = Albumentations(p=0.5)
            >>> augmented = transform(image=image, bboxes=bboxes, class_labels=classes)
            >>> augmented_image = augmented["image"]
            >>> augmented_bboxes = augmented["bboxes"]

        Notes:
            - Requires Albumentations version 1.0.3 or higher.
            - Spatial transforms are handled differently to ensure bbox compatibility.
            - Some transforms are applied with very low probability (0.01) by default.
        """
        self.p = p
        self.transform = None
        prefix = colorstr("Los Santos Custom Albumentations: ")

        try:
            import os

            os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # suppress Albumentations upgrade message
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            # List of possible spatial transforms
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

            if T is None:
            # Transforms
                T = [
                    A.Blur(p=0.01),
                    A.MedianBlur(p=0.01),
                    A.ToGray(p=0.01),
                    A.CLAHE(p=0.01),
                    A.RandomBrightnessContrast(p=0.0),
                    A.RandomGamma(p=0.0),
                    A.ImageCompression(quality_range=(75, 100), p=0.0),
                ]

            # Compose transforms
            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
            self.transform = (
                A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
                if self.contains_spatial
                else A.Compose(T)
            )
            if hasattr(self.transform, "set_random_seed"):
                # Required for deterministic transforms in albumentations>=1.4.21
                self.transform.set_random_seed(torch.initial_seed())
            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")


def v2v_transforms(dataset, imgsz, hyp, stretch=False):
    """
    Applies a series of image transformations for training.

    This function creates a composition of image augmentation techniques to prepare images for YOLO training.
    It includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (Namespace): A dictionary of hyperparameters controlling various aspects of the transformations.
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.

    Returns:
        (Compose): A composition of image transformations to be applied to the dataset.

    Examples:
        >>> from ultralytics.data.dataset import YOLODataset
        >>> from ultralytics.utils import IterableSimpleNamespace
        >>> dataset = YOLODataset(img_path="path/to/images", imgsz=640)
        >>> hyp = IterableSimpleNamespace(mosaic=1.0, copy_paste=0.5, degrees=10.0, translate=0.2, scale=0.9)
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )

    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")
    
    import albumentations as A
    Albumentations_T = [
        A.Blur(blur_limit=3, p=0.5),  # 50%機率應用輕微模糊
        A.MedianBlur(blur_limit=3, p=0.2),  # 20%機率應用中值模糊
        A.CLAHE(clip_limit=2.0, p=0.5),  # 50%機率應用CLAHE增強對比度
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # 50%機率調整亮度和對比度
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # 30%機率調整伽瑪值
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),  # 30%機率應用壓縮
    ]

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            new_Albumentations(p=1.0, T=Albumentations_T),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms