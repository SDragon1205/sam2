# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional, Union, Tuple

import pandas as pd
import cv2

import torch

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig

from training.dataset.vos_segment_loader import (
    JSONSegmentLoader,
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
    SA1BSegmentLoader,
)


@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()



class PNGRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class SA1BRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        num_frames=1,
        mask_area_frac_thresh=1.1,  # no filtering by default
        uncertain_iou=-1,  # no filtering by default
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.num_frames = num_frames
        self.mask_area_frac_thresh = mask_area_frac_thresh
        self.uncertain_iou = uncertain_iou  # stability score

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)
            subset = [
                path.split(".")[0] for path in subset if path.endswith(".jpg")
            ]  # remove extension

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files and it exists
        self.video_names = [
            video_name for video_name in subset if video_name not in excluded_files
        ]

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        video_frame_path = os.path.join(self.img_folder, video_name + ".jpg")
        video_mask_path = os.path.join(self.gt_folder, video_name + ".json")

        segment_loader = SA1BSegmentLoader(
            video_mask_path,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            video_frame_path=video_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(VOSFrame(frame_idx, image_path=video_frame_path))
        video_name = video_name.split("_")[-1]  # filename is sa_{int}
        # video id needs to be image_id to be able to load correct annotation file during eval
        video = VOSVideo(video_name, int(video_name), frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class JSONRawDataset(VOSRawDataset):
    """
    Dataset where the annotation in the format of SA-V json files
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        rm_unannotated=True,
        ann_every=1,
        frames_fps=24,
    ):
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        self.sample_rate = sample_rate
        self.rm_unannotated = rm_unannotated
        self.ann_every = ann_every
        self.frames_fps = frames_fps

        # Read and process excluded files if provided
        excluded_files = []
        if excluded_videos_list_txt is not None:
            if isinstance(excluded_videos_list_txt, str):
                excluded_videos_lists = [excluded_videos_list_txt]
            elif isinstance(excluded_videos_list_txt, ListConfig):
                excluded_videos_lists = list(excluded_videos_list_txt)
            else:
                raise NotImplementedError

            for excluded_videos_list_txt in excluded_videos_lists:
                with open(excluded_videos_list_txt, "r") as f:
                    excluded_files.extend(
                        [os.path.splitext(line.strip())[0] for line in f]
                    )
        excluded_files = set(excluded_files)

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, video_idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[video_idx]
        video_json_path = os.path.join(self.gt_folder, video_name + "_manual.json")
        segment_loader = JSONSegmentLoader(
            video_json_path=video_json_path,
            ann_every=self.ann_every,
            frames_fps=self.frames_fps,
        )

        frame_ids = [
            int(os.path.splitext(frame_name)[0])
            for frame_name in sorted(
                os.listdir(os.path.join(self.img_folder, video_name))
            )
        ]

        frames = [
            VOSFrame(
                frame_id,
                image_path=os.path.join(
                    self.img_folder, f"{video_name}/%05d.jpg" % (frame_id)
                ),
            )
            for frame_id in frame_ids[:: self.sample_rate]
        ]

        if self.rm_unannotated:
            # Eliminate the frames that have not been annotated
            valid_frame_ids = [
                i * segment_loader.ann_every
                for i, annot in enumerate(segment_loader.frame_annots)
                if annot is not None and None not in annot
            ]
            frames = [f for f in frames if f.frame_idx in valid_frame_ids]

        video = VOSVideo(video_name, video_idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)
######################################################################################################
class VOSRawDataset_yolo:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()
    
@dataclass
class VOSFrame_yolo:
    frame_idx: int
    image_path: str
    # data: Optional[torch.Tensor] = None
    # is_conditioning_only: Optional[bool] = False
    bboxes: List[Tuple[float, float, float, float]]
    classes: List[Union[int, str]]
#   <score>	      The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing 
#                 an object instance.
#                 The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in 
# 	              evaluation, while 0 indicates the bounding box will be ignored.

#   <truncation>       The score in the DETECTION file should be set to the constant -1.
#                      The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
# 	                   (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% °´ 50%)).

#   <occlusion>	      The score in the DETECTION file should be set to the constant -1.
#                     The score in the GROUNDTRUTH file indicates the fraction of objects being occluded 
# 	                  (i.e., no occlusion = 0 (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% °´ 50%), 
# 	                  and heavy occlusion = 2 (occlusion ratio 50% ~ 100%)).
    scores: List[float]                            # 置信度分數
    truncation: List[int]                           # 截斷標註 (0/1)
    occlusion: List[int]                            # 遮擋標註 (0/1/2)
    data: Optional[torch.Tensor] = None

@dataclass
class VOSVideo_yolo:
    video_name: str
    video_id: int
    frames: List[VOSFrame_yolo]
    size: Tuple[int, int]

    def __len__(self):
        return len(self.frames)

class JSONRawDataset_yolo(VOSRawDataset_yolo): # VisDrone_task2_Dataset
    """
    Dataset where the annotation in the format of visdrone task2 files
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        # sample_rate=1,
        # rm_unannotated=True,
        # ann_every=1,
        # frames_fps=24,
    ):
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        # self.sample_rate = sample_rate
        # self.rm_unannotated = rm_unannotated
        # self.ann_every = ann_every
        # self.frames_fps = frames_fps

        # Read and process excluded files if provided
        excluded_files = []
        if excluded_videos_list_txt is not None:
            if isinstance(excluded_videos_list_txt, str):
                excluded_videos_lists = [excluded_videos_list_txt]
            elif isinstance(excluded_videos_list_txt, ListConfig):
                excluded_videos_lists = list(excluded_videos_list_txt)
            else:
                raise NotImplementedError

            for excluded_videos_list_txt in excluded_videos_lists:
                with open(excluded_videos_list_txt, "r") as f:
                    excluded_files.extend(
                        [os.path.splitext(line.strip())[0] for line in f]
                    )
        excluded_files = set(excluded_files)

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, video_idx):
        """
        Given a VOSVideo object, return the frames and normalized annotations.
        """
        video_name = self.video_names[video_idx]
        video_path = os.path.join(self.img_folder, video_name)
        gt_path = os.path.join(self.gt_folder, f"{video_name}.txt")  # 標註檔案
        image_path = os.path.join(video_path, f"0000001.jpg")
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]

        # 讀取影像檔案與標註檔案
        frame_files = sorted(
            [f for f in os.listdir(video_path) if f.endswith(".jpg") or f.endswith(".png")]
        )
        with open(gt_path, "r") as f:
            lines = f.readlines()

        # 建立 frame 索引的標註對應
        annotations = {}
        for line in lines:
            parts = line.strip().split(",")
            frame_idx = int(parts[0])

            # 提取標註欄位
            target_id = int(parts[1])                  # 目標 ID (忽略，-1)
            bbox_left = float(parts[2])                # 邊界框左上角 x 座標
            bbox_top = float(parts[3])                 # 邊界框左上角 y 座標
            bbox_width = float(parts[4])               # 邊界框寬度
            bbox_height = float(parts[5])              # 邊界框高度
            score = float(parts[6])                    # 信心分數
            classes = int(parts[7])                        # 類別
            truncation = int(parts[8])                 # 截斷 (0/1)
            occlusion = int(parts[9])                  # 遮擋 (0/1/2)

            # # 取得影像尺寸以進行正規化
            # image_path = os.path.join(video_path, f"{frame_idx:06d}.jpg")
            # img = cv2.imread(image_path)
            # img_h, img_w = img.shape[:2]

            # 格式轉換並正規化 (將座標轉換為 YOLO 標準化格式)
            bbox = (
                (bbox_left + bbox_width / 2.0) / img_w,   # x_center 正規化
                (bbox_top + bbox_height / 2.0) / img_h,  # y_center 正規化
                bbox_width / img_w,                      # width 正規化
                bbox_height / img_h                      # height 正規化
            )

            if frame_idx not in annotations:
                annotations[frame_idx] = {
                    "classes": [],
                    "bboxes": [],
                    "scores": [],
                    "truncation": [],
                    "occlusion": []
                }

            annotations[frame_idx]["classes"].append(classes)
            annotations[frame_idx]["bboxes"].append(bbox)
            annotations[frame_idx]["scores"].append(score)
            annotations[frame_idx]["truncation"].append(truncation)
            annotations[frame_idx]["occlusion"].append(occlusion)

        # 建立 frame 資料
        frames = []
        for frame_file in frame_files:
            frame_idx = int(os.path.splitext(frame_file)[0])  # 取出 frame 索引
            image_path = os.path.join(video_path, frame_file)

            if frame_idx in annotations:
                anno = annotations[frame_idx]
                frame = VOSFrame_yolo(
                    frame_idx=frame_idx,
                    image_path=image_path,
                    bboxes=anno["bboxes"],  # 已正規化
                    classes=anno["classes"],
                    scores=anno["scores"],
                    truncation=anno["truncation"],
                    occlusion=anno["occlusion"]
                )
            else:
                # 若該 frame 沒有標註資料，則使用空值
                frame = VOSFrame_yolo(
                    frame_idx=frame_idx,
                    image_path=image_path,
                    bboxes=[],
                    classes=[],
                    scores=[],
                    truncation=[],
                    occlusion=[]
                )
            frames.append(frame)

        # 組裝 VOSVideo_yolo 物件
        video = VOSVideo_yolo(video_name, video_idx, frames, [img_h, img_w])
        return video

    def __len__(self):
        return len(self.video_names)
    

class SA1BRawDataset_yolo(VOSRawDataset_yolo): # VisDrone_task1_Dataset
    """
    Dataset where the annotation is in the format of VisDrone task1 files, 
    adapted for single-frame image datasets instead of video datasets.
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_files_list_txt=None,
    ):
        # 初始化資料夾路徑
        self.gt_folder = gt_folder
        self.img_folder = img_folder

        # 讀取排除的檔案 (若有提供)
        excluded_files = []
        if excluded_files_list_txt is not None:
            with open(excluded_files_list_txt, "r") as f:
                excluded_files = [
                    os.path.splitext(line.strip())[0] for line in f
                ]
        excluded_files = set(excluded_files)

        # 讀取圖片檔案清單 (若有提供篩選條件)
        if file_list_txt is not None:
            with open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = [os.path.splitext(f)[0] for f in os.listdir(self.img_folder) if f.endswith((".jpg", ".png"))]

        # 過濾排除的檔案
        self.image_names = sorted(
            [image_name for image_name in subset if image_name not in excluded_files]
        )

    def get_video(self, image_idx):
        """
        Processes a single image and its annotations as a single-frame video object.
        """
        # 取得影像與標註檔案路徑
        image_name = self.image_names[image_idx]
        image_path = os.path.join(self.img_folder, f"{image_name}.jpg")
        gt_path = os.path.join(self.gt_folder, f"{image_name}.txt")

        # 讀取影像尺寸
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]

        # 讀取標註資料
        # annotations = []
        with open(gt_path, "r") as f:
            lines = f.readlines()

        # 解析標註資料
        classes_list = []
        bbox_list = []
        scores_list = []
        truncation_list = []
        occlusion_list = []

        for line in lines:
            parts = line.strip().split(",")

            # 提取標註內容
            bbox_left = float(parts[0])                # x 左上角座標
            bbox_top = float(parts[1])                 # y 左上角座標
            bbox_width = float(parts[2])               # 寬度
            bbox_height = float(parts[3])              # 高度
            score = float(parts[4])                    # 置信度
            classes = int(parts[5])                        # 類別
            truncation = int(parts[6])                 # 截斷標註
            occlusion = int(parts[7])                  # 遮擋標註

            # 正規化 bbox 為 YOLO 格式 (0~1 範圍)
            bbox = (
                (bbox_left + bbox_width / 2.0) / img_w,   # x_center
                (bbox_top + bbox_height / 2.0) / img_h,  # y_center
                bbox_width / img_w,                      # width
                bbox_height / img_h                      # height
            )

            # 儲存標註內容
            classes_list.append(classes)
            bbox_list.append(bbox)
            scores_list.append(score)
            truncation_list.append(truncation)
            occlusion_list.append(occlusion)

        # 建立單張影像對應的 Frame_yolo 物件
        frame = VOSFrame_yolo(
            frame_idx=0,  # 單張影像視為 frame_id = 0
            image_path=image_path,
            bboxes=bbox_list,
            classes=classes_list,
            scores=scores_list,
            truncation=truncation_list,
            occlusion=occlusion_list,
        )

        # 組裝成單幀影片 (實際上是單張影像)
        video = VOSVideo_yolo(
            video_name=image_name,
            video_id=image_idx,
            frames=[frame],  # 視為只有一個 frame
            size=(img_h, img_w)
        )
        return video

    def __len__(self):
        return len(self.image_names)