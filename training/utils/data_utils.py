# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
import torch

from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    metadata: BatchedVideoMetaData

    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)


@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapoint:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                step_t_masks[t].append(obj.segment.to(torch.bool))
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))
    # print("step_t_frame_orig_size:", step_t_frame_orig_size)
    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    # print("=================================================")
    # print("step_t_obj_to_frame_idx:", step_t_obj_to_frame_idx)
    # print("=================================================")
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        dict_key=dict_key,
        batch_size=[T],
    )


######################################################################################################

@dataclass
class Frame_yolo:
    data: Union[torch.Tensor, PILImage.Image]
    # objects: List[Object]
    classes: List[Union[int, str]]
    bboxes: List[Tuple[float, float, float, float]]
    scores: List[float]                            # 置信度分數
    # truncation: List[int]                           # 截斷標註 (0/1)
    # occlusion: List[int]                            # 遮擋標註 (0/1/2)


@dataclass
class VideoDatapoint_yolo:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame_yolo]
    video_id: int
    size: Tuple[int, int]

@tensorclass
class BatchedVideoMetaData_yolo:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """
    frame_orig_size: torch.LongTensor

# @dataclass
# class BatchedVideoGtData_yolo:
#     obj_to_frame_idx: torch.IntTensor # size is [X*2], X is total objects number, 2 is [frame id, video id]
#     classes: torch.FloatTensor # size is [X]
#     bboxes: torch.FloatTensor # size is [X*4], 4 is normalize [x, y, w, h]
#     scores: torch.FloatTensor # size is [X]

@tensorclass
class BatchedVideoDatapoint_yolo:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        gtdata: A dictionary containing ground truth data, including:
            - batch_idx: A [N] tensor where N is the number of objects, representing the batch index.
            - cls: A [N] tensor containing class indices for each object.
            - bboxes: A [N, 4] tensor representing bounding boxes for each object.
            - scores: (Optional) A [N] tensor representing confidence scores for each object.

        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    metadata: BatchedVideoMetaData_yolo
    gtdata: Dict[str, torch.Tensor]

    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a 2D tensor containing the object-to-image index
        where each row corresponds to a frame, and each column corresponds to a video.
        """
        # 每个视频的帧数和视频数
        num_frames = self.num_frames
        num_videos = self.num_videos

        # 第 0 维是帧索引，第 1 维是视频索引
        frame_idx = torch.arange(num_frames, device=self.img_batch.device).unsqueeze(1)  # [num_frames, 1]
        video_idx = torch.arange(num_videos, device=self.img_batch.device).unsqueeze(0)  # [1, num_videos]

        # 展平索引公式：flat_idx = video_idx * num_frames + frame_idx
        flat_idx = video_idx * num_frames + frame_idx  # 广播到 [num_frames, num_videos]

        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)

def collate_fn_yolo(
    batch: List[VideoDatapoint_yolo],
    dict_key,
) -> BatchedVideoDatapoint_yolo:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    # # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    # step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    # step_t_masks = [[] for _ in range(T)]
    # step_t_obj_to_frame_idx = [
    #     [] for _ in range(T)
    # ]  # List to store frame indices for each time step
    step_t_classes = []
    step_t_bbox = []
    step_t_ori_shape = []
    # step_t_score = []
    # step_t_obj_to_frame_idx = [] # origin yolo
    batch_idx = []

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        # print("orig_frame_size:", orig_frame_size)
        for t, frame in enumerate(video.frames):
            # bboxes = frame.bboxes
            # print("frame.bboxes:", frame.bboxes)
            for idx in range(len(frame.bboxes)):
                if frame.scores[idx] == 0:
                    continue
                # orig_obj_id = obj.object_id
                # orig_frame_idx = obj.frame_index
                # step_t_obj_to_frame_idx[t].append(
                #     torch.tensor([t, video_idx], dtype=torch.int)
                # )
                # step_t_obj_to_frame_idx.append(
                #     torch.tensor([t, video_idx], dtype=torch.int)
                # )
                # step_t_masks[t].append(obj.segment.to(torch.bool))
                # step_t_objects_identifier[t].append(
                #     torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                # )
                batch_idx.append(video_idx * T + t)
                step_t_classes.append(frame.classes[idx]-1)
                step_t_bbox.append(torch.tensor(frame.bboxes[idx], dtype=torch.float32))
                step_t_ori_shape.append([orig_frame_size[0], orig_frame_size[1]])
                # step_t_score.append(torch.tensor(frame.scores[idx], dtype=torch.float32))
            step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))
            # print("orig_frame_size:", orig_frame_size)
    # print("step_t_obj_to_frame_idx:", step_t_obj_to_frame_idx)
    # obj_to_frame_idx = torch.stack(step_t_obj_to_frame_idx, dim=0)  # Shape: [x, 2]
    batch_idx = torch.tensor(batch_idx, dtype=torch.float32)
    classes_stack = torch.tensor(step_t_classes, dtype=torch.float32)  # Shape: [x]
    bbox_stack = torch.stack(step_t_bbox, dim=0)  # Shape: [x, 4]
    # score_stack = torch.stack(step_t_score, dim=0)  # Shape: [x]

    # masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    # objects_identifier = torch.stack(
    #     [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    # )
    # print("step_t_frame_orig_size:", step_t_frame_orig_size)
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    return BatchedVideoDatapoint_yolo(
        img_batch=img_batch,
        # obj_to_frame_idx=obj_to_frame_idx,
        # classes=classes_stack,
        # bboxes=bbox_stack,
        # scores=score_stack,
        metadata=BatchedVideoMetaData_yolo(
            # unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size
        ),
        # gtdata=BatchedVideoGtData_yolo(
        #     obj_to_frame_idx=obj_to_frame_idx,
        #     classes=classes_stack,
        #     bboxes=bbox_stack,
        #     scores=score_stack
        # ),
        gtdata={
            "batch_idx": batch_idx,
            "cls": classes_stack,
            "bboxes": bbox_stack,
            "ori_shape": step_t_ori_shape,
        },
        dict_key=dict_key,
        batch_size=[T],
    )