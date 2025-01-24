import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import logging
import math
import os
import torch.distributed as dist
from datetime import timedelta
from training.utils.data_utils import BatchedVideoDatapoint
# from training.utils.train_utils import setup_distributed_backend
# from pycocotools import mask as mask_utils
HYDRA_FULL_ERROR=1

def setup_distributed_backend(backend, timeout_mins):
    """
    Initialize torch.distributed and set the CUDA device.
    Expects environment variables to be set as per
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    along with the environ variable "LOCAL_RANK" which is used to set the CUDA device.
    """
    # enable TORCH_NCCL_ASYNC_ERROR_HANDLING to ensure dist nccl ops time out after timeout_mins
    # of waiting
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    logging.info(f"Setting up torch.distributed with a timeout of {timeout_mins} mins")
    dist.init_process_group(backend=backend, init_method='tcp://127.0.0.1:29500', world_size = 1, rank = 0, timeout=timedelta(minutes=timeout_mins))
    return dist.get_rank()

def print_batched_video_datapoint_details(batched_video: BatchedVideoDatapoint):
    # 打印基本 batch 資訊
    print("=" * 80)
    print("Batched Video Datapoint Details")
    print("=" * 80)
    print(f"Batch Size: {batched_video.batch_size}")
    print(f"Number of Videos: {batched_video.num_videos}")
    print(f"Number of Frames per Video: {batched_video.num_frames}")
    print(f"Image Batch Shape: {batched_video.img_batch.shape}")
    print(f"Mask Shape: {batched_video.masks.shape}")
    print(f"Dictionary Key: {batched_video.dict_key}")
    print(f"obj_to_frame_idx: {batched_video.obj_to_frame_idx}")
    print("=" * 80)

    # 提取 metadata 資料
    metadata = batched_video.metadata
    print("Metadata Details:")
    print(f"Frame Original Size: {metadata.frame_orig_size}")
    print(f"Unique Objects Identifier: {metadata.unique_objects_identifier}")
    print("=" * 80)

    # 遍歷每個影片
    for video_idx in range(batched_video.num_videos):
        print(f"Video {video_idx + 1}/{batched_video.num_videos}:")
        print("-" * 80)

        # 打印每個影片的基本資訊
        unique_objects_identifier = metadata.unique_objects_identifier.tolist()
        video_id = metadata.unique_objects_identifier[video_idx, :, 0].tolist()
        obj_ids = metadata.unique_objects_identifier[video_idx, :, 1].tolist()
        frame_ids = metadata.unique_objects_identifier[video_idx, :, 2].tolist()
        frame_size = metadata.frame_orig_size[video_idx].tolist()

        print(f"unique_objects_identifier: {unique_objects_identifier}")
        print(f"Video ID: {video_id}")
        print(f"Original Frame Size: {frame_size}")
        print(f"Object IDs: {obj_ids}")
        print(f"Frame IDs: {frame_ids}")
        print("-" * 40)

        # 遍歷影片的每一個 Frame
        for frame_idx in range(batched_video.num_frames):
            print(f"  Frame {frame_idx + 1}/{batched_video.num_frames}:")
            print(f"    Image Shape: {batched_video.img_batch[frame_idx, video_idx].shape}")

            # 取得當前 Frame 的物件資訊
            objects_in_frame = []
            for obj_idx in range(batched_video.masks.shape[1]):  # 遍歷所有物件
                mask = batched_video.masks[frame_idx, obj_idx]  # Mask 資訊
                obj_to_frame_idx = batched_video.obj_to_frame_idx[frame_idx, obj_idx]

                if obj_to_frame_idx[1].item() == video_idx:  # 確保是該影片的物件
                    objects_in_frame.append({
                        "Object ID": obj_ids[obj_idx],
                        "Frame Index": frame_ids[obj_idx],
                        "Mask Shape": mask.shape,
                        "Mask Sum (Active Pixels)": mask.sum().item()
                    })

            # 打印 Frame 中每個物件的資訊
            for obj in objects_in_frame:
                print(f"    Object ID: {obj['Object ID']}")
                print(f"      Frame Index: {obj['Frame Index']}")
                print(f"      Mask Shape: {obj['Mask Shape']}")
                print(f"      Active Pixels: {obj['Mask Sum (Active Pixels)']}")
            print("-" * 40)

        print("=" * 80)

@hydra.main(config_path=".", config_name="test_data", version_base="1.2")
def main(cfg: DictConfig):
    # 1. 確認 YAML 載入成功
    print(OmegaConf.to_yaml(cfg))  # 測試用，檢查設定內容

    # 2. 載入 data.train 部分
    data_conf = cfg.data  # YAML 中的 data 節點
    train_conf = data_conf.train
    distributed_conf = data_conf.distributed

    # 3. 初始化 dataset
    rank = setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )
    train_dataset = instantiate(train_conf)  # 使用 instantiate 初始化


    print("hi")

    # 4. 建立 dataloader
    dataloader = train_dataset.get_loader(epoch=0)

    # 5. 測試載入數據
    for batch in dataloader:
        print(batch)
        print_batched_video_datapoint_details(batch)
        print("flat_obj_to_img_idx:", batch.flat_obj_to_img_idx)
        # print(batch.flat_img_batch.shape)
        break


if __name__ == "__main__":
    main()