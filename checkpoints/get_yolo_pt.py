from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_base_yolo import SAM2Base_yolo

import torch
import torch.nn as nn
from sam2.modeling.sam.detection_head import Detection_head

from iopath.common.file_io import g_pathmgr

# 遞迴過濾函數
def filter_nested_state_dict(state_dict, remove_prefix="sam_mask_decoder"):
    """
    過濾掉 state_dict 中帶有特定前綴的鍵，支持嵌套結構。
    Args:
        state_dict: 原始的 state_dict。
        remove_prefix: 要移除的鍵前綴。

    Returns:
        過濾後的 state_dict。
    """
    if isinstance(state_dict, dict):
        filtered_dict = {}
        for k, v in state_dict.items():
            # 如果當前鍵不包含 remove_prefix，則保留
            if not k.startswith(remove_prefix):
                # 如果值是嵌套字典，進行遞迴處理
                filtered_dict[k] = filter_nested_state_dict(v, remove_prefix)
        return filtered_dict
    else:
        # 非字典結構直接返回
        return state_dict

# # 原始模型
# old_model = SAM2Base()
# old_state_dict = old_model.state_dict()
# torch.save(old_state_dict, "old_model.pt")

# 加載舊模型參數到新模型
# state_dict = torch.load("./checkpoints/sam2.1_hiera_base_plus.pt", map_location="cpu")
ckpt_path = "/home/si2/sdragon/sam2/sam2_logs/configs/sam2.1_training/yolo_sam2.1_hiera_b+_MOSE_finetune.yaml/checkpoints/checkpoint.pt"
with g_pathmgr.open(ckpt_path, "rb") as f:
    state_dict = torch.load(f, map_location="cpu")
# # 過濾掉嵌套結構中的 "sam_mask_decoder"
# filtered_state_dict = filter_nested_state_dict(state_dict, remove_prefix="sam_mask_decoder")

# # for k, v in filtered_state_dict.items():
# #     for k1, v1 in v.items():
# #         print(k1)

# # 保存新的模型檔案
# torch.save(filtered_state_dict, "./checkpoints/no_sam_mask_decoder_sam2.1_hiera_base_plus.pt")


yolo_detection_head = Detection_head(
            # transformer_dim=self.sam_prompt_embed_dim,
            transformer_dim=256
        )

# 3. 提取 yolo_detection_head 的參數
yolo_detection_head_state_dict = yolo_detection_head.state_dict()

# 使用兩層迴圈插入 yolo_detection_head 的參數
for k, v in state_dict.items():  # 第一層
    if isinstance(v, dict):  # 如果值是字典，進入第二層
        for k1 in v.keys():  # 第二層鍵
            # 在合適的層（例如 "model"）中插入 yolo_detection_head
            if "model" in k:  # 假設 "model" 是主要的層
                for param_name, param_value in yolo_detection_head_state_dict.items():
                    v[f"yolo_detection_head.{param_name}"] = param_value
                break  # 插入後結束第二層迴圈
        break  # 插入後結束第一層迴圈

# for k, v in state_dict.items():
#     for k1, v1 in v.items():
#         print(k1)
loss = state_dict["loss"]
print("loss:", loss)
for k, v in loss.items():
    print(f"k: {k}, v: {v}")

# 保存新的 state_dict
# torch.save(state_dict, "./checkpoints/yolo_sam2.1_hiera_base_plus.pt")
