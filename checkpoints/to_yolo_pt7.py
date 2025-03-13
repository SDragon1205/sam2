import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention
from iopath.common.file_io import g_pathmgr
from typing import List, Optional, Tuple, Type
import sys
# ckpt_path = "/home/si2/sdragon/sam2/sam2_logs/configs/sam2.1_training/yolo_sam2.1_hiera_b+_MOSE_finetune.yaml/checkpoints/checkpoint.pt"
ckpt_path = "/home/si2/sdragon/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
# ckpt_path = "/home/si2/sdragon/sam2/sam2_logs/configs/sam2.1_training/yolom_s_num_maskmem_1_memory_position_0_no_mask_downsampler.yaml/checkpoints/checkpoint_old.pt"
# ckpt_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s_m_num_maskmem_1_memory_position_0_no_mask_downsampler_old.pt"
with g_pathmgr.open(ckpt_path, "rb") as f:
    state_dict = torch.load(f, map_location="cpu")

state_dict_copy = state_dict.copy()

# 使用兩層迴圈插入 yolo_detection_head 的參數
for k, v in state_dict.items():  # 第一層
    if isinstance(v, dict):  # 如果值是字典，進入第二層
        # for k1 in v.keys():  # 第二層鍵
        #     # if "model" in k:  # 假設 "model" 是主要的層
        #     print(k1)
        # 過濾掉所有 "image_encoder.xxx" 的參數
        filtered_v = {k1: v1 for k1, v1 in v.items() if not (k1.startswith("memory_attention.") or k1.startswith("memory_encoder.") or k1.startswith("memory_encoder.mask_downsampler.") or k1.startswith("mask_downsample.") or k1.startswith("no_obj_ptr") or k1.startswith("image_encoder.") or k1.startswith("sam_prompt_encoder.") or k1.startswith("yolo_detection_head.") or k1.startswith("sam_mask_decoder.") or k1.startswith("obj_ptr_proj") or k1.startswith("obj_ptr_tpos_proj.") or k1.startswith("no_obj_embed_spatial"))}
        
        # 如果 `filtered_v` 變成空的，就直接刪掉 `k`
        if not filtered_v:
            del state_dict_copy[k]
        else:
            state_dict_copy[k] = filtered_v
    # else:
    #     print("other:", k)

# for k, v in state_dict_copy.items():  # 第一層
#     if isinstance(v, dict):  # 如果值是字典，進入第二層
#         for k1 in v.keys():  # 第二層鍵
#             # if "model" in k:  # 假設 "model" 是主要的層
#             print(k1)
#     else:
#         print("other:", k, v)

ckpt_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s.pt"
with g_pathmgr.open(ckpt_path, "rb") as f:
    state_dict_yolo = torch.load(f, map_location="cpu")

# for k, v in state_dict_yolo['model'].state_dict().items():  # 第一層
#     print(k)
from torch.nn.init import trunc_normal_
hidden_dim = 512
no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
trunc_normal_(no_mem_embed, std=0.02)
# print("self.no_mem_embed:", no_mem_embed.shape, no_mem_embed)
trunc_normal_(no_mem_pos_enc, std=0.02)
# print("self.no_mem_pos_enc:", no_mem_pos_enc.shape, no_mem_pos_enc)

    # memory_encoder:
    #     _target_: sam2.modeling.memory_encoder.MemoryEncoder
    #     out_dim: 64
    #     position_encoding:
    #       _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
    #       num_pos_feats: 64
    #       normalize: true
    #       scale: null
    #       temperature: 10000
    #     mask_downsampler:
    #       _target_: sam2.modeling.memory_encoder.MaskDownSampler
    #       kernel_size: 3
    #       stride: 2
    #       padding: 1
    #       in_chans: ${scratch.nc}
    #     fuser:
    #       _target_: sam2.modeling.memory_encoder.Fuser
    #       layer:
    #         _target_: sam2.modeling.memory_encoder.CXBlock
    #         dim: 512
    #         kernel_size: 7
    #         padding: 3
    #         layer_scale_init_value: 1e-6
    #         use_dwconv: True  # depth-wise convs
    #       num_layers: 2
position_encoding = PositionEmbeddingSine(num_pos_feats=512,normalize= True,scale=None, temperature= 10000)
mask_downsampler = MaskDownSampler(embed_dim=512, in_chans= 74, has_mask_down=False)
layer = CXBlock(dim= 512,kernel_size= 7,padding= 3,layer_scale_init_value= 1e-6,use_dwconv= True)
fuser=Fuser(layer=layer, num_layers= 2)
memory_encoder = MemoryEncoder(out_dim= 512, position_encoding=position_encoding, mask_downsampler=mask_downsampler, fuser=fuser, in_dim=512, only_pos=True)#no_mask_downsampler= True)#, only_pos=True)
# for k, v in memory_encoder.state_dict().items():  # 第一層
#     print(k)
memory_attention= MemoryAttention(
    d_model= 512,
    pos_enc_at_input= False,
    layer=MemoryAttentionLayer(
        activation= "relu",
        dim_feedforward= 2048,
        dropout= 0.1,
        pos_enc_at_attn= True,
        self_attention=RoPEAttention(
            rope_theta= 10000.0,
            feat_sizes= [20, 20],
            embedding_dim= 512,
            num_heads= 1,
            downsample_rate= 1,
            dropout= 0.1
        ),
        d_model= 512,
        pos_enc_at_cross_attn_keys= True,
        pos_enc_at_cross_attn_queries= False,
        cross_attention=RoPEAttention(
            rope_theta= 10000.0,
            feat_sizes= [20, 20],
            rope_k_repeat= False,
            embedding_dim= 512,
            num_heads= 1,
            downsample_rate= 1,
            dropout= 0.1,
            # kv_in_dim= 512
        ),
    ),
    num_layers= 1
)
# for k, v in memory_attention.state_dict().items():  # 第一層
#     print(k)
# # 使用兩層迴圈插入 yolo_detection_head 的參數
# transformer_dim=512
# activation = nn.GELU
# output_upscaling = nn.Sequential(
#                     nn.ConvTranspose2d(
#                         transformer_dim, transformer_dim // 2, kernel_size=2, stride=2
#                     ),
#                     nn.GroupNorm(num_groups=32, num_channels=transformer_dim // 2),
#                     activation(),
#                     nn.ConvTranspose2d(
#                         transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2
#                     ),
#                     activation(),
#                 )
# # for k, v in output_upscaling.state_dict().items():  # 第一層
# #     print(k)
# # sys.exit()

# Temporal encoding of the memories
num_maskmem = 1
mem_dim = 512
maskmem_tpos_enc = torch.nn.Parameter(
    torch.zeros(num_maskmem, 1, 1, mem_dim)
)
trunc_normal_(maskmem_tpos_enc, std=0.02)

for k, v in state_dict_copy.items():  # 第一層
    if isinstance(v, dict):  # 如果值是字典，進入第二層
        for k1 in v.keys():  # 第二層鍵
            # 在合適的層（例如 "model"）中插入 yolo_detection_head
            if "model" in k:  # 假設 "model" 是主要的層
                for param_name, param_value in state_dict_yolo['model'].state_dict().items():
                    v[f"yolo.detection_model.{param_name}"] = param_value
                # for param_name, param_value in state_dict_yolo['model'].state_dict().items():
                #     v[f"freeze_model.model.{param_name}"] = param_value
                v["no_mem_embed"] = no_mem_embed
                v["no_mem_pos_enc"] = no_mem_pos_enc
                for param_name, param_value in memory_encoder.state_dict().items():
                    v[f"memory_encoder.{param_name}"] = param_value
                for param_name, param_value in memory_attention.state_dict().items():
                    v[f"memory_attention.{param_name}"] = param_value
                # for param_name, param_value in output_upscaling.state_dict().items():
                #     v[f"yolo.output_upscaling.{param_name}"] = param_value
                v["maskmem_tpos_enc"] = maskmem_tpos_enc
                break  # 插入後結束第二層迴圈
        break  # 插入後結束第一層迴圈
for k, v in state_dict_copy.items():  # 第一層
    if isinstance(v, dict):  # 如果值是字典，進入第二層
        for k1 in v.keys():  # 第二層鍵
            # if "model" in k:  # 假設 "model" 是主要的層
            print(k1)
    else:
        print("other:", k, v)

# output_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s_m_num_maskmem_1_memory_position_0_self_memory_encode.pt"
# output_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s_m_num_maskmem_1_memory_position_0_no_mask_downsampler.pt"
# output_path = "/home/si2/sdragon/sam2/sam2_logs/configs/sam2.1_training/yolom_s_num_maskmem_1_memory_position_0_no_mask_downsampler.yaml/checkpoints/checkpoint.pt"
# output_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s_m_num_maskmem_1_memory_position_2_no_mask_downsampler_attentionlayer_1.pt"
output_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s_m_num_maskmem_1_memory_position_2_only_pos_attentionlayer_1.pt"


# 儲存新的模型權重
torch.save(state_dict_copy, output_path)

print(f"Filtered checkpoint saved to: {output_path}")