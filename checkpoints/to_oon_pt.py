import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention, Attention
from iopath.common.file_io import g_pathmgr
from typing import List, Optional, Tuple, Type
import sys
from torch.nn.init import trunc_normal_

import v2vdet.v2vdet_ultralytics as new_module
sys.modules["v2vdet_ultralytics"] = new_module
from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_ObjectOriented_Model, V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented_Model, V2V_With_MultiScale_SAVPE_SigLIP2_L_ObjectOriented_Model
# ckpt_path = "/home/si2/sdragon/sam2/sam2_logs/configs/sam2.1_training/yolo_sam2.1_hiera_b+_MOSE_finetune.yaml/checkpoints/checkpoint.pt"
# ckpt_path = "/home/si2/sdragon/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
# ckpt_path = "/home/si2/sdragon/sam2/sam2_logs/configs/sam2.1_training/yolom_s_num_maskmem_1_memory_position_0_no_mask_downsampler.yaml/checkpoints/checkpoint_old.pt"
# ckpt_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s_m_num_maskmem_1_memory_position_0_no_mask_downsampler_old.pt"

ckpt_path = "yolov8s_m_num_maskmem_1_memory_position_0_no_mask_downsampler_downsampler_attentionlayer_1.pt"
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
        filtered_v = {k1: v1 for k1, v1 in v.items() if not (k1.startswith("yolo."))} #(k1.startswith("memory_attention.") or k1.startswith("memory_encoder.") or k1.startswith("memory_encoder.mask_downsampler.") or k1.startswith("mask_downsample.") or k1.startswith("no_obj_ptr") or k1.startswith("image_encoder.") or k1.startswith("sam_prompt_encoder.") or k1.startswith("yolo_detection_head.") or k1.startswith("sam_mask_decoder.") or k1.startswith("obj_ptr_proj") or k1.startswith("obj_ptr_tpos_proj.") or k1.startswith("no_obj_embed_spatial"))}
        
        # 如果 `filtered_v` 變成空的，就直接刪掉 `k`
        if not filtered_v:
            del state_dict_copy[k]
        else:
            state_dict_copy[k] = filtered_v
    # else:
    #     print("other:", k)

for k, v in state_dict.items():  # 第一層
    if isinstance(v, dict):  # 如果值是字典，進入第二層
        # for k1 in v.keys():  # 第二層鍵
        #     # if "model" in k:  # 假設 "model" 是主要的層
        #     print(k1)
        # 過濾掉所有 "image_encoder.xxx" 的參數
        filtered_v = {k1: v1 for k1, v1 in v.items() if not (k1.startswith("yolo."))}
        
        # 如果 `filtered_v` 變成空的，就直接刪掉 `k`
        if not filtered_v:
            del state_dict_copy[k]
        else:
            state_dict_copy[k] = filtered_v
    # else:
    #     print("other:", k)

# # for k, v in state_dict_copy.items():  # 第一層
# #     if isinstance(v, dict):  # 如果值是字典，進入第二層
# #         for k1 in v.keys():  # 第二層鍵
# #             # if "model" in k:  # 假設 "model" 是主要的層
# #             print(k1)
# #     else:
# #         print("other:", k, v)

ckpt_path = "/home/user/sdragon/sam2/checkpoints/OO_v11n_SAVPE_SigLIP2_FT_multi_layer_135.pt"
# ckpt_path = "/home/user/sdragon/sam2/checkpoints/OO_v11n_SAVPE_SigLIP2_FT_multi_layer_135_nc10.pt"
with g_pathmgr.open(ckpt_path, "rb") as f:
    state_dict_yolo = torch.load(f, map_location="cpu")


# # for k, v in state_dict_yolo['model'].state_dict().items():  # 第一層
# #     print(k)
# print("===============================================================")
# model = V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented_Model(cfg="/home/user/sdragon/sam2/v2vdet/v2vdet_ultralytics/cfg/models/v2v/11/yolo11n-v2v-multiscale_1_3_5.yaml", nc=10)
# model.load_state_dict(state_dict_yolo['model'].state_dict())
# print("model.nc:", model.nc)
# # model.nc = 10
# # model.model[-1].nc = 10
# # print("model.nc:", model.nc)
# # torch.save({'model': model}, "OO_v11n_SAVPE_SigLIP2_FT_multi_layer_135_nc10.pt")
# sys.exit()

# import clip
# clip_ = clip
# text_model, _  = clip_.load("ViT-B/32")#, device=self.yolo.detection_model.device)
# for p in text_model.parameters():
#     p.requires_grad_(False)

# # for k, v in text_model.state_dict().items():  # 第一層
# #     print(k)
# # sys.exit()
hidden_dim = 128
no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
trunc_normal_(no_mem_embed, std=0.02)
# print("self.no_mem_embed:", no_mem_embed.shape, no_mem_embed)
trunc_normal_(no_mem_pos_enc, std=0.02)
# print("self.no_mem_pos_enc:", no_mem_pos_enc.shape, no_mem_pos_enc)

# #     # memory_encoder:
# #     #     _target_: sam2.modeling.memory_encoder.MemoryEncoder
# #     #     out_dim: 64
# #     #     position_encoding:
# #     #       _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
# #     #       num_pos_feats: 64
# #     #       normalize: true
# #     #       scale: null
# #     #       temperature: 10000
# #     #     mask_downsampler:
# #     #       _target_: sam2.modeling.memory_encoder.MaskDownSampler
# #     #       kernel_size: 3
# #     #       stride: 2
# #     #       padding: 1
# #     #       in_chans: ${scratch.nc}
# #     #     fuser:
# #     #       _target_: sam2.modeling.memory_encoder.Fuser
# #     #       layer:
# #     #         _target_: sam2.modeling.memory_encoder.CXBlock
# #     #         dim: 512
# #     #         kernel_size: 7
# #     #         padding: 3
# #     #         layer_scale_init_value: 1e-6
# #     #         use_dwconv: True  # depth-wise convs
# #     #       num_layers: 2
position_encoding = PositionEmbeddingSine(num_pos_feats=128,normalize= True,scale=None, temperature= 10000)
mask_downsampler = MaskDownSampler(embed_dim=128, in_chans= 74, has_mask_down=False)
layer = CXBlock(dim= 128,kernel_size= 7,padding= 3,layer_scale_init_value= 1e-6,use_dwconv= True)
fuser=Fuser(layer=layer, num_layers= 2)
memory_encoder = MemoryEncoder(out_dim= 128, position_encoding=position_encoding, mask_downsampler=mask_downsampler, fuser=fuser, in_dim=128, no_mask_downsampler= True)#, only_pos=True)
# for k, v in memory_encoder.state_dict().items():  # 第一層
#     print(k)
memory_attention= MemoryAttention(
    d_model= 128,
    pos_enc_at_input= True,
    layer=MemoryAttentionLayer(
        activation= "relu",
        dim_feedforward= 2048,
        dropout= 0.1,
        pos_enc_at_attn= False,
        # self_attention=RoPEAttention(
        self_attention=Attention(
            # rope_theta= 10000.0,
            # feat_sizes= [80, 80],
            embedding_dim= 128,
            num_heads= 1,
            downsample_rate= 1,
            dropout= 0.1
        ),
        d_model= 128,
        pos_enc_at_cross_attn_keys= True,
        pos_enc_at_cross_attn_queries= False,
        # cross_attention=RoPEAttention(
        cross_attention=Attention(
            # rope_theta= 10000.0,
            # feat_sizes= [80, 80],
            # rope_k_repeat= True,
            embedding_dim= 128,
            num_heads= 1,
            downsample_rate= 1,
            dropout= 0.1,
            kv_in_dim= 128
        ),
    ),
    num_layers= 1
)
# # # for k, v in memory_attention.state_dict().items():  # 第一層
# # #     print(k)
# # # # 使用兩層迴圈插入 yolo_detection_head 的參數
# # # transformer_dim=128
# # # activation = nn.GELU
# # # output_upscaling = nn.Sequential(
# # #                     nn.ConvTranspose2d(
# # #                         transformer_dim, transformer_dim // 2, kernel_size=2, stride=2
# # #                     ),
# # #                     nn.GroupNorm(num_groups=32, num_channels=transformer_dim // 2),
# # #                     activation(),
# # #                     nn.ConvTranspose2d(
# # #                         transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2
# # #                     ),
# # #                     activation(),
# # #                 )
# # # # for k, v in output_upscaling.state_dict().items():  # 第一層
# # # #     print(k)
# # # # sys.exit()

# Temporal encoding of the memories
num_maskmem = 1
mem_dim = 128
maskmem_tpos_enc = torch.nn.Parameter(
    torch.zeros(num_maskmem, 1, 1, mem_dim)
)
trunc_normal_(maskmem_tpos_enc, std=0.02)
print("maskmem_tpos_enc Total elements:", maskmem_tpos_enc.numel())
print("maskmem_tpos_enc Number of unique values:", torch.unique(maskmem_tpos_enc).numel())

for k, v in state_dict_copy.items():  # 第一層
    if isinstance(v, dict):  # 如果值是字典，進入第二層
        for k1 in v.keys():  # 第二層鍵
            # 在合適的層（例如 "model"）中插入 yolo_detection_head
            if "model" in k:  # 假設 "model" 是主要的層
                for param_name, param_value in state_dict_yolo['model'].state_dict().items():
                    v[f"yolo.detection_model.{param_name}"] = param_value
                # # for param_name, param_value in text_model.state_dict().items():
                # #     v[f"text_model.{param_name}"] = param_value
                # # for param_name, param_value in state_dict_yolo['model'].state_dict().items():
                # #     v[f"freeze_model.model.{param_name}"] = param_value
                # v["no_mem_embed"] = no_mem_embed
                # v["no_mem_pos_enc"] = no_mem_pos_enc
                # for param_name, param_value in memory_encoder.state_dict().items():
                #     v[f"memory_encoder.{param_name}"] = param_value
                # for param_name, param_value in memory_attention.state_dict().items():
                #     v[f"memory_attention.{param_name}"] = param_value
                # # for param_name, param_value in output_upscaling.state_dict().items():
                # #     v[f"yolo.output_upscaling.{param_name}"] = param_value
                # v["maskmem_tpos_enc"] = maskmem_tpos_enc
                break  # 插入後結束第二層迴圈
        break  # 插入後結束第一層迴圈
for k, v in state_dict_copy.items():  # 第一層
    if isinstance(v, dict):  # 如果值是字典，進入第二層
        for k1 in v.keys():  # 第二層鍵
            # if "model" in k:  # 假設 "model" 是主要的層
            print(k1)
    else:
        print("other:", k, v)

# # output_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s_m_num_maskmem_1_memory_position_0_self_memory_encode.pt"
# # output_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s_m_num_maskmem_1_memory_position_0_no_mask_downsampler.pt"
# # output_path = "/home/si2/sdragon/sam2/sam2_logs/configs/sam2.1_training/yolom_s_num_maskmem_1_memory_position_0_no_mask_downsampler.yaml/checkpoints/checkpoint.pt"
# # output_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s_m_num_maskmem_1_memory_position_0_no_mask_downsampler_attentionlayer_1.pt"
# # output_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s_m_num_maskmem_39_memory_position_0_no_mask_downsampler_attentionlayer_1.pt"
# # output_path = "/home/si2/sdragon/sam2/checkpoints/yolov8s_m_num_maskmem_2_memory_position_0_no_mask_downsampler_attentionlayer_1_emcoder_out_dim_64.pt"
output_path = "oom11_n_num_maskmem_1_memory_position_0_no_mask_downsampler_downsampler_attentionlayer_1.pt"
# 儲存新的模型權重
torch.save(state_dict_copy, output_path)

print(f"Filtered checkpoint saved to: {output_path}")