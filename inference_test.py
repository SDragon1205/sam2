import torch
import numpy as np
import time

# start = time.time()
# end = time.time()
# print(f"耗時: {end - start:.4f} 秒")
# MODEL_YAML2 = "/home/user/sdragon/sam2/sam2/configs/abo/stream_tt20_oom11_s_mode2.yaml"

# MODEL_YAML2 = "/home/user/sdragon/sam2/sam2/configs/abo/tt20_oom11_s_mode2_init_cond_frames_mode_1.yaml"
# CKPT_NAME2 = '/home/user/sdragon/sam2/checkpoints/oom11_s_num_maskmem_1_memory_position_0_no_mask_downsampler_downsampler_attentionlayer_1.pt'

MODEL_YAML2 = "/home/user/sdragon/sam2/sam2/configs/abo/20_tt20_oom11_s_mode2_freeze_num_maskmem_1_before_neck_memory_position_0_no_mask_downsampler_pos_enc_at_attn_num_layers_1.yaml"
CKPT_NAME2 = "/home/user/sdragon/sam2/sam2_logs/configs/abo/20_tt20_oom11_s_mode2_freeze_num_maskmem_1_before_neck_memory_position_0_no_mask_downsampler_pos_enc_at_attn_num_layers_1.yaml/checkpoints/best.pt"

def load_model_from_config(MODEL_YAML2, CKPT_NAME2):
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from hydra import compose
    from sam2.build_sam import _load_checkpoint
    # config_file = "/home/user/sdragon/sam2/sam2/configs/abo/tt20_oom11_s_mode2.yaml"
    cfg = OmegaConf.load(MODEL_YAML2)
    model_cfg = OmegaConf.to_container(cfg.trainer.model, resolve=True)
    model2 = instantiate(model_cfg, _convert_="all")
    # print("model2:", model2)
    _load_checkpoint(model2, CKPT_NAME2)
    return model2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model_from_config(MODEL_YAML2, CKPT_NAME2).to(device) #.eval()
# model.yolo.detection_model.training=False
# model = model.half()
test = torch.zeros((1, 3, 640, 640), device=device) #.half()
print("device:", device)
print("=======================================")
print("000000")
start = time.time()
output0 = model(test)
end = time.time()
print(f"耗時: {end - start:.4f} 秒")
# print("output0:", output0[0].shape, output0[1].shape, output0[2].shape)
# print("output0[0]:", output0[0].shape)
# print("output0[1]:", output0[1][0].shape, output0[1][1].shape, output0[1][2].shape)
print("=======================================")
print("111111")
start = time.time()
output1 = model(test)
end = time.time()
print(f"耗時: {end - start:.4f} 秒")
print("=======================================")
print("222222")
start = time.time()
output2 = model(test)
end = time.time()
print(f"耗時: {end - start:.4f} 秒")
print("=======================================")
print("333333")
start = time.time()
output3 = model(test)
end = time.time()
print(f"耗時: {end - start:.4f} 秒")