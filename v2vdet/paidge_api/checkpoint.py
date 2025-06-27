import torch
import os, sys
import numpy as np
sys.path.insert(0, "/home/user/sdragon/sam2")

ckpt_path1 = '/DATA3/erictsai/v2vdet/v2v_training_result/from_H100/SigLIP2_series/OO_v11s_SAVPE_SigLIP2_FT_multi_layer_135/weights/best.pt'
ckpt_path2 = '/home/user/sdragon/sam2/checkpoints/oom11_s_num_maskmem_1_memory_position_0_no_mask_downsampler_downsampler_attentionlayer_1.pt'
ckpt1 = torch.load(ckpt_path1, map_location='cpu', weights_only=False)  # 建議先用 CPU 載入
ckpt2 = torch.load(ckpt_path2, map_location='cpu', weights_only=False)  # 建議先用 CPU 載入

def print_pt(ckpt):
    print(type(ckpt))             # 看看是 dict / list / nn.Module
    if isinstance(ckpt, dict):
        print("Top-level keys:", ckpt.keys())  # 顯示頂層 keys

        # 若含 state_dict，可檢查有哪些模型層
        if 'model' in ckpt:
            print("model type:", type(ckpt['model']))

        if 'state_dict' in ckpt:
            print("State dict keys:")
            for k in ckpt['state_dict'].keys():
                print(k)

print_pt(ckpt1)
print("=========================================")
print_pt(ckpt2)

from v2vdet.v2vdet_ultralytics.models.v2vdet.model import V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented
scale = 's'
MODEL1_YAML = f"/DATA3/erictsai/v2vdet/v2vdet_ultralytics/cfg/models/v2v/11/yolo11{scale}-v2v-multiscale_1_3_5.yaml"
model1 = V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented(MODEL1_YAML)
model1.info()
model1._load(ckpt_path1, task='detect')
# print("model1:", model1)
# print("model1.model.forward:", model1.model.forward)
# model1.predict(np.zeros((640, 640, 3), dtype=np.uint8))
print("model1.predictor:", model1.predictor)


from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import compose
from sam2.build_sam import _load_checkpoint
# config_file = "/home/user/sdragon/sam2/sam2/configs/abo/tt20_oom11_s_mode2.yaml"
config_file = "/home/user/sdragon/sam2/sam2/configs/abo/stream_tt20_oom11_s_mode2.yaml"
cfg = OmegaConf.load(config_file)
model_cfg = OmegaConf.to_container(cfg.trainer.model, resolve=True)
model2 = instantiate(model_cfg, _convert_="all")
# print("model2:", model2)
_load_checkpoint(model2, ckpt_path2)

print("model1.model:", model1.model)
model1.model = model2
print("model1.model:", model1.model)
print("model1.model.init_cond_frames_mode:", model1.model.init_cond_frames_mode)
model1.predict(np.zeros((640, 640, 3), dtype=np.uint8))