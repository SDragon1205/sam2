import torch
import os, sys
sys.path.insert(0, "/home/user/sdragon/sam2")

ckpt_path = '/DATA3/erictsai/v2vdet/v2v_training_result/from_H100/SigLIP2_series/OO_v11s_SAVPE_SigLIP2_FT_multi_layer_135/weights/best.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # 建議先用 CPU 載入

print(type(ckpt))             # 看看是 dict / list / nn.Module
if isinstance(ckpt, dict):
    print("Top-level keys:", ckpt.keys())  # 顯示頂層 keys

    # 若含 state_dict，可檢查有哪些模型層
    if 'model' in ckpt:
        print("model type:", type(ckpt['model']))

        model = ckpt['model']
        print("predictor type:", type(model.predictor))
        # 如果是 nn.Module 類型，可以印出模型架構
        if hasattr(model.predictor, 'forward'):
            print("predictor structure:\n", model.predictor)

        # 如果是 dict 或物件，可用 vars 或 __dict__ 印出屬性
        try:
            print("predictor attributes:\n", vars(model.predictor))
        except TypeError:
            print("predictor __dict__:\n", model.predictor.__dict__)
        
        for name, param in model.predictor.named_parameters():
            print(f"{name}: {tuple(param.shape)}")
    if 'state_dict' in ckpt:
        print("State dict keys:")
        for k in ckpt['state_dict'].keys():
            print(k)
