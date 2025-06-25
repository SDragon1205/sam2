import torch
from transformers import SiglipVisionModel, AutoProcessor
from transformers.image_utils import load_image

# load the model and processor
ckpt = "google/siglip2-base-patch16-224"
model = SiglipVisionModel.from_pretrained(
        pretrained_model_name_or_path="google/siglip2-large-patch16-256").eval()
processor = AutoProcessor.from_pretrained(ckpt)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 使用範例
total_params = count_parameters(model)
print(f"模型參數量：{total_params:,}")