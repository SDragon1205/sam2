import torch
from PIL import Image
import v2vdet.v2vdet_ultralytics.perception_models.core.vision_encoder.pe as pe
import v2vdet.v2vdet_ultralytics.perception_models.core.vision_encoder.transforms as transforms
from v2vdet.v2vdet_ultralytics.nn.modules.block import PATCH_EMBEDDING_SAVPE

print("VisionTransformer configs:", pe.VisionTransformer.available_configs())
# CLIP configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224']

'''
PE-Core-B16-224: width = 768, params = 90M
PE-Core-L14-336: width = 1024, params = 320M
PE-Core-G14-448" width = 1536, params = 1.88B
'''

model = pe.VisionTransformer.from_config("PE-Core-B16-224", pretrained=True)  # Downloads from HF

width = 768
model = model.cuda()

preprocess = transforms.get_image_transform(model.image_size)
# tokenizer = transforms.get_text_tokenizer(model.context_length)

image = preprocess(Image.open("image/cat_scenario.jpg")).unsqueeze(0).cuda()
# text = tokenizer(["a diagram", "a dog", "a cat"]).cuda()

# from transformers import AutoImageProcessor
# image_processor = AutoImageProcessor.from_pretrained(
#         pretrained_model_name_or_path="google/siglip2-base-patch32-256")

# chi = image_processor(Image.open("image/cat_scenario.jpg"), return_tensors="pt")

with torch.no_grad(), torch.autocast("cuda"):
    # image_features, text_features, logit_scale = model(image, text)
    # text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
    encode_image = model(image, output_hidden_list=[-1, -2, -3], strip_cls_token=True)
pass
# print("Label probs:", text_probs)  # prints: [[0.0, 0.0, 1.0]]

# patch_embedding_list = [encode_image['hidden_states'][i] for i in [-1, -3, -5]]
patch_embedding_list = encode_image['hidden_states']
patch_embedding = torch.stack(patch_embedding_list, dim=1)

patch_embedding_savpe_layer = PATCH_EMBEDDING_SAVPE(embed_dim=width)
vp = torch.zeros(patch_embedding.shape[0], 1, 80, 80).to(device='cuda')
patch_embedding_savpe_layer = patch_embedding_savpe_layer.to(device='cuda')
patch_embedding_result = patch_embedding_savpe_layer(patch_embedding, vp)
pass