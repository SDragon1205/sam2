import torch
from PIL import Image
import perception_models.core.vision_encoder.pe as pe
import perception_models.core.vision_encoder.transforms as transforms

print("CLIP configs:", pe.CLIP.available_configs())
# CLIP configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224']

model = pe.CLIP.from_config("PE-Spatial-G14-448", pretrained=True)  # Downloads from HF
model = model.cuda()

preprocess = transforms.get_image_transform(model.image_size)
tokenizer = transforms.get_text_tokenizer(model.context_length)

image = preprocess(Image.open("image/cat_scenario.jpg")).unsqueeze(0).cuda()
text = tokenizer(["a diagram", "a dog", "a cat"]).cuda()

with torch.no_grad(), torch.autocast("cuda"):
    # image_features, text_features, logit_scale = model(image, text)
    # text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
    encode_image = model.encode_image(image)
pass
# print("Label probs:", text_probs)  # prints: [[0.0, 0.0, 1.0]]