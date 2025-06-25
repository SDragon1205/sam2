from transformers import AutoImageProcessor, Dinov2Model
import torch
from datasets import load_dataset
from PIL import Image
from v2vdet.v2vdet_ultralytics.nn.modules import MultiLevelTemplateAttentionPooling
from v2vdet.v2vdet_ultralytics.nn.modules.block import SAVPE
from ultralytics import YOLO

# dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
# image = dataset["test"]["image"][0]

image = Image.open('image/cat_scenario.jpg')

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = Dinov2Model.from_pretrained("facebook/dinov2-base")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)

hidden_states_list = [outputs.hidden_states[i] for i in [-2, -4, -6]]

mm = MultiLevelTemplateAttentionPooling(num_patches=257)
result = mm(hidden_states_list)

ch = [256, 512, 512]
c3 = 256
embed = 512
bs = 1
image_size = 224
x = [torch.zeros(bs, 256 if i==0 else 512, image_size//(2**i), image_size//(2**i)) for i in range(3)]
vp = torch.zeros(bs, 10, image_size, image_size)

savpe_layer = SAVPE(ch = ch, c3 = c3, embed=embed)
savpe_layer(x, vp)