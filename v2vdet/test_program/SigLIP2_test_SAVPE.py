from transformers import AutoImageProcessor, Dinov2Model, AutoProcessor, SiglipVisionModel
import torch
from datasets import load_dataset
from PIL import Image
from v2vdet.v2vdet_ultralytics.nn.modules import MultiLevelTemplateAttentionPooling
from v2vdet.v2vdet_ultralytics.nn.modules.block import SAVPE, PATCH_EMBEDDING_SAVPE
from ultralytics import YOLO
import pickle
from ultralytics.utils.plotting import feature_visualization

# dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
# image = dataset["test"]["image"][0]
BATCH_SIZE=4

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
image = Image.open('image/cat_scenario.jpg')
image = [image for _ in range(BATCH_SIZE*80)]

# image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
# model = Dinov2Model.from_pretrained("facebook/dinov2-base")
image_processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
model = SiglipVisionModel.from_pretrained(
    pretrained_model_name_or_path="google/siglip2-base-patch16-224") 

# image_processor.to(device='cuda')


inputs = image_processor(image, return_tensors="pt")

model = model.to(device='cuda')
inputs = inputs.to(device='cuda')
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)

hidden_states_list = [outputs.hidden_states[i] for i in [-2, -4, -6]]

del outputs

# mm = MultiLevelTemplateAttentionPooling(num_patches=257, num_levels=len(hidden_states_list))
# result = mm(hidden_states_list)


# yolo_ckpt = 'ckpt/yolo11s.pt'
# yolo_ckpt = 'ckpt/yoloe-v8s-seg.pt'
# yolo_model = YOLO(yolo_ckpt, task='detect')
# yolo_model.predict(source='image/cat_scenario.jpg', show=True, conf=0.25, iou=0.45, agnostic_nms=True, device='cpu', embed=[])

# with open('yoloe_11_feature_paymarid_output.pkl', 'rb') as f:
#     feature_paymarid_output = pickle.load(f)

# feature_visualization(feature_paymarid_output, module_type='0.SAVPE', stage=0, save_dir='image/feature_visualization')

ch = [256, 512, 512]
c3 = 256
embed = 512
bs = 80
image_size = 80
x = [torch.zeros(bs, 256 if i==0 else 512, image_size//(2**i), image_size//(2**i)) for i in range(3)]
# x = torch.zeros()
vp = torch.zeros(BATCH_SIZE*80, 1, image_size, image_size)

hidden_states_tensor = torch.stack(tensors=hidden_states_list, dim=1)
patch_embedding_layer = PATCH_EMBEDDING_SAVPE().to('cuda')
vp = vp.to('cuda')
patch_embedding_layer(hidden_states_list, vp)
params_count = count_parameters(patch_embedding_layer)
print(f"PATCH_EMBEDDING_SAVPE Total number of parameters: {params_count}")

# savpe_layer = SAVPE(ch = ch, c3 = c3, embed=embed)
# params_count = count_parameters(savpe_layer)
# print(f"SAVPE Total number of parameters: {params_count}")
# result = savpe_layer(x, vp)
# breakpoint()