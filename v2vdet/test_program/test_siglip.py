from PIL import Image
import requests
from transformers import AutoProcessor, SiglipVisionModel
from v2vdet.v2vdet_ultralytics.models.transformers.siglip import SiglipVisionModelWithProjection

def count_parameters(model):
    # 計算所有需要梯度的參數總量
    return sum(p.numel() for p in model.parameters())

model = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
print(last_hidden_state.shape)  # (batch_size, sequence_length, hidden_size)
# pooled_output = outputs.pooler_output  # pooled features
parameter_count = count_parameters(model)
print(f"SigLIP2 Total number of parameters: {parameter_count}")