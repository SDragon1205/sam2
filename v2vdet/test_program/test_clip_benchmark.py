import enum
import torch
from PIL import Image
from tqdm import tqdm
import time
import sys
from torchinfo import summary

import transformers

if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  preprocess = transformers.CLIPImageProcessor().from_pretrained("openai/clip-vit-base-patch32")
  model = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
  txt_name = "clip-vit-base-patch32_model_summary.txt"

  model_summary = summary(model, input_size=(1, 3, 224, 224))
  print(model_summary)
  with open (txt_name, "w") as f:
    f.write(str(model_summary))
    f.write("\n")

  model = model.to("cuda")

  input = torch.randn(80*16, 3, 224, 224)
  input = input.to("cuda")
  start = time.time()
  bf16_features = []
  with torch.autocast(device_type=device, dtype=torch.bfloat16):
    with torch.inference_mode():
      for idx in tqdm(range(10), desc="With AMP bfloat16"):
        # breakpoint()
        output = model(input, return_dict=True)

  bf16_time = time.time()-start
  print(f"Time for bf16 method: {bf16_time:.2f}")
  # fp16_features = [gg.to("cpu") for gg in fp16_features]
  torch.cuda.empty_cache()
  total_time = 0

  start = time.time()
  with torch.autocast(device_type=device, dtype=torch.float16):
    with torch.inference_mode():
      for idx in tqdm(range(10), desc="With AMP float16"):
        # breakpoint()
        output = model(input, return_dict=True)

  fp16_time = time.time()-start
  print(f"Time for fp16 method: {fp16_time:.2f}")
  torch.cuda.empty_cache()
  total_time = 0

  start = time.time()
  with torch.autocast(device_type=device, dtype=torch.float32):
    with torch.inference_mode():
      for idx in tqdm(range(10), desc="FP32"):
        # breakpoint()
        output = model(input, return_dict=True)

  fp32_time = time.time()-start
  print(f"Time for fp32 method: {fp32_time:.2f}")
  torch.cuda.empty_cache()
  total_time = 0

  with open (txt_name, "a") as f:
    f.write(f"bf16_time: {bf16_time:.2f}s\n")
    f.write(f"fp16_time: {fp16_time:.2f}s\n")
    f.write(f"fp32_time: {fp32_time:.2f}s\n")
