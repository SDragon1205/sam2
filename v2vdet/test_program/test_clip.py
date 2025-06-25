import enum
import torch
from PIL import Image
from tqdm import tqdm
import time
import sys

import transformers

def count_parameters(model):
    # 計算所有需要梯度的參數總量
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  preprocess = transformers.CLIPImageProcessor().from_pretrained("openai/clip-vit-base-patch32")
  model = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
  model = model.to("cuda")
  
  parameter_count = count_parameters(model)
  print(f"CLIP Total number of parameters: {parameter_count}")
  
  # for i in range (10):
  # img_pil = [Image.open("test_program/f16.jpg")]*38496
    
  total_num = 38400
  batch_size= 512
  size = total_num//batch_size
  gg = time.time()
  crop_img_tensor = torch.randn(total_num, 3, 224, 224)
  s1 = time.time()
  # clip_input = [transformers.BatchFeature(
  #           {'pixel_values': crop_img_tensor}, tensor_type='pt') for _ in range(total_num//batch_size)]
  # clip_input = preprocess.preprocess(img_pil, return_tensors="pt")
  # clip_input = [cl.to("cuda") for cl in clip_input]

  dataset = torch.utils.data.TensorDataset(crop_img_tensor)

  dataloader = torch.utils.data.DataLoader(
          dataset,
          batch_size=batch_size,
          pin_memory=True,  # 固定記憶體，加速CPU到GPU的傳輸
          num_workers=0     # 多執行緒載入資料
      )
  print(f"Time: {time.time()-gg}")
  
  start = time.time()
  bf16_features = []
  with torch.autocast(device_type=device, dtype=torch.bfloat16):
    with torch.inference_mode():
      for batch in tqdm(dataloader, desc="With AMP"):
        batch[0] = batch[0].to('cuda')
        # breakpoint()
        output = model(batch[0], return_dict=True)
        bf16_features.append(output['last_hidden_state'])

  print(f"Time for bf16 method: {time.time()-start}")
  # fp16_features = [gg.to("cpu") for gg in fp16_features]
  bf16_features = torch.stack([gg.to("cpu") for gg in bf16_features], dim=0)
  torch.cuda.empty_cache()
  total_time = 0
  
  start = time.time()
  fp16_features = []
  
  all_cls_tokens = []
  all_patches = []
  with torch.autocast(device_type=device, dtype=torch.bfloat16):
    # with torch.inference_mode():
      for idx, batch in enumerate(tqdm(dataloader, desc="With AMP")):
        batch[0] = batch[0].to('cuda')
        # breakpoint()
        
        with torch.inference_mode():
          vision_model_output = model(batch[0], output_hidden_states=True, return_dict=True)
        
        cls_token = (vision_model_output['pooler_output'])
        # hidden_states = [tensor for tensor in vision_model_output['hidden_states']]
        # hidden_states = [vision_model_output['hidden_states'][:-5]]
        
        all_cls_tokens.append(cls_token.clone())
        # all_patches.append(hidden_states)
        
        # if (idx == len(dataloader)//2) or (idx == len(dataloader)-1):
        #   all_cls_tokens = [cls.to("cpu") for cls in all_cls_tokens]
        #   all_patches = [[tensor.to("cpu") for tensor in hidden_states] for hidden_states in all_patches]

  
  all_cls_tokens = torch.stack([cls.to("cpu") for cls in all_cls_tokens], dim=0)

  print(f"Time for fp16 method: {time.time()-start}")
  # fp16_features = [gg.to("cpu") for gg in fp16_features]
  # fp16_features = torch.stack([gg.to("cpu") for gg in fp16_features], dim=0)
  torch.cuda.empty_cache()
  total_time = 0
  sys.exit()

  fp32_features = []
  start=time.time()
  with torch.inference_mode():
    for batch in tqdm(dataloader, desc="Without AMP"):
      batch[0] = batch[0].to('cuda')
      # breakpoint()
      output = model(batch[0], return_dict=True)
      fp32_features.append(output['last_hidden_state'])
  
  print(f"Time for fp32 method: {time.time()-start}")
  fp32_features = torch.stack([gg.to("cpu") for gg in fp32_features], dim=0)
  torch.cuda.empty_cache()
  
  # similarity = torch.nn.functional.cosine_similarity(
  #       fp32_features, fp16_features).mean()
  
  # relative_error = torch.abs(fp32_features - fp16_features) / torch.abs(fp32_features)
  # mean_relative_error = relative_error.mean()


  sys.exit()
  outputs_my_shit = []
  # with torch.autocast(device_type=device, dtype=torch.float16):
    # clip_input[:] = [c.to("cuda") for c in clip_input]
  with torch.inference_mode():
    for i in tqdm(range (38496//batch_size)):
      # s1 = time.time()
      mid_idx = (batch_size-1)//2
      # if i == (mid_idx):
      #   clip_input[:mid_idx]=[c.to("cpu") for c in clip_input[:mid_idx]]
      #   clip_input[mid_idx:]=[c.to("cuda") for c in clip_input[mid_idx:]]
      
      output = model(**clip_input[i].to("cuda"), return_dict=True)
      # clip_input[i] = clip_input[i].to("cpu")
      # s2 = time.time()
      # print(f"Time taken for loop {i}: {time.time()-s1}")
      # total_time += (time.time()-s1)

      outputs_my_shit.append(output['last_hidden_state'])
  
  print(f"Time for my shit method: {time.time()-start}")
  
  print(f"Batch = {batch_size}, Total time taken for all: {total_time}")
  breakpoint()