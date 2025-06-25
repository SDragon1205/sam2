from transformers import AutoProcessor, SiglipVisionModel
if __name__ == "__main__":

  processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
  vision_encoder = SiglipVisionModel.from_pretrained(
    pretrained_model_name_or_path="google/siglip2-base-patch16-224") 
  param=sum(p.numel() for p in vision_encoder.parameters())
  print(f"SigLIP2 Vision Encoder Parameters: {param}")
  