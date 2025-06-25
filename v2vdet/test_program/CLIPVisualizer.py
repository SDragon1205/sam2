import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from transformers import CLIPProcessor, CLIPModel

class CLIPVisualizer:
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        """
        初始化 CLIP 視覺化器
        Args:
            model_name: CLIP 模型名稱
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # CLIP 的標準配置
        self.image_size = 224
        self.patch_size = 16
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
    def get_attention_maps(self, image_path):
        """
        獲取圖像的 attention maps
        Args:
            image_path: 圖像路徑
        Returns:
            原始圖像和 attention maps
        """
        # 載入並處理圖像
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 獲取 attention weights
        with torch.no_grad():
            outputs = self.model.vision_model(**inputs)
            attentions = outputs.attentions  # tuple of attention weights
        
        # 將圖像調整為標準大小
        image = image.resize((self.image_size, self.image_size))
        image_array = np.array(image)
        
        return image_array, attentions
    
    def visualize_patches(self, image_array, attentions, layer_idx=-1, head_idx=0):
        """
        視覺化 patch attention
        Args:
            image_array: 原始圖像陣列
            attentions: attention weights
            layer_idx: 要視覺化的層索引
            head_idx: 要視覺化的注意力頭索引
        """
        # 獲取指定層和頭的 attention weights
        attn = attentions[layer_idx][0, head_idx].cpu().numpy()
        
        # 創建 attention map
        attention_map = np.zeros((self.image_size, self.image_size))
        
        # 為每個 patch 填充 attention 值
        for i in range(self.image_size // self.patch_size):
            for j in range(self.image_size // self.patch_size):
                patch_idx = i * (self.image_size // self.patch_size) + j
                attention_score = attn[patch_idx].mean()
                
                # 填充對應的 patch 區域
                y_start = i * self.patch_size
                y_end = (i + 1) * self.patch_size
                x_start = j * self.patch_size
                x_end = (j + 1) * self.patch_size
                
                attention_map[y_start:y_end, x_start:x_end] = attention_score
        
        # 正規化 attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # 創建熱力圖
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 疊加原始圖像和熱力圖
        alpha = 0.5
        overlayed = cv2.addWeighted(image_array, alpha, heatmap, 1-alpha, 0)
        
        # 顯示結果
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_array)
        plt.title('原始圖像')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title('Attention Heatmap')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlayed)
        plt.title('疊加結果')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def visualize_all_heads(self, image_path, layer_idx=-1):
        """
        視覺化所有注意力頭的 attention
        Args:
            image_path: 圖像路徑
            layer_idx: 要視覺化的層索引
        """
        image_array, attentions = self.get_attention_maps(image_path)
        num_heads = attentions[layer_idx].shape[1]
        
        plt.figure(figsize=(20, 5 * ((num_heads + 3) // 4)))
        for head_idx in range(num_heads):
            plt.subplot(((num_heads + 3) // 4), 4, head_idx + 1)
            self.visualize_patches(image_array, attentions, layer_idx, head_idx)
            plt.title(f'Head {head_idx}')
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    visualizer = CLIPVisualizer()
    
    # 替換為您的圖像路徑
    image_path = "test_program/f16.jpg"
    
    # 視覺化最後一層的所有注意力頭
    visualizer.visualize_all_heads(image_path)
    
    # 視覺化特定層和特定頭的 attention
    image_array, attentions = visualizer.get_attention_maps(image_path)
    visualizer.visualize_patches(image_array, attentions, layer_idx=-1, head_idx=0)