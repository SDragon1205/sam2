import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import cm
from sklearn.decomposition import PCA
import torch.nn.functional as F
import os

def generate_feature_heatmap(feature_tensor, image_tensor, output_path='heatmap_result.jpg', 
                             batch_idx=0, alpha=0.5, use_pca=True, grid_size=None):
    """
    將模型特徵轉為熱力圖並疊加到原始圖像上
    
    參數:
    - feature_tensor: 特徵張量，形狀為 [batch_size, n_classes, feature_dim]
    - image_tensor: 原始圖像張量，形狀為 [batch_size, channels, height, width]
    - output_path: 輸出圖像的路徑
    - batch_idx: 要處理的批次索引
    - alpha: 熱力圖疊加的透明度 (0~1)
    - use_pca: 是否使用PCA進行降維
    - grid_size: 自定義網格大小，若為None則自動計算
    
    返回:
    - overlay: 疊加了熱力圖的原始圖像 (numpy陣列)
    """
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 1. 預處理資料
    # 選擇指定批次
    feature_sample = feature_tensor[batch_idx]  # 形狀 [n_classes, feature_dim]
    image_sample = image_tensor[batch_idx]      # 形狀 [channels, height, width]
    
    # 將圖像轉換為numpy格式以便顯示
    if image_sample.dim() == 3:  # [C, H, W]
        # 將圖像從PyTorch格式 [C, H, W] 轉換為 [H, W, C]
        image_np = image_sample.permute(1, 2, 0).cpu().numpy()
        
        # 如果是標準化的圖像，轉換回0-255範圍
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # 如果是單通道圖像，轉為三通道
        if image_sample.shape[0] == 1:  # 灰度圖
            image_np = np.repeat(image_np, 3, axis=2)
            
        # 確保圖像是RGB格式
        if image_np.shape[2] > 3:  # 如果有Alpha通道等
            image_np = image_np[:, :, :3]
    else:
        raise ValueError("圖像張量必須是3維 [C, H, W]")
    
    # 獲取圖像尺寸
    h, w = image_np.shape[:2]
    n_classes, feature_dim = feature_sample.shape
    
    # 2. 降維處理特徵
    feature_np = feature_sample.detach().cpu().numpy()
    
    if use_pca:
        # 使用PCA降維
        pca = PCA(n_components=2)
        feature_2d = pca.fit_transform(feature_np)  # [n_classes, 2]
        
        # 計算每個類別的顯著性
        significance = np.sqrt(np.sum(feature_2d**2, axis=1))  # [n_classes]
    else:
        # 不使用PCA，直接取特徵向量的均值或範數
        significance = np.linalg.norm(feature_np, axis=1)  # [n_classes]
    
    # 3. 準備網格
    # 如果沒有指定網格大小，自動計算
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(n_classes)))
        
    # 創建網格並填充
    heatmap_grid = np.zeros((grid_size, grid_size))
    for i in range(min(n_classes, grid_size * grid_size)):
        row = i // grid_size
        col = i % grid_size
        heatmap_grid[row, col] = significance[i]
    
    # 4. 調整熱力圖尺寸以匹配原始圖像
    heatmap_tensor = torch.from_numpy(heatmap_grid).float().unsqueeze(0).unsqueeze(0)
    resized_heatmap = F.interpolate(
        heatmap_tensor, 
        size=(h, w), 
        mode='bilinear', 
        align_corners=False
    ).squeeze().cpu().numpy()
    
    # 5. 標準化熱力圖
    if resized_heatmap.max() != resized_heatmap.min():
        normalized_heatmap = (resized_heatmap - resized_heatmap.min()) / (resized_heatmap.max() - resized_heatmap.min())
    else:
        normalized_heatmap = np.zeros_like(resized_heatmap)
    
    # 6. 應用顏色映射
    colored_heatmap = cm.jet(normalized_heatmap)[:, :, :3]
    
    # 7. 疊加熱力圖到原始圖像
    overlay = (1-alpha) * image_np + alpha * (colored_heatmap * 255).astype(np.uint8)
    overlay = overlay.astype(np.uint8)
    
    # 8. 視覺化和保存結果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image_np)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Feature heat map')
    plt.imshow(colored_heatmap)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Overlay Image')
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.jpg', '_comparison.png'), dpi=300)
    
    # 僅保存疊加結果
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # 返回疊加結果
    return overlay

# 使用範例
if __name__ == "__main__":
    # 模擬數據
    feature_tensor = torch.randn(8, 199, 512)  # [batch_size, n_classes, feature_dim]
    image_tensor = torch.rand(8, 3, 224, 224)  # [batch_size, channels, height, width]
    
    # 生成並保存熱力圖
    result = generate_feature_heatmap(
        feature_tensor, 
        image_tensor, 
        output_path='output/feature_heatmap.jpg',
        batch_idx=0,
        alpha=0.6,
        use_pca=True
    )
    
    print(f"熱力圖已生成並保存到 'output/feature_heatmap.jpg'")