import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import sqrt
from PIL import Image
import io
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

def generate_attention_heatmap(encoder_output, layer_idx=0, batch_idx=0, method='norm'):
    """
    從ViT的輸出中生成注意力熱圖
    
    參數：
    - encoder_output: 形狀為[layers, batch, patches, dim]的張量
    - layer_idx: 要視覺化的層索引
    - batch_idx: 要視覺化的批次索引
    - method: 特徵聚合方法，可選'norm'(歐氏範數),'mean'(平均值),'max'(最大值)
    
    返回：
    - attention_map: 2D熱圖陣列, 值範圍在[0,1]之間
    """
    # 選擇特定層和批次的特徵
    features = encoder_output[layer_idx, batch_idx]  # [576, 1024]
    
    # 根據選擇的方法聚合特徵維度
    if method == 'norm':
        # 使用歐氏範數（特徵向量的長度）
        attention_scores = torch.norm(features, dim=1)
    elif method == 'mean':
        # 使用平均值
        attention_scores = torch.mean(features, dim=1)
    elif method == 'max':
        # 使用最大值
        attention_scores = torch.max(features, dim=1)[0]
    else:
        raise ValueError(f"不支援的方法: {method}")
    
    # 轉換為numpy陣列
    attention_scores = attention_scores.detach().cpu().numpy()
    
    # 計算網格大小（假設是正方形）
    grid_size = int(sqrt(features.shape[0]))
    
    # 將扁平的注意力分數重塑為2D網格
    attention_map = attention_scores.reshape(grid_size, grid_size)
    
    # 標準化注意力圖到[0,1]區間
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    return attention_map

def overlay_heatmap_on_image_pil(image, attention_map, colormap='jet', alpha=0.7):
    """
    將熱圖疊加在原始圖像上，返回PIL圖像
    
    參數：
    - image: 原始圖像，PIL圖像或numpy陣列
    - attention_map: 注意力熱圖，形狀為[grid_size, grid_size]的numpy陣列
    - colormap: 熱圖的色彩映射（'jet', 'viridis', 'inferno'等）
    - alpha: 熱圖的透明度，0為完全透明，1為完全不透明
    
    返回：
    - overlaid_image: 疊加了熱圖的PIL圖像
    """
    # 確保image是PIL圖像
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image * 255) if image.max() <= 1.0 else np.uint8(image))
    
    # 確保image是RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 將PIL圖像轉換為numpy陣列以便處理
    image_np = np.array(image)
    
    # 調整熱圖大小以匹配原始圖像
    attention_resized = cv2.resize(attention_map, (image.width, image.height), interpolation=cv2.INTER_LINEAR)
    
    # 創建熱圖色彩映射
    if colormap == 'jet':
        # 使用OpenCV的色彩映射
        attention_heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # OpenCV使用BGR而不是RGB，所以需要轉換
        attention_heatmap = cv2.cvtColor(attention_heatmap, cv2.COLOR_BGR2RGB)
    else:
        # 使用matplotlib的色彩映射
        cmap = plt.get_cmap(colormap)
        attention_colored = cmap(attention_resized)[:, :, :3]  # 取RGB通道
        attention_heatmap = (attention_colored * 255).astype(np.uint8)
    
    # 疊加熱圖和原始圖像
    overlaid_image_np = cv2.addWeighted(image_np, 1-alpha, attention_heatmap, alpha, 0)
    
    # 轉換回PIL圖像
    overlaid_image = Image.fromarray(overlaid_image_np)
    
    return overlaid_image

def fig2pil(fig):
    """
    將matplotlib圖形轉換為PIL圖像
    
    參數：
    - fig: matplotlib圖形對象
    
    返回：
    - pil_image: PIL圖像
    """
    # 使用io緩衝區
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0, dpi=300)
    buf.seek(0)
    
    # 從緩衝區創建PIL圖像
    pil_image = Image.open(buf)
    
    # 關閉matplotlib圖形，釋放記憶體
    plt.close(fig)
    
    return pil_image

def visualize_vit_attention_pil(encoder_output, original_images, layer_indices=None, show_original=True, 
                             colormap='jet', alpha=0.7, method='norm'):
    """
    視覺化ViT的多層注意力並與原始圖像對比，返回PIL圖像
    
    參數：
    - encoder_output: 形狀為[layers, batch, patches, dim]的張量
    - original_images: 原始輸入圖像，可以是PIL圖像、numpy陣列或PyTorch張量
    - layer_indices: 要視覺化的層索引列表（可選，默認所有層）
    - show_original: 是否顯示原始圖像作為對比
    - colormap: 熱圖的色彩映射
    - alpha: 熱圖的透明度
    - method: 特徵聚合方法
    
    返回：
    - pil_image: 包含視覺化結果的PIL圖像
    """
    n_layers = encoder_output.shape[0]
    if layer_indices is None:
        layer_indices = list(range(n_layers))
    else:
        n_layers = len(layer_indices)
    
    n_samples = min(4, encoder_output.shape[1])  # 最多顯示4個樣本
    
    # 確保original_images是列表或batch
    if not isinstance(original_images, (list, tuple, np.ndarray)) or len(np.array(original_images).shape) < 3:
        original_images = [original_images]
    
    # 處理PyTorch張量
    if isinstance(original_images, torch.Tensor):
        if len(original_images.shape) == 4 and original_images.shape[1] == 3:
            # [batch, C, H, W] -> [batch, H, W, C]
            original_images = original_images.permute(0, 2, 3, 1).detach().cpu().numpy()
        else:
            original_images = original_images.detach().cpu().numpy()
    
    # 創建圖形，如果顯示原始圖像則增加一列
    cols = n_samples
    if show_original:
        cols += 1
    
    fig, axes = plt.subplots(n_layers, cols, figsize=(3*cols, 3*n_layers), dpi=150)
    
    # 處理單行或單列的情況
    if n_layers == 1 and cols == 1:
        axes = np.array([[axes]])
    elif n_layers == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    for i, layer_idx in enumerate(layer_indices):
        col_offset = 0
        
        # 顯示原始圖像作為第一列
        if show_original:
            for batch_idx in range(n_samples):
                if batch_idx < len(original_images):
                    # 確保圖像是PIL或numpy
                    if isinstance(original_images[batch_idx], torch.Tensor):
                        img = original_images[batch_idx].detach().cpu().numpy()
                    else:
                        img = original_images[batch_idx]
                    
                    # 顯示原始圖像
                    if isinstance(img, Image.Image):
                        axes[i, 0].imshow(np.array(img))
                    else:
                        axes[i, 0].imshow(img)
                    
                    axes[i, 0].set_title(f"原始圖像")
                    axes[i, 0].axis('off')
            col_offset = 1
        
        # 為每個樣本生成並顯示熱圖
        for batch_idx in range(n_samples):
            if batch_idx < len(original_images):
                # 獲取原始圖像
                if isinstance(original_images[batch_idx], torch.Tensor):
                    img = original_images[batch_idx].detach().cpu().numpy()
                else:
                    img = original_images[batch_idx]
                
                if isinstance(img, Image.Image):
                    img_pil = img
                else:
                    # 轉換numpy陣列為PIL
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img)
                
                # 生成注意力熱圖
                attention_map = generate_attention_heatmap(encoder_output, layer_idx, batch_idx, method=method)
                
                # 在原始圖像上疊加熱圖
                overlaid_image = overlay_heatmap_on_image_pil(
                    img_pil, attention_map, colormap=colormap, alpha=alpha)
                
                # 顯示疊加的圖像
                axes[i, batch_idx + col_offset].imshow(np.array(overlaid_image))
                axes[i, batch_idx + col_offset].set_title(f"第{layer_idx+1}層，樣本{batch_idx+1}")
                axes[i, batch_idx + col_offset].axis('off')
    
    plt.tight_layout()
    
    # 轉換為PIL圖像
    pil_image = fig2pil(fig)
    
    return pil_image

def get_single_attention_map_pil(encoder_output, original_image, layer_idx=0, batch_idx=0, 
                              method='norm', colormap='jet', alpha=0.7):
    """
    生成單一圖像的注意力熱圖，返回PIL圖像
    
    參數：
    - encoder_output: 形狀為[layers, batch, patches, dim]的張量
    - original_image: 原始圖像，可以是PIL圖像、numpy陣列或PyTorch張量
    - layer_idx: 要視覺化的層索引
    - batch_idx: 要視覺化的批次索引
    - method: 特徵聚合方法
    - colormap: 熱圖的色彩映射
    - alpha: 熱圖的透明度
    
    返回：
    - pil_image: 疊加了熱圖的PIL圖像
    """
    # 處理原始圖像
    if isinstance(original_image, torch.Tensor):
        if len(original_image.shape) == 4:  # [batch, C, H, W]
            original_image = original_image[batch_idx].permute(1, 2, 0).detach().cpu().numpy()
        elif len(original_image.shape) == 3 and original_image.shape[0] == 3:  # [C, H, W]
            original_image = original_image.permute(1, 2, 0).detach().cpu().numpy()
        else:
            original_image = original_image.detach().cpu().numpy()
    
    # 轉換為PIL圖像
    if isinstance(original_image, np.ndarray):
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        original_image_pil = Image.fromarray(original_image)
    else:
        original_image_pil = original_image
    
    # 生成注意力熱圖
    attention_map = generate_attention_heatmap(encoder_output, layer_idx, batch_idx, method=method)
    
    # 在原始圖像上疊加熱圖
    overlaid_image = overlay_heatmap_on_image_pil(
        original_image_pil, attention_map, colormap=colormap, alpha=alpha)
    
    return overlaid_image

def visualize_all_methods_pil(encoder_output, original_image, layer_idx=0, batch_idx=0):
    """
    比較不同特徵聚合方法的注意力熱圖，返回PIL圖像
    
    參數：
    - encoder_output: 形狀為[layers, batch, patches, dim]的張量
    - original_image: 原始圖像，可以是PIL圖像、numpy陣列或PyTorch張量
    - layer_idx: 要視覺化的層索引
    - batch_idx: 要視覺化的批次索引
    
    返回：
    - pil_image: 包含不同方法比較的PIL圖像
    """
    methods = ['norm', 'mean', 'max']
    
    # 處理原始圖像
    if isinstance(original_image, torch.Tensor):
        if len(original_image.shape) == 4:  # [batch, C, H, W]
            original_image = original_image[batch_idx].permute(1, 2, 0).detach().cpu().numpy()
        elif len(original_image.shape) == 3 and original_image.shape[0] == 3:  # [C, H, W]
            original_image = original_image.permute(1, 2, 0).detach().cpu().numpy()
        else:
            original_image = original_image.detach().cpu().numpy()
    
    # 轉換為PIL圖像
    if isinstance(original_image, np.ndarray):
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        original_image_pil = Image.fromarray(original_image)
    else:
        original_image_pil = original_image
    
    # 創建圖形
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(4*(len(methods) + 1), 4), dpi=150)
    
    # 顯示原始圖像
    axes[0].imshow(np.array(original_image_pil))
    axes[0].set_title("原始圖像")
    axes[0].axis('off')
    
    # 為每種方法生成並顯示熱圖
    for i, method in enumerate(methods):
        # 生成注意力熱圖
        attention_map = generate_attention_heatmap(encoder_output, layer_idx, batch_idx, method=method)
        
        # 在原始圖像上疊加熱圖
        overlaid_image = overlay_heatmap_on_image_pil(
            original_image_pil, attention_map, colormap='jet', alpha=0.7)
        
        # 顯示疊加的圖像
        axes[i + 1].imshow(np.array(overlaid_image))
        axes[i + 1].set_title(f"方法: {method}")
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    # 轉換為PIL圖像
    pil_image = fig2pil(fig)
    
    return pil_image

def visualize_multi_layer_single_image_pil(encoder_output, original_image, batch_idx=0, 
                                         layer_indices=None, method='norm', colormap='jet', alpha=0.7):
    """
    視覺化單一圖像在多個層的注意力，返回PIL圖像
    
    參數：
    - encoder_output: 形狀為[layers, batch, patches, dim]的張量
    - original_image: 原始圖像，可以是PIL圖像、numpy陣列或PyTorch張量
    - batch_idx: 要視覺化的批次索引
    - layer_indices: 要視覺化的層索引列表（可選，默認所有層）
    - method: 特徵聚合方法
    - colormap: 熱圖的色彩映射
    - alpha: 熱圖的透明度
    
    返回：
    - pil_image: 包含多層注意力視覺化的PIL圖像
    """
    n_layers = encoder_output.shape[0]
    if layer_indices is None:
        layer_indices = list(range(n_layers))
    else:
        n_layers = len(layer_indices)
    
    # 處理原始圖像
    if isinstance(original_image, torch.Tensor):
        if len(original_image.shape) == 4:  # [batch, C, H, W]
            original_image = original_image[batch_idx].permute(1, 2, 0).detach().cpu().numpy()
        elif len(original_image.shape) == 3 and original_image.shape[0] == 3:  # [C, H, W]
            original_image = original_image.permute(1, 2, 0).detach().cpu().numpy()
        else:
            original_image = original_image.detach().cpu().numpy()
    
    # 轉換為PIL圖像
    if isinstance(original_image, np.ndarray):
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        original_image_pil = Image.fromarray(original_image)
    else:
        original_image_pil = original_image
    
    # 創建圖形
    fig, axes = plt.subplots(1, n_layers + 1, figsize=(4*(n_layers + 1), 4), dpi=150)
    
    # 顯示原始圖像
    axes[0].imshow(np.array(original_image_pil))
    axes[0].set_title("原始圖像")
    axes[0].axis('off')
    
    # 為每一層生成並顯示熱圖
    for i, layer_idx in enumerate(layer_indices):
        # 生成注意力熱圖
        attention_map = generate_attention_heatmap(encoder_output, layer_idx, batch_idx, method=method)
        
        # 在原始圖像上疊加熱圖
        overlaid_image = overlay_heatmap_on_image_pil(
            original_image_pil, attention_map, colormap=colormap, alpha=alpha)
        
        # 顯示疊加的圖像
        axes[i + 1].imshow(np.array(overlaid_image))
        axes[i + 1].set_title(f"層 {layer_idx+1}")
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    # 轉換為PIL圖像
    pil_image = fig2pil(fig)
    
    return pil_image