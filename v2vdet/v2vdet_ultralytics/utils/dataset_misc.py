import numpy as np
import cv2
from scipy.spatial import ConvexHull

def extract_and_resize_masked_region(image, normalized_points, target_size=(224, 224)):
    """
    從圖片中提取由點集合定義的區域，並調整到目標大小
    
    參數:
        image: numpy array，原始圖片
        points: shape為(1000, 2)的numpy array，表示mask的點集合
        target_size: 目標背景的大小，默認為(224, 224)
        
    返回:
        放置在黑色背景上的調整大小後的遮罩區域
    """
    
    height, width = image.shape[:2]
    
    # 將歸一化點轉換為圖片座標
    points = np.zeros_like(normalized_points)
    points[:, 0] = normalized_points[:, 0] * width   # x座標
    points[:, 1] = normalized_points[:, 1] * height  # y座標
    
    # 確保轉換後的點是整數（用於繪製mask）
    points_int = points.astype(np.int32)
    
    # 計算凸包以創建mask
    hull = ConvexHull(points)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 繪製凸包作為mask
    hull_points = points_int[hull.vertices]
    cv2.fillPoly(mask, [hull_points.astype(np.int32)], 255)
    
    # 或者，如果點已經是有序的輪廓點，可以直接使用
    # cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    
    # 應用mask到原始圖片
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # 獲取mask的邊界框
    x, y, w, h = cv2.boundingRect(mask)
    
    # 提取包含物體的ROI
    roi = masked_image[y:y+h, x:x+w]
    
    # 計算縮放因子，確保不超過目標大小
    scale = min(target_size[0] / w, target_size[1] / h)
    
    # 如果ROI太小，進行放大但不超過目標大小
    if scale > 1:
        new_w = min(int(w * scale), target_size[0])
        new_h = min(int(h * scale), target_size[1])
        roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        # 如果ROI已經太大，則縮小
        new_w = int(w * scale)
        new_h = int(h * scale)
        roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 創建黑色背景
    background = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # 計算放置位置（居中）
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    
    # 將調整大小後的ROI放到黑色背景上
    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi_resized
    
    return background