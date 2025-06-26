import os
import lmdb
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import shutil

def process_image(args):
    """
    處理單張圖片並返回序列化後的資料
    
    參數:
    - args: (rel_path, abs_path) 元組
    
    返回:
    - (key, value) 元組，用於LMDB存儲
    """
    rel_path, abs_path = args
    try:
        # 使用OpenCV讀取圖片
        img = cv2.imread(abs_path)
        if img is None:
            return None, f"無法讀取圖片: {abs_path}"
        
        # 使用相對路徑作為key
        key = rel_path.encode('utf-8')
        
        # 將圖片數據序列化
        value = pickle.dumps(img)
        
        return (key, value), None
    except Exception as e:
        return None, f"處理圖片時發生錯誤 {abs_path}: {str(e)}"

def create_image_lmdb(image_dir, lmdb_path, max_db_size=1024*1024*1024*1024*10, num_workers=None):  # 預設10TB
    """
    使用平行處理將指定資料夾下所有圖片存入LMDB
    
    參數:
    - image_dir: 圖片資料夾路徑
    - lmdb_path: 要創建的LMDB資料庫路徑
    - max_db_size: LMDB最大容量(bytes)，預設10TB
    - num_workers: 平行處理的工作進程數，預設為CPU核心數-1
    
    返回:
    - 創建的LMDB檔案路徑
    """
    # 設定工作進程數量
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)  # 預留一個核心給系統
    
    print(f"使用 {num_workers} 個工作進程進行平行處理")
    
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)  # 如果LMDB已存在，則刪除
        
    # 確保lmdb_path的目錄存在
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    
    # 建立LMDB環境
    env = lmdb.open(lmdb_path, map_size=max_db_size, writemap=True)
    
    # 收集所有圖片檔案路徑
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                abs_path = os.path.join(root, file)
                # 計算相對路徑作為key
                rel_path = os.path.relpath(abs_path, image_dir)
                image_paths.append((rel_path, abs_path))
    
    total_images = len(image_paths)
    print(f"找到 {total_images} 張圖片")
    
    # 使用進度條追蹤總體進度
    pbar = tqdm(total=total_images, desc="處理圖片")
    
    # 批量處理，每批處理後寫入LMDB
    batch_size = min(1000, max(100, total_images // 20))  # 動態調整批次大小
    
    start_time = time.time()
    processed_count = 0
    error_count = 0
    
    # 分批處理所有圖片
    for i in range(0, total_images, batch_size):
        batch = image_paths[i:i+batch_size]
        batch_results = []
        
        # 使用ProcessPoolExecutor平行處理一批圖片
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_image, args): args for args in batch}
            
            for future in as_completed(futures):
                result, error = future.result()
                if error:
                    error_count += 1
                    print(error)
                else:
                    batch_results.append(result)
                
                # 更新進度條
                processed_count += 1
                pbar.update(1)
        
        # 將這批處理結果寫入LMDB
        with env.begin(write=True) as txn:
            for key, value in batch_results:
                txn.put(key, value)
        
        # 計算和顯示處理速度
        elapsed_time = time.time() - start_time
        images_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"已處理 {processed_count}/{total_images} 張圖片，"
              f"平均速度: {images_per_second:.2f} 張/秒，"
              f"錯誤: {error_count} 張")
    
    pbar.close()
    
    # 關閉LMDB環境
    env.close()
    
    # 最終處理統計
    total_time = time.time() - start_time
    final_speed = total_images / total_time if total_time > 0 else 0
    
    print(f"LMDB創建完成: {lmdb_path}")
    print(f"總處理時間: {total_time:.2f} 秒")
    print(f"平均處理速度: {final_speed:.2f} 張/秒")
    print(f"成功處理: {processed_count - error_count} 張")
    print(f"處理失敗: {error_count} 張")
    
    return lmdb_path

def read_from_lmdb(lmdb_path, rel_path):
    """
    從LMDB中讀取指定相對路徑的圖片
    
    參數:
    - lmdb_path: LMDB資料庫路徑
    - rel_path: 要讀取的圖片相對路徑
    
    返回:
    - 圖片的numpy數組
    """
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        # 將相對路徑轉換為bytes作為key
        key = rel_path.encode('utf-8')
        value = txn.get(key)
        
        if value is None:
            env.close()
            raise KeyError(f"找不到key: {rel_path}")
        
        # 反序列化為numpy數組
        img = pickle.loads(value)
    
    env.close()
    return img

def list_lmdb_keys(lmdb_path, max_keys=100):
    """
    列出LMDB中的所有鍵（最多顯示max_keys個）
    
    參數:
    - lmdb_path: LMDB資料庫路徑
    - max_keys: 最多顯示的鍵數量
    
    返回:
    - 鍵列表
    """
    env = lmdb.open(lmdb_path, readonly=True)
    keys = []
    
    with env.begin() as txn:
        cursor = txn.cursor()
        count = 0
        
        for key, _ in cursor:
            keys.append(key.decode('utf-8'))
            count += 1
            if count >= max_keys:
                break
    
    env.close()
    
    if count == max_keys:
        print(f"只顯示前 {max_keys} 個鍵，LMDB中可能有更多...")
    
    return keys


# 例子: 創建和讀取LMDB
if __name__ == "__main__":
    # 設定資料夾路徑和LMDB檔案路徑
    image_dir = "DATASET/Object365/images/train"  # 修改為您的圖片資料夾路徑
    lmdb_path = "DATASET/Object365/images/train.lmdb"  # 修改為您想存儲LMDB的路徑
    
    # 創建LMDB
    lmdb_file = create_image_lmdb(image_dir, lmdb_path)
    
    print("DONE.")
    
    # 測試讀取 - 請替換為實際的相對路徑
    # test_rel_path = "train2017/000000064619.jpg"  # 修改為您要測試的圖片相對路徑
    # try:
    #     img = read_from_lmdb(lmdb_path, test_rel_path)
    #     print(f"成功讀取圖片: {test_rel_path}, 形狀: {img.shape}")
        
    #     # # 可選: 顯示圖片
    #     # cv2.imshow("Test Image", img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    # except KeyError as e:
    #     print(f"錯誤: {str(e)}")