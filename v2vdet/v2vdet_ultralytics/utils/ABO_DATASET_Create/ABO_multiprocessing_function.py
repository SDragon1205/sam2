import os
import multiprocessing
from tqdm import tqdm
from functools import partial
from create_ABO_dataset import create_ABO_dataset

def process_single_sample(i, ABO, IMG_PATH, LABEL_PATH):
    """處理單個樣本的函數"""
    try:
        # 獲取一個樣本
        gay = ABO.get_one_sample()
        
        # 處理輸入圖像
        input_img = f"{i:07d}_I_"
        ABO.save_img(gay['query_img'], f"{IMG_PATH}/{input_img}.jpg")
        ABO.save_yolo_annotations(LABEL_PATH, f"{input_img}.txt", gay['query_img_bboxes'])
        
        # 處理模板圖像
        template_img = f"{i:07d}_T_"
        ABO.save_img(gay['template_img'], f"{IMG_PATH}/{template_img}.jpg")
        ABO.save_yolo_annotations(LABEL_PATH, f"{template_img}.txt", gay['template_img_bboxes'])
        
        return True, i
    except Exception as e:
        # 處理異常
        return False, f"處理樣本 {i} 時發生錯誤: {e}"

def process_samples_parallel(ABO, IMG_PATH, LABEL_PATH, num_samples=10, num_processes=None):
    """
    並行處理多個樣本
    
    參數:
    ABO: ABO 類的實例
    IMG_PATH: 圖像保存路徑
    LABEL_PATH: 標註保存路徑
    num_samples: 要處理的樣本數量
    num_processes: 並行進程數，如果為 None，則使用 CPU 核心數
    """
    # 確保輸出目錄存在
    os.makedirs(IMG_PATH, exist_ok=True)
    os.makedirs(LABEL_PATH, exist_ok=True)
    
    # 如果未指定進程數，則使用 CPU 核心數
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # 限制進程數不超過樣本數量和 CPU 核心數的較小值
    num_processes = min(num_processes, num_samples, multiprocessing.cpu_count())
    
    print(f"使用 {num_processes} 個進程並行處理 {num_samples} 個樣本...")
    
    # 創建進程池
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 準備部分函數，固定除 i 外的參數
        process_func = partial(process_single_sample, ABO=ABO, IMG_PATH=IMG_PATH, LABEL_PATH=LABEL_PATH)
        
        # 使用 imap 來處理樣本，並用 tqdm 顯示進度
        results = list(tqdm(pool.imap(process_func, range(num_samples)), total=num_samples))
    
    # 檢查結果
    successful = sum(1 for success, _ in results if success)
    print(f"成功處理 {successful}/{num_samples} 個樣本")
    
    # 如果有錯誤，顯示錯誤信息
    errors = [msg for success, msg in results if not success]
    if errors:
        print(f"發生了 {len(errors)} 個錯誤:")
        for error in errors[:5]:  # 只顯示前5個錯誤
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  - ... 以及 {len(errors) - 5} 個其他錯誤")


if __name__ == "__main__":
    # ABO 類的實例
    ABO = create_ABO_dataset()  # 請替換為您的實際類
    object_name = ABO.get_random_classes(total_class=160)
    ABO.dump_ultralytics_style_dataset_cfg()
    
    IMG_PATH = "DATASET/ABO/v2v_dataset/images/train"
    LABEL_PATH = "DATASET/ABO/v2v_dataset/labels/train"
    # 並行處理樣本
    process_samples_parallel(ABO, IMG_PATH, LABEL_PATH, num_samples=100000, num_processes=32)