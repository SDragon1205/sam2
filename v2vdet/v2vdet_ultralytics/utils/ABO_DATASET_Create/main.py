import os
import random
import time
import multiprocessing as mp
from pathlib import Path
from PIL import Image, ImageFilter
from typing import Union
from tqdm import tqdm
import concurrent.futures
import shutil
import tempfile
from create_ABO_dataset import create_ABO_dataset

# 從現有的 create_ABO_dataset 類擴展
class accelerated_ABO_dataset(create_ABO_dataset):
    """添加加速方法的 ABO 數據集類"""

    def __init__(self, ABO_DATASET_PATH="DATASET/ABO/abo-spins/spins/original", cache_size=50):
        super().__init__(ABO_DATASET_PATH)
        # 添加快取屬性
        self.template_cache = {}  # 類別 ID -> 處理好的模板列表
        self.background_cache = None  # 快取背景圖像
        self.cache_size = cache_size
        
    def preload_backgrounds(self, folder_path="DATASET/Object365/images/train", count=50):
        """預加載背景圖像到記憶體"""
        print(f"預加載 {count} 張背景圖像...")
        files = os.listdir(folder_path)
        selected_files = random.sample(files, min(count, len(files)))
        
        self.background_cache = []
        for file in tqdm(selected_files, desc="加載背景"):
            try:
                img = Image.open(os.path.join(folder_path, file))
                self.background_cache.append(img)
            except Exception as e:
                print(f"加載背景 {file} 時發生錯誤: {e}")
        
        print(f"已加載 {len(self.background_cache)} 張背景圖像")
    
    def preload_templates(self, class_ids=None, templates_per_class=10):
        """預處理和快取模板圖像"""
        if class_ids is None:
            class_ids = list(range(self.nc))
        
        print(f"預處理 {len(class_ids)} 個類別的模板圖像...")
        
        for cls_id in tqdm(class_ids, desc="處理模板"):
            if cls_id not in self.template_cache:
                self.template_cache[cls_id] = []
                
            folder_path = self.object_folder_path[cls_id]
            template_files = os.listdir(folder_path)
            
            # 隨機選擇一些模板檔案
            selected_files = random.sample(template_files, min(templates_per_class, len(template_files)))
            
            for file in selected_files:
                try:
                    template_path = os.path.join(folder_path, file)
                    nowhite_template_img = self.remove_white_background(template_path)
                    resize_template_img = self.random_resize_smaller_than(nowhite_template_img, max_size=(150, 150))
                    self.template_cache[cls_id].append(resize_template_img)
                except Exception as e:
                    print(f"處理模板 {file} 時發生錯誤: {e}")
    
    def optimized_get_one_sample(self, object_amount_bound=10, output_size=(640, 640)):
        """優化版的 get_one_sample 方法，使用快取"""
        # 確保已加載背景
        if self.background_cache is None or len(self.background_cache) == 0:
            self.preload_backgrounds(count=self.cache_size)
        
        # 選擇隨機類別
        total_nc = random.randint(5, min(object_amount_bound, self.nc))
        random_cls = [random.randint(0, self.nc-1) for _ in range(total_nc)]
        random_cls = sorted(list(set(random_cls)))  # 移除重複並排序
        
        # 確保所有選擇的類別都有快取的模板
        for cls_id in random_cls:
            if cls_id not in self.template_cache or len(self.template_cache[cls_id]) < 2:
                # 如果沒有快取這個類別，快取它
                folder_path = self.object_folder_path[cls_id]
                template_files = os.listdir(folder_path)
                
                if cls_id not in self.template_cache:
                    self.template_cache[cls_id] = []
                
                # 處理一些模板並快取
                templates_to_add = max(2 - len(self.template_cache[cls_id]), 0)
                for _ in range(templates_to_add):
                    template_path = os.path.join(folder_path, random.choice(template_files))
                    nowhite_template_img = self.remove_white_background(template_path)
                    resize_template_img = self.random_resize_smaller_than(nowhite_template_img, max_size=(150, 150))
                    self.template_cache[cls_id].append(resize_template_img)
        
        # 從快取中選擇模板
        template_imgs_list1 = [random.choice(self.template_cache[cls_id]) for cls_id in random_cls]
        template_imgs_list2 = [random.choice(self.template_cache[cls_id]) for cls_id in random_cls]
        
        # 從快取中選擇背景
        bg_img1 = random.choice(self.background_cache).copy()
        bg_img2 = random.choice(self.background_cache).copy()
        
        # 調整背景大小
        new_img1 = self.resize_and_paste(bg_img1, output_size=output_size, maintain_aspect_ratio=False)
        new_img2 = self.resize_and_paste(bg_img2, output_size=output_size, maintain_aspect_ratio=False)
        
        # 貼上物件
        canvas_copy1, yolo_bboxes1, absolute_bboxes1 = self.paste_with_spacing(
            new_img1, template_imgs_list1, min_distance=10, class_ids=random_cls
        )
        canvas_copy2, yolo_bboxes2, absolute_bboxes2 = self.paste_with_spacing(
            new_img2, template_imgs_list2, min_distance=10, class_ids=random_cls
        )
        
        return {
            "query_img": canvas_copy1,
            "template_img": canvas_copy2,
            "query_img_bboxes": yolo_bboxes1,
            "template_img_bboxes": yolo_bboxes2,
            "object_class": random_cls,
        }
    
    def process_batch(self, batch_start, batch_size, IMG_PATH, LABEL_PATH, use_temp_dir=True):
        """處理一批次樣本"""
        results = []
        
        # 使用臨時目錄以減少 I/O 瓶頸
        if use_temp_dir:
            temp_img_dir = tempfile.mkdtemp()
            temp_label_dir = tempfile.mkdtemp()
            save_img_path = temp_img_dir
            save_label_path = temp_label_dir
        else:
            save_img_path = IMG_PATH
            save_label_path = LABEL_PATH
        
        for i in range(batch_start, batch_start + batch_size):
            try:
                # 獲取樣本
                sample = self.optimized_get_one_sample()
                
                # 保存輸入圖像和標註
                input_img = f"{i:07d}_I_"
                self.save_img(sample['query_img'], f"{save_img_path}/{input_img}.jpg")
                self.save_yolo_annotations(save_label_path, input_img, sample['query_img_bboxes'])
                
                # 保存模板圖像和標註
                template_img = f"{i:07d}_T_"
                self.save_img(sample['template_img'], f"{save_img_path}/{template_img}.jpg")
                self.save_yolo_annotations(save_label_path, template_img, sample['template_img_bboxes'])
                
                results.append((True, i))
            except Exception as e:
                results.append((False, f"處理樣本 {i} 時發生錯誤: {e}"))
        
        # 如果使用了臨時目錄，將文件移動到最終目錄
        if use_temp_dir:
            try:
                # 移動圖像
                for filename in os.listdir(temp_img_dir):
                    shutil.move(f"{temp_img_dir}/{filename}", f"{IMG_PATH}/{filename}")
                
                # 移動標註
                for filename in os.listdir(temp_label_dir):
                    shutil.move(f"{temp_label_dir}/{filename}", f"{LABEL_PATH}/{filename}")
                
                # 清理臨時目錄
                shutil.rmtree(temp_img_dir)
                shutil.rmtree(temp_label_dir)
            except Exception as e:
                print(f"移動文件時發生錯誤: {e}")
        
        return results
    
    def generate_dataset_parallel(self, num_samples, IMG_PATH, LABEL_PATH, batch_size=100, num_processes=None, preload=True):
        """使用平行處理生成大量樣本"""
        # 確保輸出目錄存在
        os.makedirs(IMG_PATH, exist_ok=True)
        os.makedirs(LABEL_PATH, exist_ok=True)
        
        # 確定使用的進程數
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        print(f"將使用 {num_processes} 個進程生成 {num_samples} 個樣本...")
        
        # 預加載資源以提高效率
        if preload:
            print("預加載資源...")
            self.preload_backgrounds(count=self.cache_size)
            self.preload_templates()
        
        # 準備批次參數
        num_batches = (num_samples + batch_size - 1) // batch_size
        batch_args = []
        
        for batch_id in range(num_batches):
            batch_start = batch_id * batch_size
            current_batch_size = min(batch_size, num_samples - batch_start)
            batch_args.append((batch_start, current_batch_size, IMG_PATH, LABEL_PATH))
        
        # 使用進程池處理批次
        start_time = time.time()
        
        with mp.Pool(processes=num_processes) as pool:
            all_results = []
            
            # 使用 tqdm 顯示進度
            for batch_results in tqdm(pool.starmap(self.process_batch, batch_args), total=len(batch_args), desc="生成樣本"):
                all_results.extend(batch_results)
        
        # 計算結果統計
        successful = sum(1 for success, _ in all_results if success)
        elapsed_time = time.time() - start_time
        
        print(f"\n完成! 成功生成 {successful}/{num_samples} 個樣本")
        print(f"總耗時: {elapsed_time:.2f} 秒, 平均每樣本 {elapsed_time/num_samples:.4f} 秒")
        
        # 列出錯誤（如果有）
        errors = [msg for success, msg in all_results if not success]
        if errors:
            print(f"發生了 {len(errors)} 個錯誤")
            error_file = "generation_errors.log"
            with open(error_file, "w") as f:
                for error in errors:
                    f.write(f"{error}\n")
            print(f"錯誤詳情已保存到 {error_file}")
    
    def generate_dataset_threadpool(self, num_samples, IMG_PATH, LABEL_PATH, max_workers=16):
        """使用線程池生成樣本（適用於 I/O 密集型任務）"""
        os.makedirs(IMG_PATH, exist_ok=True)
        os.makedirs(LABEL_PATH, exist_ok=True)
        
        # 預加載資源
        print("預加載資源...")
        self.preload_backgrounds(count=self.cache_size)
        self.preload_templates()
        
        def process_single_sample(i):
            try:
                sample = self.optimized_get_one_sample()
                
                input_img = f"{i:07d}_I"
                self.save_img(sample['query_img'], f"{IMG_PATH}/{input_img}.jpg")
                self.save_yolo_annotations(LABEL_PATH, input_img, sample['query_img_bboxes'])
                
                template_img = f"{i:07d}_T"
                self.save_img(sample['template_img'], f"{IMG_PATH}/{template_img}.jpg")
                self.save_yolo_annotations(LABEL_PATH, template_img, sample['template_img_bboxes'])
                
                return True, i
            except Exception as e:
                return False, f"處理樣本 {i} 時發生錯誤: {e}"
        
        start_time = time.time()
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任務
            future_to_idx = {executor.submit(process_single_sample, i): i for i in range(num_samples)}
            
            # 使用 tqdm 顯示進度
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=num_samples, desc="生成樣本"):
                all_results.append(future.result())
        
        # 計算結果統計
        successful = sum(1 for success, _ in all_results if success)
        elapsed_time = time.time() - start_time
        
        print(f"\n完成! 成功生成 {successful}/{num_samples} 個樣本")
        print(f"總耗時: {elapsed_time:.2f} 秒, 平均每樣本 {elapsed_time/num_samples:.4f} 秒")

# 使用範例
if __name__ == "__main__":
    # 設定參數

    TYPE = ["HEADBOARD"]
    NUM_SAMPLES_TRAIN = 300000
    NUM_SAMPLES_VAL = 512
    NUM_SAMPLES_TEST = 5000
    TRAIN_COUNT = 70
    NUM_CLASSES = int(os.environ.get('NUM_CLASSES', 50))
    IMG_PATH = f"DATASET/ABO/ABO_v2v_dataset_{NUM_CLASSES}_class/images"
    LABEL_PATH = f"DATASET/ABO/ABO_v2v_dataset_{NUM_CLASSES}_class/labels"
    
    BASE_PATH = f"DATASET/ABO/ABO_v2v_dataset_{TRAIN_COUNT}_seen_{92-TRAIN_COUNT}_unseen"
    IMG_PATH = f"{BASE_PATH}/images"
    LABEL_PATH = f"{BASE_PATH}/labels"
    # IMG_PATH = f"DATASET/ABO/ABO_v2v_dataset_{TRAIN_COUNT}_seen_{92-TRAIN_COUNT}_unseen/images"
    # LABEL_PATH = f"DATASET/ABO/ABO_v2v_dataset_{TRAIN_COUNT}_class/labels"
    # IMG_PATH = f"DATASET/ABO/ABO_v2v_dataset_{NUM_CLASSES}_class_only_{TYPE}/images"
    # LABEL_PATH = f"DATASET/ABO/ABO_v2v_dataset_{NUM_CLASSES}_class_only_{TYPE}/labels"
    BATCH_SIZE = 128
    CACHE_SIZE = 500
    NUM_PROCESSES = None  # 設為 None 將使用所有可用 CPU 核心
    
    # 創建優化版數據集
    ABO = accelerated_ABO_dataset(cache_size=CACHE_SIZE)
    
    # 隨機選擇類別
    # object_name = ABO.get_random_classes(NUM_CLASSES)
    # object_name = ABO.find_spin_ids_by_product_type("DATASET/ABO/listings/spin_split_json/listings_spin_split.json", TYPE)

    train_classes, val_classes = ABO.get_class_from_several_classes("v2vdet_ultralytics/utils/ABO_DATASET_Create/class_name.txt", train_count=TRAIN_COUNT)
    
    ABO.find_spin_ids_by_product_type(data="DATASET/ABO/listings/spin_split_json/listings_spin_split.json", product_type_value=train_classes, max_class_num=10)

    # ABO.dump_ultralytics_style_dataset_cfg(f"v2vdet_ultralytics/cfg/datasets/ABO_{ABO.nc}_{TYPE}.yaml")
    # ABO.dump_ultralytics_style_dataset_cfg(f"v2vdet_ultralytics/cfg/datasets/ABO_{ABO.nc}.yaml")
    ABO.dump_ultralytics_style_dataset_cfg(f"v2vdet_ultralytics/cfg/datasets/ABO_train.yaml")
    

    # 使用平行處理生成樣本
    print("Dump Training Set...")
    ABO.generate_dataset_parallel(
        num_samples=NUM_SAMPLES_TRAIN,
        IMG_PATH=f"{IMG_PATH}/train",
        LABEL_PATH=f"{LABEL_PATH}/train",
        batch_size=BATCH_SIZE,
        num_processes=NUM_PROCESSES,
        preload=True
    )
    print("Training Set has done!")

    print("Dump Validation Set...")
    # 使用平行處理生成樣本
    ABO.generate_dataset_parallel(
        num_samples=NUM_SAMPLES_VAL,
        IMG_PATH=f"{IMG_PATH}/val",
        LABEL_PATH=f"{LABEL_PATH}/val",
        batch_size=BATCH_SIZE,
        num_processes=NUM_PROCESSES,
        preload=True
    )
    print("Validation Set has done!")

    ABO.get_class_and_create_simple(json_data="DATASET/ABO/listings/spin_split_json/listings_spin_split.json", input_classes_list=val_classes, max_class_num=2)
    ABO.dump_ultralytics_style_dataset_cfg(f"v2vdet_ultralytics/cfg/datasets/ABO_unseen_val.yaml")

    print("Dump Validation Set...")
    # 使用平行處理生成樣本
    ABO.generate_dataset_parallel(
        num_samples=NUM_SAMPLES_VAL,
        IMG_PATH=f"{IMG_PATH}/unseen/val",
        LABEL_PATH=f"{LABEL_PATH}/unseen/val",
        batch_size=BATCH_SIZE,
        num_processes=NUM_PROCESSES,
        preload=True
    )
    print("Validation Set has done!")
    
    # print("Dump Testing Set...")
    # # 使用平行處理生成樣本
    # ABO.generate_dataset_parallel(
    #     num_samples=NUM_SAMPLES_TEST,
    #     IMG_PATH=f"{IMG_PATH}/test",
    #     LABEL_PATH=f"{LABEL_PATH}/test",
    #     batch_size=BATCH_SIZE,
    #     num_processes=NUM_PROCESSES,
    #     preload=True
    # )
    # print("Testing Set has done!")
    
    # 或者使用線程池（對於 I/O 密集型任務可能更有效）
    # ABO.generate_dataset_threadpool(
    #     num_samples=NUM_SAMPLES,
    #     IMG_PATH=IMG_PATH,
    #     LABEL_PATH=LABEL_PATH,
    #     max_workers=32
    # )