import os
import random
from pathlib import Path
from PIL import Image, ImageFilter
from typing import Union
from tqdm import tqdm
import json

ABO_DATASET_PATH = "DATASET/ABO/abo-spins/spins/original"

class create_ABO_dataset():
  
  photo_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']
  random_image_path = "DATASET/Object365/images/train"
  
  def __init__ (self, ABO_DATASET_PATH = "DATASET/ABO/abo-spins/spins/original"):
    self.abo_spin_dataset_path = ABO_DATASET_PATH
  
  def get_class_by_order(self, total_class=400):
    type_dirs = [os.path.join(self.abo_spin_dataset_path, d) for d in sorted(os.listdir(self.abo_spin_dataset_path)) if os.path.isdir(os.path.join(self.abo_spin_dataset_path, d))]
    object_name = [obj for type_dir in type_dirs for obj in sorted(os.listdir(type_dir))]
    
    self.object_name = [object_name[i] for i in range(total_class)]
    self.object_folder_path = [os.path.join(self.abo_spin_dataset_path, f"{object_name[i][:2]}/{object_name[i].split('_')[0]}") for i in range(total_class)]
    self.nc = total_class
    return self.object_name
  
  def get_random_classes(self, total_class=400):
    """
    從資料集中隨機選擇指定數量的類別
    
    參數:
    total_class (int): 要選擇的類別總數，預設為 400
    
    回傳:
    list: 隨機選擇的物件名稱列表
    """
    # 獲取所有類型目錄
    type_dirs = [os.path.join(self.abo_spin_dataset_path, d) for d in sorted(os.listdir(self.abo_spin_dataset_path)) 
                if os.path.isdir(os.path.join(self.abo_spin_dataset_path, d))]
    
    # 獲取所有物件名稱
    all_object_names = [obj for type_dir in type_dirs for obj in sorted(os.listdir(type_dir))]
    
    # 確保要選擇的類別數量不超過可用類別總數
    available_classes = len(all_object_names)
    if total_class > available_classes:
        print(f"警告: 要求的類別數量 {total_class} 超過了可用類別數量 {available_classes}。將使用所有可用類別。")
        total_class = available_classes
    
    # 隨機選擇類別
    random_indices = random.sample(range(available_classes), total_class)
    self.object_name = [all_object_names[i] for i in random_indices]
    
    # 設置物件資料夾路徑和類別數量
    self.object_folder_path = [os.path.join(self.abo_spin_dataset_path, f"{self.object_name[i][:2]}/{self.object_name[i].split('_')[0]}") 
                              for i in range(total_class)]
    self.nc = total_class
    
    return self.object_name
  
  def get_class_and_idx_from_several_classes(self, json_data, class_name_txt, train_count=80, max_class_num=2):
      
      '''
        Total 90 Classes,
        By default is 80 for training and 10 for eval as unseen classes.
      '''

      train_classes, val_classes = self.get_class_from_several_classes(class_name_txt, train_count)

      train_object_name_list = []
      for train_class in train_classes:
        train_object_name_list.extend(self.find_spin_ids_by_product_type(json_data, train_class, max_class_num=max_class_num))
    
      val_object_name_list = []
      for val_class in val_classes:
        val_object_name_list.extend(self.find_spin_ids_by_product_type(json_data, val_class, max_class_num=max_class_num))
      
      return {
        "train": train_object_name_list,
        "val": val_object_name_list,
      }
  
  def get_class_and_create_simple(self, json_data, input_classes_list, max_class_num=2):
    
    self.find_spin_ids_by_product_type(json_data, input_classes_list, max_class_num=max_class_num)
  
  def get_class_from_several_classes(self, class_name_txt, train_count=80):

    '''
        Total Classes are 90 classes,
        By default is 80 for training and 10 for eval as unseen classes.
    '''

    def read_product_types(class_name_txt):
        with open(class_name_txt, 'r', encoding='utf-8') as file:
            product_types = [line.strip() for line in file if line.strip()]
        return product_types

    # 隨機分配產品類型到兩個清單
    def split_product_types(all_products, main_count=80):
        # 確保產品類型被打亂
        shuffled_products = all_products.copy()
        random.shuffle(shuffled_products)
        
        # 分成兩個清單
        main_list = shuffled_products[:main_count]
        remaining_list = shuffled_products[main_count:]
        
        return main_list, remaining_list

    all_product_types = read_product_types(class_name_txt)

    train_classes, val_classes = split_product_types(all_product_types, main_count=train_count)

    return train_classes, val_classes
  
  def find_spin_ids_by_product_type(self, data, product_type_value, max_class_num=-1):
    """
    從 JSON 數據中找出符合特定 product_type value 的所有 spin_id
    
    Product Type               Total  Train    Val   Test
    =======================================================
    CHAIR                        994    798    104     92
    RUG                          802    639     69     94
    SOFA                         665    530     67     68
    WALL_ART                     595    492     44     59
    HOME_FURNITURE_AND_DECOR     460    354     55     51
    TABLE                        387    317     29     41
    LIGHT_FIXTURE                358    284     33     41
    LAMP                         334    267     36     31
    STOOL_SEATING                319    256     32     31
    OTTOMAN                      284    225     30     29
    PILLOW                       271    216     30     25
    HOME                         212    170     31     11
    HEADBOARD                    199    148     23     28
    PLANTER                      182    147     17     18
    BED                          138    114     10     14
    HOME_MIRROR                  106     90     10      6
    SHELF                         67     53      7      7
    DESK                          66     56      4      6
    VASE                          50     39      4      7
    CABINET                       38     29      3      6
    FURNITURE_COVER               34     29      4      1
    BENCH                         33     25      4      4
    ELECTRIC_FAN                  29     21      8      0
    SPORTING_GOODS                27     18      7      2
    DRESSER                       25     19      2      4
    BED_FRAME                     22     17      1      4
    FREESTANDING_SHELTER          18     12      1      5
    STORAGE_BOX                   18     14      3      1
    CURTAIN                       17     12      2      3
    OUTDOOR_LIVING                16     13      2      1
    HOME_LIGHTING_AND_LAMPS       15     10      2      3
    HOME_BED_AND_BATH             14     12      2      0
    AUTO_ACCESSORY                14     13      0      1
    AIR_CONDITIONER               13     10      0      3
    WIRELESS_ACCESSORY            13     11      1      1
    BEAN_BAG_CHAIR                12     10      1      1
    LADDER                        11      9      1      1
    CLOTHES_RACK                  10     10      0      0
    COMPUTER_ADD_ON               10      6      1      2
    BUILDING_MATERIAL              9      8      1      0
    CANDLE_HOLDER                  9      8      0      1
    FURNITURE                      8      7      0      1
    EXERCISE_MAT                   7      6      1      0
    PICTURE_FRAME                  7      3      2      2
    WILDLIFE_FEEDER                6      4      2      0
    CLOCK                          6      5      1      0
    FLAT_SCREEN_DISPLAY_MOUNT      6      6      0      0
    PLACEMAT                       5      4      1      0
    BISS                           5      3      1      1
    BOTTLE_RACK                    5      2      2      1
    WRITING_BOARD                  5      4      1      0
    PORTABLE_ELECTRONIC_DEVICE_STAND      5      3      2      0
    JANITORIAL_SUPPLY              5      5      0      0
    SAUTE_FRY_PAN                  5      4      1      0
    STORAGE_HOOK                   4      4      0      0
    CE_ACCESSORY                   4      3      1      0
    SOUND_AND_RECORDING_EQUIPMENT      4      4      0      0
    INSTRUMENT_PARTS_AND_ACCESSORIES      3      2      1      0
    WATER_PURIFICATION_UNIT        3      2      1      0
    DEHUMIDIFIER                   3      2      0      1
    TOOLS                          3      1      0      2
    DRINKING_CUP                   3      3      0      0
    PAPER_PRODUCT                  3      2      0      1
    PORTABLE_ELECTRONIC_DEVICE_COVER      2      2      0      0
    PORTABLE_ELECTRONIC_DEVICE_MOUNT      2      2      0      0
    ANIMAL_COLLAR                  2      1      0      1
    DRINK_COASTER                  2      2      0      0
    DISHWARE_BOWL                  2      1      0      1
    HANDBAG                        2      2      0      0
    SCULPTURE                      1      1      0      0
    MATTRESS                       1      1      0      0
    FILE_FOLDER                    1      0      0      1
    CELLULAR_PHONE_CASE            1      1      0      0
    MOUSE_PAD                      1      1      0      0
    JAR                            1      1      0      0
    ELECTRONIC_ADAPTER             1      1      0      0
    BODY_POSITIONER                1      1      0      0
    FIGURINE                       1      1      0      0
    FITNESS_BENCH                  1      1      0      0
    JEWELRY_STORAGE                1      1      0      0
    BLANKET                        1      1      0      0
    UTILITY_CART_WAGON             1      1      0      0
    ACCESSORY_OR_PART_OR_SUPPLY      1      0      0      1
    ABIS_BOOK                      1      1      0      0
    CARRYING_CASE_OR_BAG           1      1      0      0
    BASKET                         1      1      0      0
    OFFICE_PRODUCTS                1      1      0      0
    ARTIFICIAL_PLANT               1      1      0      0
    COMPUTER                       1      1      0      0
    BATTERY                        1      1      0      0
    STORAGE_RACK                   1      1      0      0
    KITCHEN                        1      0      0      1

    參數:
    data (dict): 包含 'train', 'val', 'test' 鍵的字典數據
    product_type_value (list): 要搜尋的 product_type value 值，例如 ['SOFA', 'CHAIR']
    
    返回:
    dict: 按照數據集分類的符合條件的 spin_id 字典
    """
    # result = {
    #     'train': [],
    #     'val': [],
    #     'test': []
    # }
    result = []

    if isinstance(data, str):
        with open(data, 'r') as f:
            data = json.load(f)
    elif not isinstance(data, dict):
        raise ValueError("data must be a file path or a dictionary")
    
    if not isinstance(product_type_value, list):
        raise ValueError("Input: product_type_value must be a list")

    # 遍歷所有數據集

    for product_type in product_type_value:
        each_type_result = []
        for dataset_name in ['train', 'val', 'test']:
            if dataset_name in data:
                # 遍歷數據集中的每個項目
                for item in data[dataset_name]:
                    # 檢查 product_type 列表
                    for inner_product_type in item.get('product_type', []):
                        # 如果找到匹配的 value
                        if inner_product_type.get('value') == product_type:
                            # result[dataset_name].append(item['spin_id'])
                            each_type_result.append(f'{item['spin_id']}_{product_type}')
                            break  # 找到匹配後跳出內部循環
        
        if max_class_num > 0 and len(each_type_result) > 0:
            each_type_result = random.sample(each_type_result, min(max_class_num, len(each_type_result)))

        result.extend(each_type_result)

    self.object_name = result
    self.object_folder_path = [os.path.join(self.abo_spin_dataset_path, f"{self.object_name[i][:2]}/{self.object_name[i].split('_')[0]}") 
                              for i in range(len(result))]
    
    self.nc = len(result)
    return result
  
  def dump_ultralytics_style_dataset_cfg(self, output_path=None):
      if output_path is None:
         output_path = f"v2vdet_ultralytics/cfg/datasets/ABO_{self.nc}.yaml"
      with open(output_path, 'w') as f:
        f.write(f"# Classes\n")
        f.write(f"names:\n")
        for class_idx, class_name in enumerate(self.object_name):
           f.write(f"  {class_idx}: {class_name}\n")
  
  def get_one_sample(self, object_amount_bound=10, output_size=(640, 640), random_image_folder_path="DATASET/Object365/images/train", min_distance=10, max_tries=50, max_object_size=(150, 150)):
    """
    生成一個樣本，包含隨機選擇的物件貼在隨機背景上
    
    參數:
    object_amount_bound (int): 最大物件類別數量
    output_size (tuple): 輸出圖像大小 (寬, 高)
    random_image_folder_path (str): 隨機背景圖像的資料夾路徑
    min_distance (int): 物件之間的最小距離
    max_tries (int): 嘗試放置物件的最大次數
    max_object_size (tuple): 物件的最大尺寸
    
    回傳:
    tuple: (兩個圖像樣本, 兩個YOLO邊界框列表, 兩個絕對邊界框列表, 物件類別列表)
    """
    # 選擇隨機物件類別
    total_nc = random.randint(5, min(object_amount_bound, self.nc))
    random_cls = [random.randint(0, self.nc-1) for _ in range(total_nc)]
    random_cls = sorted(list(set(random_cls)))  # 移除重複並排序
    
    # 更新暫存變數
    temp_total_nc = len(random_cls)
    temp_object_name = [self.object_name[_] for _ in random_cls]
    temp_object_folder_path = [self.object_folder_path[_] for _ in random_cls]
    
    # 生成兩組樣本使用的函數
    def generate_template_images(folder_paths):
        """為每個類別創建一個已去背和調整大小的模板圖像"""
        template_imgs = []
        for idx, folder_path in enumerate(folder_paths):
            template_files = os.listdir(folder_path)
            template_path = os.path.join(folder_path, random.choice(template_files))
            nowhite_template_img = self.remove_white_background(template_path)
            resize_template_img = self.random_resize_smaller_than(nowhite_template_img, max_size=max_object_size)
            template_imgs.append(resize_template_img)
        return template_imgs
    
    # 生成兩組模板圖像
    template_imgs_list1 = generate_template_images(temp_object_folder_path)
    template_imgs_list2 = generate_template_images(temp_object_folder_path)
    
    # 生成樣本的函數
    def create_sample(template_imgs):
        """創建一個樣本，將模板圖像貼到隨機背景上"""
        # 選擇隨機背景圖像
        files = os.listdir(random_image_folder_path)
        input_path = os.path.join(random_image_folder_path, random.choice(files))
        img = Image.open(input_path)
        
        # 調整背景圖像大小
        new_img = self.resize_and_paste(img, output_size=output_size, maintain_aspect_ratio=False)
        
        # 貼上物件，確保它們之間保持距離
        canvas_copy, yolo_bboxes, absolute_bboxes = self.paste_with_spacing(
            new_img, 
            template_imgs, 
            min_distance=min_distance, 
            max_attempts=max_tries,
            class_ids=random_cls
        )
        
        return canvas_copy, yolo_bboxes, absolute_bboxes
    
    # 創建兩個樣本
    canvas1, yolo_bboxes1, absolute_bboxes1 = create_sample(template_imgs_list1)
    canvas2, yolo_bboxes2, absolute_bboxes2 = create_sample(template_imgs_list2)
    
    # 視覺化結果
    # self.visualize_bboxes(canvas1, absolute_bboxes1, class_names=self.temp_object_name, output_path='test1.png')
    # self.visualize_bboxes(canvas2, absolute_bboxes2, class_names=self.temp_object_name, output_path='test2.png')

    return {
        "query_img": canvas1,
        "template_img": canvas2,
        "query_img_bboxes": yolo_bboxes1,
        "template_img_bboxes": yolo_bboxes2,
        "object_class": random_cls,
    }
    
    return (canvas1, canvas2), (yolo_bboxes1, yolo_bboxes2), (absolute_bboxes1, absolute_bboxes2), self.temp_object_name
    
  def old_get_one_sample(self, object_amount_bound=10, output_size=(640, 640), random_image_folder_path = "DATASET/Object365/images/train"): 
    
    # Select random template objects
    total_nc = random.randint(5, object_amount_bound)
    random_cls = [random.randint(0, self.nc-1) for _ in range(total_nc)]
    random_cls = sorted(list(set(random_cls)))
    self.temp_total_nc = len(random_cls)
    self.temp_object_name = [self.object_name[_] for _ in random_cls]
    self.temp_object_folder_path = [self.object_folder_path[_] for _ in random_cls]   
    
    template_imgs_list = []
    for idx, folder_path in enumerate(self.temp_object_folder_path):
      template_files_list = os.listdir(folder_path)
      template_path = os.path.join(folder_path, random.choice(template_files_list))
      nowhite_template_img = self.remove_white_background(template_path)
      resize_template_img = self.random_resize_smaller_than(nowhite_template_img, max_size=(150, 150))
      template_imgs_list.append(resize_template_img)
      
    template_imgs_list2 = []
    for idx, folder_path in enumerate(self.temp_object_folder_path):
      template_files_list = os.listdir(folder_path)
      template_path = os.path.join(folder_path, random.choice(template_files_list))
      nowhite_template_img = self.remove_white_background(template_path)
      resize_template_img = self.random_resize_smaller_than(nowhite_template_img, max_size=(150, 150))
      template_imgs_list2.append(resize_template_img)
    
    # --------
    # Pick a random image from the random_image_path
    files = os.listdir(self.random_image_path)
    input_path = os.path.join(self.random_image_path, random.choice(files))
    
    img = Image.open(input_path)
    
    new_img = self.resize_and_paste(img, output_size=(640, 640), maintain_aspect_ratio=False)
    
    # canvas_copy, yolo_bboxes, absolute_bboxes = self.paste_multiple_images(new_img, template_imgs_list, random_cls, num_to_paste=random.randint(1, len(random_cls)))
    canvas_copy, yolo_bboxes, absolute_bboxes = self.paste_with_spacing(new_img, template_imgs_list, min_distance=10, class_ids=random_cls)
    self.visualize_bboxes(canvas_copy, absolute_bboxes, class_names=random_cls, output_path='test.png')


    # ------
    files = os.listdir(self.random_image_path)
    input_path = os.path.join(self.random_image_path, random.choice(files))
    
    img = Image.open(input_path)
    
    new_img = self.resize_and_paste(img, output_size=(640, 640), maintain_aspect_ratio=False)

    canvas_copy2, yolo_bboxes2, absolute_bboxes2 = self.paste_with_spacing(new_img, template_imgs_list2, min_distance=10, class_ids=random_cls)
    self.visualize_bboxes(canvas_copy, absolute_bboxes, class_names=random_cls, output_path='test2.png')
    pass
    
  def remove_white_background(self, img: Union[str, Image.Image], edge_detection=True, tolerance=30, output_path=None) -> Image:
    """
    移除照片中的白色背景，將其轉換為透明背景
    
    參數:
    image_path (str): 輸入照片的路徑
    output_path (str): 輸出照片的路徑，若為 None 則使用原始檔案名稱加上 '_transparent'
    tolerance (int): 判斷為白色的容忍度 (0-255)，越高則移除範圍越廣
    
    回傳:
    str: 去背後的照片路徑，如果處理失敗則回傳 None
    """
    try:
        # 開啟照片
        if isinstance(img, str):
          img = Image.open(img)
        elif not isinstance(img, Image.Image): 
          raise ValueError("img must be a file path or PIL Image object")
        
        # 轉換為 RGBA 模式 (紅, 綠, 藍, 透明度)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
                # 如果啟用邊緣檢測
        if edge_detection:
            # 創建邊緣檢測遮罩
            edge_img = img.convert('L').filter(ImageFilter.FIND_EDGES)
            edge_img = edge_img.filter(ImageFilter.SMOOTH)  # 平滑化邊緣
            
            # 獲取圖像數據
            img_data = img.getdata()
            edge_data = edge_img.getdata()
            width, height = img.size
            
            # 創建新的數據，更智能地移除背景
            new_data = []
            for i, item in enumerate(img_data):
                # 獲取該像素的邊緣強度
                edge_value = edge_data[i]
                
                # 計算像素的亮度 (簡單平均)
                brightness = (item[0] + item[1] + item[2]) / 3
                
                # 計算像素偏離白色的程度
                whiteness = 255 - brightness
                
                # 檢查像素是否滿足以下條件:
                # 1. 接近白色
                # 2. 不是邊緣部分
                # 3. 不是物體內部的紋理
                if (item[0] > 255 - tolerance and 
                    item[1] > 255 - tolerance and 
                    item[2] > 255 - tolerance and 
                    edge_value < 30 and 
                    whiteness < tolerance):
                    # 將該像素設為完全透明 (alpha=0)
                    new_data.append((255, 255, 255, 0))
                else:
                    # 保留原始像素
                    new_data.append(item)
        else:
            # 使用原始的簡單去背方法
            img_data = img.getdata()
            new_data = []
            for item in img_data:
                if (item[0] > 255 - tolerance and 
                    item[1] > 255 - tolerance and 
                    item[2] > 255 - tolerance):
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)
        
        # 獲取圖像數據
        datas = img.getdata()
        
        # 創建新的數據，將白色背景變為透明
        new_data = []
        for item in datas:
            # 檢查像素是否接近白色 (判斷 R, G, B 值是否都接近 255)
            if item[0] > 255 - tolerance and item[1] > 255 - tolerance and item[2] > 255 - tolerance:
                # 將該像素設為完全透明 (alpha=0)
                new_data.append((255, 255, 255, 0))
            else:
                # 保留原始像素
                new_data.append(item)
        
        # 更新圖像數據
        img.putdata(new_data)
        
        # 設定輸出路徑
        if output_path is not None:
            filename = os.pathjoin(output_path, '.png')
            # 保存為 PNG 格式 (支援透明度)
            img.save(filename, 'PNG')
            print(f"已去除白色背景並保存至: {filename}")
        
        return img
    
    except Exception as e:
        print(f"Some Error occurred: {e}")
  
  def random_resize_smaller_than(self, img: Union[str, Image.Image], max_size=(320, 320), output_path=None):
    """
    隨機縮小照片，確保尺寸小於指定的最大尺寸
    
    參數:
    img Union[str, Image]: Path or Image object
    max_size (tuple): 最大尺寸 (寬, 高)
    output_path (str): 輸出照片的路徑，若為 None 則使用原始檔案名稱加上 '_small'
    
    回傳:
    tuple: (調整後的照片路徑, 新尺寸)，如果處理失敗則回傳 (None, None)
    """
    try:
        # 開啟照片
        if isinstance(img, str):
          img = Image.open(img)
        elif not isinstance(img, Image.Image):
          raise ValueError("img must be a file path or PIL Image object")
        
        # 確保是 RGBA 模式以保留透明度
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # 獲取原始尺寸
        original_width, original_height = img.size
        
        # 計算縮放比例，確保寬和高都小於最大尺寸
        scale_width = max_size[0] / original_width
        scale_height = max_size[1] / original_height
        scale = min(scale_width, scale_height)
        
        # 隨機生成一個比例，介於 0.25 和計算出的最大比例之間
        # 如果計算出的比例已經小於 0.25，則使用該比例
        if scale >= 0.25:
            random_scale = random.uniform(0.25, scale)
        else:
            random_scale = scale
        
        # 計算新尺寸
        new_width = int(original_width * random_scale)
        new_height = int(original_height * random_scale)
        
        # 縮小照片
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # 設定輸出路徑
        if output_path is not None:
          # filename = Path(output_path).stem
          # output_path = os.path.join(os.path.dirname(image_path), f"{filename}_small.png")
        
          # 保存為 PNG 格式
          resized_img.save(output_path, 'PNG')
          print(f"已隨機縮小照片並保存至: {output_path}")
          print(f"新尺寸: {new_width} x {new_height}")
        
        return resized_img
    
    except Exception as e:
        print(f"處理照片時發生錯誤: {e}")
        return None, None

  def paste_image_at_random_position(self, canvas, image, class_id=0):
    """
    將照片貼到畫布上的隨機位置，並返回 YOLO 格式的邊界框座標
    
    參數:
    canvas (PIL.Image): 要貼上的畫布
    image (PIL.Image 或 str): 要貼上的照片或照片路徑
    class_id (int): 物件的類別編號
    
    回傳:
    tuple: (貼上照片後的畫布, YOLO 格式的邊界框座標 [class_id, x_center, y_center, width, height])
    """
    try:
        # 如果輸入是路徑，則開啟照片
        if isinstance(image, str):
            img = Image.open(image)
            # 確保是 RGBA 模式
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        else:
            img = image
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        
        # 獲取畫布和照片的尺寸
        canvas_width, canvas_height = canvas.size
        img_width, img_height = img.size
        
        # 確保照片不會超出畫布
        max_x = canvas_width - img_width
        max_y = canvas_height - img_height
        if max_x < 0: max_x = 0
        if max_y < 0: max_y = 0
        
        # 隨機選擇位置 (左上角座標)
        x_min = random.randint(0, max_x)
        y_min = random.randint(0, max_y)
        
        # 計算右下角座標
        x_max = x_min + img_width
        y_max = y_min + img_height
        
        # 將照片貼到畫布上，使用 alpha 通道作為遮罩
        canvas.paste(img, (x_min, y_min), img)
        
        # 計算 YOLO 格式的邊界框座標 [class_id, x_center, y_center, width, height]
        x_center = (x_min + x_max) / (2 * canvas_width)  # 中心點 x 座標（相對值）
        y_center = (y_min + y_max) / (2 * canvas_height)  # 中心點 y 座標（相對值）
        width = img_width / canvas_width  # 寬度（相對值）
        height = img_height / canvas_height  # 高度（相對值）
        
        # YOLO 格式的邊界框
        yolo_bbox = [class_id, x_center, y_center, width, height]
        
        # 也計算絕對像素座標，方便參考
        absolute_bbox = [x_min, y_min, x_max, y_max]
        
        return canvas, yolo_bbox, absolute_bbox
    
    except Exception as e:
        print(f"貼上照片時發生錯誤: {e}")
        return canvas, None, None

  def paste_multiple_images(self, canvas, images, class_ids=None, num_to_paste=1):
      """
      將多張照片貼到畫布上的隨機位置，並返回所有 YOLO 格式的邊界框座標
      
      參數:
      canvas (PIL.Image): 要貼上的畫布
      images (list): 要貼上的照片列表 (可以是 PIL.Image 物件或照片路徑)
      class_ids (list): 每張照片對應的類別編號，若為 None 則所有照片使用類別 0
      num_to_paste (int): 要貼上的照片數量
      
      回傳:
      tuple: (貼上照片後的畫布, YOLO 格式的邊界框座標列表)
      """
      if class_ids is None:
          class_ids = [0] * len(images)
      
      # 確保 num_to_paste 不超過照片數量
      num_to_paste = min(num_to_paste, len(images))
      
      # 複製畫布，避免修改原始畫布
      canvas_copy = canvas.copy()
      
      # 用於存儲所有 YOLO 格式的邊界框座標
      yolo_bboxes = []
      absolute_bboxes = []
      
      # 隨機選擇並貼上指定數量的照片
      selected_indices = random.sample(range(len(images)), num_to_paste)
      
      for idx in selected_indices:
          image = images[idx]
          class_id = class_ids[idx]
          
          # 貼上照片並獲取邊界框座標
          canvas_copy, yolo_bbox, absolute_bbox = self.paste_image_at_random_position(canvas_copy, image, class_id)
          
          if yolo_bbox:
              yolo_bboxes.append(yolo_bbox)
              absolute_bboxes.append(absolute_bbox)
      
      return canvas_copy, yolo_bboxes, absolute_bboxes
    
  def resize_and_paste(self, img, output_size=(640, 640), maintain_aspect_ratio=True, background_color=(0, 0, 0), center=True):
    """
    調整圖像大小並貼到指定大小的畫布上
    
    參數:
    img (PIL.Image): 要調整大小的圖像
    output_size (tuple): 輸出圖像的大小 (寬, 高)
    maintain_aspect_ratio (bool): 是否保持原始比例
        若為 True，將按比例縮小後置中貼上
        若為 False，將強制縮放至指定大小
    background_color (tuple): 背景顏色 (R, G, B) 或 (R, G, B, A)
    center (bool): 是否將圖像置中（僅當 maintain_aspect_ratio=True 時有效）
    
    回傳:
    PIL.Image: 處理後的圖像
    """
    # 檢查輸入圖像的模式
    output_mode = "RGBA" if len(background_color) == 4 or img.mode == "RGBA" else "RGB"
    
    # 創建新的背景圖像
    new_img = Image.new(output_mode, output_size, background_color)
    
    if maintain_aspect_ratio:
        # 按比例縮小
        img_ratio = min(output_size[0]/img.width, output_size[1]/img.height)
        new_size = (int(img.width * img_ratio), int(img.height * img_ratio))
        img_resized = img.resize(new_size, Image.LANCZOS)
        
        # 根據 center 參數決定貼上位置
        if center:
            # 將調整大小後的圖像置中貼上
            paste_position = ((output_size[0] - new_size[0]) // 2, (output_size[1] - new_size[1]) // 2)
        else:
            # 將調整大小後的圖像貼在左上角
            paste_position = (0, 0)
        
        # 如果輸入圖像有透明通道，使用透明通道作為遮罩
        if img_resized.mode == 'RGBA' and output_mode == 'RGBA':
            new_img.paste(img_resized, paste_position, img_resized)
        else:
            new_img.paste(img_resized, paste_position)
    else:
        # 強制縮放至指定大小（不保持原始比例）
        img_resized = img.resize(output_size, Image.LANCZOS)
        
        # 直接使用調整後的圖像
        if output_mode == 'RGBA' and img_resized.mode == 'RGBA':
            # 保留透明度
            new_img = img_resized
        else:
            # 轉換為指定的輸出模式
            new_img = img_resized.convert(output_mode)
    
    return new_img

  def save_yolo_annotations(self, output_dir, image_name, yolo_bboxes):
    """
    保存 YOLO 格式的標註文件
    
    參數:
    output_dir (str): 輸出目錄路徑
    image_name (str): 圖像文件名（不含副檔名）
    yolo_bboxes (list): YOLO 格式的邊界框座標列表
    """
    # 創建輸出目錄（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 標註文件路徑
    annotation_path = os.path.join(output_dir, f"{image_name}.txt")
    
    # 寫入標註信息
    with open(annotation_path, 'w') as f:
        for bbox in yolo_bboxes:
            # 將邊界框座標轉換為文本格式
            bbox_str = ' '.join(map(str, bbox))
            f.write(f"{bbox_str}\n")
    
    # print(f"已保存 YOLO 標註至: {annotation_path}")

  def visualize_bboxes(self, image, absolute_bboxes, class_names=None, output_path=None):
    """
    在圖像上可視化邊界框
    
    參數:
    image (PIL.Image): 要可視化的圖像
    absolute_bboxes (list): 絕對像素座標的邊界框列表 [x_min, y_min, x_max, y_max]
    class_names (list): 類別名稱列表，若為 None 則顯示類別編號
    output_path (str): 輸出圖像路徑，若為 None 則僅顯示不保存
    
    回傳:
    PIL.Image: 可視化後的圖像
    """
    from PIL import ImageDraw, ImageFont
    
    # 複製圖像以避免修改原始圖像
    result_img = image.copy()
    
    # 創建繪圖對象
    draw = ImageDraw.Draw(result_img)
    
    # 對每個邊界框進行繪製
    for bbox in absolute_bboxes:
        x_min, y_min, x_max, y_max = bbox
        
        # 繪製矩形
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
    
    # 保存結果圖像
    if output_path:
        result_img.save(output_path)
        print(f"已保存可視化結果至: {output_path}")
    
    return result_img
  
  def paste_with_spacing(self, canvas, images, min_distance=50, max_attempts=100, class_ids=None):
    """
    將多個圖像貼到畫布上，確保它們之間保持一定距離
    
    參數:
    canvas (PIL.Image): 要貼上的畫布
    images (list): 要貼上的圖像列表
    min_distance (int): 圖像之間的最小距離（像素）
    max_attempts (int): 每個圖像的最大嘗試次數
    class_ids (list): 每個圖像的類別 ID，若為 None 則所有圖像使用類別 0
    
    回傳:
    tuple: (貼上圖像後的畫布, YOLO 格式的邊界框座標列表, 絕對像素的邊界框座標列表)
    """
    if class_ids is None:
        class_ids = [0] * len(images)
    
    # 複製畫布，避免修改原始畫布
    canvas_copy = canvas.copy()
    canvas_width, canvas_height = canvas_copy.size
    
    # 記錄已貼上圖像的邊界框
    placed_boxes = []  # 格式: [x_min, y_min, x_max, y_max]
    yolo_boxes = []    # YOLO 格式: [class_id, x_center, y_center, width, height]
    
    for i, img in enumerate(images):
        # 如果輸入是路徑，則開啟圖像
        if isinstance(img, str):
            img = Image.open(img)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        
        # 獲取圖像尺寸
        img_width, img_height = img.size
        
        # 嘗試找到合適的位置
        position_found = False
        attempts = 0
        
        while not position_found and attempts < max_attempts:
            # 隨機選擇位置 (左上角座標)
            x_min = random.randint(0, canvas_width - img_width)
            y_min = random.randint(0, canvas_height - img_height)
            x_max = x_min + img_width
            y_max = y_min + img_height
            
            # 檢查與已貼上圖像的距離
            too_close = False
            for box in placed_boxes:
                # 計算兩個邊界框之間的距離
                # 如果兩個框不重疊，則距離為它們最近邊之間的距離
                # 如果重疊，則距離為負數
                x_distance = max(0, min(box[2], x_max) - max(box[0], x_min))
                y_distance = max(0, min(box[3], y_max) - max(box[1], y_min))
                
                # 檢查是否重疊
                if x_distance > 0 and y_distance > 0:
                    # 有重疊，距離為負
                    distance = -1
                else:
                    # 不重疊，計算最近邊之間的距離
                    if x_distance == 0 and y_distance == 0:
                        # 兩個框完全分離
                        distance = min(
                            abs(box[0] - x_max),  # 左-右
                            abs(box[2] - x_min),  # 右-左
                            abs(box[1] - y_max),  # 上-下
                            abs(box[3] - y_min)   # 下-上
                        )
                    elif x_distance == 0:
                        # 水平方向分離
                        distance = min(
                            abs(box[0] - x_max),  # 左-右
                            abs(box[2] - x_min)   # 右-左
                        )
                    else:  # y_distance == 0
                        # 垂直方向分離
                        distance = min(
                            abs(box[1] - y_max),  # 上-下
                            abs(box[3] - y_min)   # 下-上
                        )
                
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                position_found = True
                
                # 將圖像貼到畫布上
                if img.mode == 'RGBA':
                    canvas_copy.paste(img, (x_min, y_min), img)
                else:
                    canvas_copy.paste(img, (x_min, y_min))
                
                # 記錄邊界框
                placed_boxes.append([x_min, y_min, x_max, y_max])
                
                # 計算 YOLO 格式的邊界框座標
                x_center = (x_min + x_max) / (2 * canvas_width)
                y_center = (y_min + y_max) / (2 * canvas_height)
                width = img_width / canvas_width
                height = img_height / canvas_height
                
                yolo_boxes.append([class_ids[i], x_center, y_center, width, height])
            
            attempts += 1
        
        if not position_found:
            print(f"警告: 無法找到合適的位置放置圖像 {i+1}，已嘗試 {max_attempts} 次")
    
    return canvas_copy, yolo_boxes, placed_boxes
  
  def save_img(self, img, output_path):
    """
    保存圖像到指定路徑
    
    參數:
    img (PIL.Image): 要保存的圖像
    output_path (str): 輸出路徑
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    # print(f"已保存圖像至: {output_path}")

if __name__ == "__main__":
    ABO = create_ABO_dataset()
    object_name = ABO.get_random_classes(80)
    ABO.dump_ultralytics_style_dataset_cfg()

    IMG_PATH = "DATASET/ABO/v2v_dataset/images/train"
    LABEL_PATH = "DATASET/ABO/v2v_dataset/labels/train"

    for i in tqdm(range(10)):
        gay = ABO.get_one_sample()

        input_img = f"{i:07d}_I_"
        ABO.save_img(gay['query_img'], f"{IMG_PATH}/{input_img}.jpg")
        ABO.save_yolo_annotations(LABEL_PATH, f"{input_img}.txt", gay['query_img_bboxes'])
        
        template_img = f"{i:07d}_T_"
        ABO.save_img(gay['template_img'], f"{IMG_PATH}/{template_img}.jpg")
        ABO.save_yolo_annotations(LABEL_PATH, f"{template_img}.txt", gay['template_img_bboxes'])