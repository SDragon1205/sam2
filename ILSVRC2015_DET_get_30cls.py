import os
import shutil
import xml.etree.ElementTree as ET

def filter_and_flatten_ILSVRC_DET(old_anno_root, old_img_root, new_anno_root, new_img_root):
    wnid_to_id = {
        "n02691156", "n02419796", "n02131653", "n02834778", "n01503061", "n02924116", "n02958343", "n02402425",
        "n02084071", "n02121808", "n02503517", "n02118333", "n02510455", "n02342885", "n02374451", "n02129165",
        "n01674464", "n02484322", "n03790512", "n02324045", "n02509815", "n02411705", "n01726692", "n02355227",
        "n02129604", "n04468005", "n01662784", "n04530566", "n02062744", "n02391049"
    }

    os.makedirs(new_anno_root, exist_ok=True)
    os.makedirs(new_img_root, exist_ok=True)

    for root, _, files in os.walk(old_anno_root):
        for file in files:
            if not file.endswith(".xml"):
                continue

            xml_path = os.path.join(root, file)
            tree = ET.parse(xml_path)
            root_elem = tree.getroot()
            objects = root_elem.findall("object")

            valid_objects = []
            for obj in objects:
                name = obj.find("name").text.strip()
                if name in wnid_to_id:
                    valid_objects.append(obj)

            # 如果完全沒有 30 類別，跳過
            if len(valid_objects) == 0:
                continue

            # 清除不是目標類別的標註
            for obj in objects:
                name = obj.find("name").text.strip()
                if name not in wnid_to_id:
                    root_elem.remove(obj)

            # 重新命名輸出用名稱（例如 n01234567_000123.xml）
            rel_path = os.path.relpath(xml_path, old_anno_root)
            name_base = rel_path.replace(os.sep, "_").replace(".xml", "")

            new_xml_name = f"{name_base}.xml"
            new_img_name = f"{name_base}.JPEG"

            new_xml_path = os.path.join(new_anno_root, new_xml_name)
            new_img_path = os.path.join(new_img_root, new_img_name)

            # 保存 xml
            tree.write(new_xml_path)

            # 對應圖檔複製
            img_path = os.path.join(old_img_root, rel_path.replace(".xml", ".JPEG"))
            if os.path.exists(img_path):
                shutil.copyfile(img_path, new_img_path)
            else:
                print(f"[WARN] JPEG not found for: {img_path}")

filter_and_flatten_ILSVRC_DET(
    old_anno_root="/home/user/sdragon/ILSVRC2015_DET/OpenDataLab___ILSVRC2015_DET/raw/ILSVRC2015/Annotations/DET",
    old_img_root="/home/user/sdragon/ILSVRC2015_DET/OpenDataLab___ILSVRC2015_DET/raw/ILSVRC2015/Data/DET",
    new_anno_root="/home/user/sdragon/ILSVRC2015_DET/OpenDataLab___ILSVRC2015_DET/raw/ILSVRC2015/Annotations/all_30cls/",
    new_img_root="/home/user/sdragon/ILSVRC2015_DET/OpenDataLab___ILSVRC2015_DET/raw/ILSVRC2015/Data/all_30cls/"
)