import os
import shutil
import xml.etree.ElementTree as ET

def convert_vid_to_image_dataset(img_folder, gt_folder): #, new_img_folder, new_gt_folder):
    # new_img_folder = os.path.join(img_folder, "train_image")
    # new_gt_folder = os.path.join(gt_folder, "val_image")
    # os.makedirs(new_img_folder, exist_ok=True)
    # os.makedirs(new_gt_folder, exist_ok=True)

    for split in ["train", "val"]:
        img_split_dir = os.path.join(img_folder, split)
        gt_split_dir = os.path.join(gt_folder, split)
        new_img_split = os.path.join(img_folder, f"{split}_image")
        new_gt_split = os.path.join(gt_folder, f"{split}_image")

        os.makedirs(new_img_split, exist_ok=True)
        os.makedirs(new_gt_split, exist_ok=True)

        video_folders = sorted(os.listdir(img_split_dir))
        for vid in video_folders:
            vid_img_dir = os.path.join(img_split_dir, vid)
            vid_gt_dir = os.path.join(gt_split_dir, vid)

            if not os.path.isdir(vid_img_dir):
                continue

            frame_files = sorted([
                f for f in os.listdir(vid_img_dir)
                if f.endswith(".JPEG") or f.endswith(".jpg") or f.endswith(".png")
            ])

            for frame_file in frame_files:
                frame_id = os.path.splitext(frame_file)[0]
                xml_file = os.path.join(vid_gt_dir, f"{frame_id}.xml")

                if not os.path.exists(xml_file):
                    continue

                tree = ET.parse(xml_file)
                root = tree.getroot()
                objects = root.findall("object")

                if len(objects) == 0:
                    continue  # 沒有物件，跳下一張

                # 找到第一張有標註的就處理它
                new_name = f"{vid}_{frame_id}"
                new_img_path = os.path.join(new_img_split, new_name + ".JPEG")
                new_xml_path = os.path.join(new_gt_split, new_name + ".xml")

                shutil.copyfile(os.path.join(vid_img_dir, frame_file), new_img_path)
                shutil.copyfile(xml_file, new_xml_path)
                break  # 這個影片只挑一張，跳下一個影片

convert_vid_to_image_dataset(
    img_folder="/home/user/sdragon/ILSVRC2015_VID/ILSVRC2015/Data/VID/",
    gt_folder="/home/user/sdragon/ILSVRC2015_VID/ILSVRC2015/Annotations/VID/",
    # new_img_folder="/path/to/new_dataset/images",
    # new_gt_folder="/path/to/new_dataset/annotations"
)