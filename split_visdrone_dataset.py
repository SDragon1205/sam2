import os
import shutil
import cv2

def split_dataset(img_folder, gt_folder, output_folder, chunk_size=10):
    """
    Splits each video dataset into multiple subfolders and updates annotations.
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "sequences"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "annotations"), exist_ok=True)  # Ensure annotation folder exists

    video_folders = sorted(os.listdir(img_folder))
    
    for video_name in video_folders:
        video_path = os.path.join(img_folder, video_name)
        gt_file = os.path.join(gt_folder, f"{video_name}.txt")
        
        if not os.path.isdir(video_path) or not os.path.exists(gt_file):
            continue
        
        # Read all image files
        image_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg") or f.endswith(".png")])
        
        if len(image_files) <= chunk_size:
            continue  # Skip if not enough frames
        
        # Read annotations
        with open(gt_file, "r") as f:
            lines = f.readlines()
        
        annotations = {}  # {frame_idx: [annotation_lines]}
        for line in lines:
            parts = line.strip().split(",")
            frame_idx = int(parts[0])
            if frame_idx not in annotations:
                annotations[frame_idx] = []
            annotations[frame_idx].append(line)
        
        num_splits = len(image_files) // chunk_size
        
        for split_idx in range(num_splits):
            split_name = f"{video_name}_{split_idx+1}"
            split_path = os.path.join(output_folder + "/sequences", split_name)
            os.makedirs(split_path, exist_ok=True)
            split_gt_file = os.path.join(output_folder + "/annotations", f"{split_name}.txt")
            
            start_idx = split_idx * chunk_size
            end_idx = start_idx + chunk_size
            
            with open(split_gt_file, "w") as f:
                for new_idx, img_file in enumerate(image_files[start_idx:end_idx], start=1):
                    old_frame_idx = int(os.path.splitext(img_file)[0])
                    new_img_name = f"{new_idx:07d}.jpg"
                    old_img_path = os.path.join(video_path, img_file)
                    new_img_path = os.path.join(split_path, new_img_name)
                    
                    shutil.copy(old_img_path, new_img_path)
                    
                    if old_frame_idx in annotations:
                        for line in annotations[old_frame_idx]:
                            parts = line.strip().split(",")
                            parts[0] = str(new_idx)  # Update frame index
                            f.write(",".join(parts) + "\n")
        
        print(f"Processed {video_name} into {num_splits} chunks.")

# Example Usage
chunk_size = 10
new_folder = "VisDrone2019-VID-test-dev"
split_dataset(new_folder+"/sequences", new_folder+"/annotations", new_folder+f"_{chunk_size}", chunk_size=chunk_size)