import tensorflow_datasets as tfds
print(tfds.__version__)
# 載入資料集（預設會使用最新版 2.1.0）
dataset, info = tfds.load('davis', with_info=True, split=['train', 'validation'])
train_ds, val_ds = dataset

# 查看資料集資訊
print(info)
for sample in train_ds.take(1):
    print("Video name:", sample['metadata']['video_name'].numpy())
    print("Number of frames:", sample['metadata']['num_frames'].numpy())

    frames = sample['video']['frames']
    masks = sample['video']['segmentations']

    print("Frames shape:", frames.shape)       # (num_frames, height, width, 3)
    print("Segmentations shape:", masks.shape) # (num_frames, height, width, 1)
    print("sample['video']:", sample['video'])