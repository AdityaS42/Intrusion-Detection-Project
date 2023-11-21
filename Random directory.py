import os
import random
import shutil

dataset_dir = "D:/Python Codes/Training Datasheet"
train_dir = "D:/Python Codes/Training"
val_dir = "D:/Python Codes/Validation"
test_dir = "D:/Python Codes/Testing"
split_ratio = [0.7, 0.15, 0.15]  # Train, Validation, Test

random.seed(42)  # Set a seed for reproducibility
for class_dir in os.listdir(dataset_dir):
    if os.path.isdir(os.path.join(dataset_dir, class_dir)):
        video_files = os.listdir(os.path.join(dataset_dir, class_dir))
        random.shuffle(video_files)
        num_videos = len(video_files)
        train_split = int(num_videos * split_ratio[0])
        val_split = int(num_videos * split_ratio[1])

        for i, video_file in enumerate(video_files):
            src_path = os.path.join(dataset_dir, class_dir, video_file)
            if i < train_split:
                dst_dir = train_dir
            elif i < train_split + val_split:
                dst_dir = val_dir
            else:
                dst_dir = test_dir
            dst_path = os.path.join(dst_dir, class_dir, video_file)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
