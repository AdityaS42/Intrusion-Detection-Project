import cv2
import os

# Define the source and target resolutions
source_resolution = (320, 240)
target_resolution = (1280, 720)

# Directory containing your original training data
train_dir = "D:/Python Codes/Testing"

# Directory to save resized training data
resized_train_dir = "D:/Python Codes/Resized_Testing"

# Create the target directory if it doesn't exist
os.makedirs(resized_train_dir, exist_ok=True)

# Loop through the original training data
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    resized_class_dir = os.path.join(resized_train_dir, class_name)
    os.makedirs(resized_class_dir, exist_ok=True)

    for video_name in os.listdir(class_dir):
        video_path = os.path.join(class_dir, video_name)
        resized_video_path = os.path.join(resized_class_dir, video_name)

        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(resized_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, target_resolution)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, target_resolution)
            out.write(resized_frame)

        cap.release()
        out.release()

print("Resizing completed.")
