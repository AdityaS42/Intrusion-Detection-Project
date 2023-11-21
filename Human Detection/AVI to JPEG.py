import os
import cv2
import numpy as np

# Specify the path to your video files and where you want to save the JPEG frames
video_dir = 'D:/Python Codes/Library/Train/walk'  # Change this to your video directory
output_dir = 'D:/Python Codes/Library/Train_frames/Train_walk'  # Change this to the output directory

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to convert a video into JPEG frames
def video_to_frames(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as a JPEG file
        frame_filename = os.path.join(output_path, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()

# Iterate through video files in the specified directory and convert them
for video_filename in os.listdir(video_dir):
    if video_filename.endswith('.avi'):
        video_path = os.path.join(video_dir, video_filename)
        video_output_dir = os.path.join(output_dir, video_filename[:-4])  # Create a subdirectory for frames
        os.makedirs(video_output_dir, exist_ok=True)
        
        video_to_frames(video_path, video_output_dir)

# Now, you can use 'video_output_dir' as the path for your ImageDataGenerator to train on JPEG frames.
