import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Define paths
object_detection_model_path = "D:/Python Codes/Library/Yolo/yolov3.weights"
object_detection_config_path = "D:/Python Codes/Library/Yolo/yolov3.cfg"
coco_names_path = "D:/Python Codes/Library/Yolo/coco.names"
input_folder = "D:/Python Codes/Test/Sample Projects/Detection videos"
train_dir = "D:/Python Codes/Training"
val_dir = "D:/Python Codes/Validation"
output_folder = "detected_frames"
os.makedirs(output_folder, exist_ok=True)

# Load YOLO object detection model
yolo_net = cv2.dnn.readNet(object_detection_model_path, object_detection_config_path)

# Load COCO class names
with open(coco_names_path, "r") as f:
    classes = f.read().strip().split("\n")

# Define the region of interest (ROI) coordinates
roi_x, roi_y, roi_width, roi_height = 400, 150, 480, 360  # Adjusted for 1280x720 videos

# Define hyperparameters for action recognition
batch_size = 8
epochs = 10
input_shape_action = (64, 64, 64, 3)  # Adjust the input shape as needed for action recognition

# Set max_frames to 120
max_frames = 120

# Function to preprocess video frames for action recognition
def preprocess_video_action(video_path, max_frames, input_shape):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break

        frame = cv2.resize(frame, (input_shape[0], input_shape[1]))
        frame = frame / 255.0  # Normalize pixel values
        frames.append(frame)

    frames = np.array(frames)
    frames = np.pad(frames, [(0, max_frames - len(frames)), (0, 0), (0, 0), (0, 0)], mode='constant')
    return frames

# Function to load and preprocess data with padding for action recognition
def load_data_action(data_dir, max_frames, input_shape):
    video_paths = []
    labels = []

    class_names = sorted(os.listdir(data_dir))
    class_to_label = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for video_name in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_name)
            video_paths.append(video_path)
            labels.append(class_name)

    videos = []
    label_ids = [class_to_label[label] for label in labels]

    for video_path in video_paths:
        frames = preprocess_video_action(video_path, max_frames, input_shape)
        videos.append(frames)

    return np.array(videos), np.array(label_ids)

# Load training and validation data with padding for action recognition
x_train_action, y_train_action = load_data_action(train_dir, max_frames, input_shape_action)
x_val_action, y_val_action = load_data_action(val_dir, max_frames, input_shape_action)

# Load the pre-trained action recognition model
action_recognition_model = models.load_model('D:/Python Codes/Test/Yolo/action_recognition_model.h5')

# Configure YOLO parameters for better human detection
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Iterate through input videos
for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        video_path = os.path.join(input_folder, filename)
        cap = cv2.VideoCapture(video_path)

        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Crop the frame to the specified ROI
            roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

            # Prepare input blob for YOLO model using the cropped ROI
            blob = cv2.dnn.blobFromImage(roi, 1/255.0, (416, 416), swapRB=True, crop=False)
            yolo_net.setInput(blob)

            # Get detection results
            layer_names = yolo_net.getUnconnectedOutLayersNames()
            detections = yolo_net.forward(layer_names)

            # Process detections
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5 and classes[class_id] == "person":
                        center_x = int(obj[0] * roi.shape[1])
                        center_y = int(obj[1] * roi.shape[0])
                        w = int(obj[2] * roi.shape[1])
                        h = int(obj[3] * roi.shape[0])
                        x = int(center_x - w / 2) + roi_x
                        y = int(center_y - h / 2) + roi_y

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Save the detected frame
                        detected_frame_path = os.path.join(output_folder, f"{filename[:-4]}_frame_{frame_number:04d}.jpg")
                        cv2.imwrite(detected_frame_path, frame)

                        # Perform action recognition on the detected frame
                        detected_frame = cv2.resize(roi, (64, 64))
                        detected_frame = detected_frame / 255.0
                        detected_frame = np.expand_dims(detected_frame, axis=0)
                        detected_frame = np.repeat(detected_frame, 64, axis=0)  # Repeat to match (batch_size, 64, 64, 64, 3)
                        predicted_class = np.argmax(action_recognition_model.predict(detected_frame))
                        print(f"Action prediction for frame {frame_number}: Class {predicted_class}")

            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_number += 1

        cap.release()

cv2.destroyAllWindows()
