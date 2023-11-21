import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Load YOLO model for object detection
yolo_net = cv2.dnn.readNet("D:/Python Codes/Library/Yolo/yolov3.weights", "D:/Python Codes/Library/Yolo/yolov3.cfg")

# Load COCO class names
with open("D:/Python Codes/Library/Yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Define the paths to your data directories
train_dir = "D:/Python Codes/Training"
val_dir = "D:/Python Codes/Validation"
test_dir = "D:/Python Codes/Testing"

# Define hyperparameters for action recognition
batch_size = 2  # Reduce batch size for lower RAM usage
epochs = 7
input_shape = (64, 144, 256, 1)  # Adjust the input shape for grayscale (1 channel) 320x240 data
num_classes = len(os.listdir(train_dir))  # Number of action classes

# Data preprocessing function for 320x240 grayscale frames
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frame = cv2.resize(frame, (input_shape[2], input_shape[1]))  # Resize to 320x240
        frames.append(frame)

    frames = np.array(frames, dtype=np.float32)  # Use float32 data type
    frames = frames / 255.0  # Normalize pixel values
    return frames[..., np.newaxis]  # Add a single channel for grayscale

# Load and preprocess data with padding for action recognition
def load_data(data_dir, max_frames):
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
        frames = preprocess_video(video_path)
        # Pad or truncate frames to the specified maximum length (max_frames)
        if len(frames) > max_frames:
            frames = frames[:max_frames]
        else:
            padding = [(0, max_frames - len(frames))] + [(0, 0)] * 3  # Pad with zeros
            frames = np.pad(frames, padding, mode='constant')

        videos.append(frames)

    return np.array(videos), np.array(label_ids)

# Load training and validation data with padding
max_frames = 64  # Adjust the maximum number of frames as needed
x_train, y_train = load_data(train_dir, max_frames)
x_val, y_val = load_data(val_dir, max_frames)

# Define the 3D CNN model for 320x240 grayscale frames
model = models.Sequential([
    layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling3D((1, 2, 2)),  # Adjust pooling for 320x240
    layers.Conv3D(64, (3, 3, 3), activation='relu'),
    layers.MaxPooling3D((1, 2, 2)),  # Adjust pooling for 320x240
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the action recognition model with reduced batch size
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_val, y_val))

# Load YOLO model for object detection
yolo_net = cv2.dnn.readNet("D:/Python Codes/Library/Yolo/yolov3.weights", "D:/Python Codes/Library/Yolo/yolov3.cfg")

# Folder containing input videos
input_folder = "D:/Python Codes/Test/Sample Projects/Detection videos"

# Create a folder to save the detected frames
output_folder = "detected_frames"
os.makedirs(output_folder, exist_ok=True)

# Define the region of interest (ROI) coordinates
roi_x, roi_y, roi_width, roi_height = 300, 50, 480, 320  # Adjusted ROI

# Initialize a list to accumulate frames for action recognition
frames_accumulated = []

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
            blob = cv2.dnn.blobFromImage(roi, 0.00392, (416, 416), swapRB=True, crop=False)
            yolo_net.setInput(blob)

            # Get detection results
            layer_names = yolo_net.getUnconnectedOutLayersNames()
            detections = yolo_net.forward(layer_names)

            # Process detections
            for out in detections:
                for detection in out:
                    scores = detection[5:]
                    class_id = scores.argmax()
                    confidence = scores[class_id]
                    if confidence > 0.5 and classes[class_id] == "person":
                        # When a person is detected, add the current frame to the accumulated frames list
                        current_frame = frame.copy()
                        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                        current_frame = cv2.resize(current_frame, (input_shape[2], input_shape[1]))
                        current_frame = current_frame / 255.0
                        current_frame = current_frame[..., np.newaxis]

                        frames_accumulated.append(current_frame)

                        # If we have accumulated 64 frames, process them with the action recognition model
                        if len(frames_accumulated) == max_frames:
                            # Predict the action class using the action recognition model
                            action_class = model.predict(np.array([frames_accumulated]))
                            action_label = np.argmax(action_class)

                            # Clear the accumulated frames list
                            frames_accumulated = []

                        # Continue with drawing bounding boxes and annotations as before
                        center_x = int(detection[0] * roi.shape[1])
                        center_y = int(detection[1] * roi.shape[0])
                        w = int(detection[2] * roi.shape[1])
                        h = int(detection[3] * roi.shape[0])
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # Adjust the coordinates to the original frame
                        x += roi_x
                        y += roi_y

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Action: {action_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Save the detected frame
                        detected_frame_path = os.path.join(output_folder, f"{filename[:-4]}_frame_{frame_number:04d}.jpg")
                        cv2.imwrite(detected_frame_path, frame)

            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_number += 1

        cap.release()

cv2.destroyAllWindows()
