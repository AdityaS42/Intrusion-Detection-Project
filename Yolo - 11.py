import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Load YOLO model
yolo_net = cv2.dnn.readNet("D:/Python Codes/Library/Yolo/yolov3.weights", "D:/Python Codes/Library/Yolo/yolov3.cfg")

# Load COCO class names
with open("D:/Python Codes/Library/Yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Folder containing input videos
input_folder = "D:/Python Codes/Test/Sample Projects/Detection videos"

# Create a folder to save the detected frames
output_folder = "detected_frames"
os.makedirs(output_folder, exist_ok=True)

# Create a folder to save recognized action frames
recognized_actions_folder = "recognized_actions"
os.makedirs(recognized_actions_folder, exist_ok=True)

# Define the region of interest (ROI) coordinates
roi_x, roi_y, roi_width, roi_height = 300, 50, 480, 320  # Adjusted ROI

# Function to preprocess frames for action recognition
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (input_shape[0], input_shape[1]))  # Define input_shape here
        frames.append(frame)

    frames = np.array(frames)
    frames = frames / 255.0  # Normalize pixel values
    return frames

# Function to load and preprocess frames for action recognition
def load_action_recognition_data(data_dir, max_frames):
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

# Load training and validation data with padding for action recognition
max_frames = 64  # Adjust the maximum number of frames as needed
train_dir = "D:/Python Codes/Training"
val_dir = "D:/Python Codes/Validation"
test_dir = "D:/Python Codes/Testing"

input_shape = (64, 64, 64, 3)  # Define input_shape here
x_train, y_train = load_action_recognition_data(train_dir, max_frames)
x_val, y_val = load_action_recognition_data(val_dir, max_frames)

# Define the 3D CNN model for action recognition
num_classes = len(os.listdir(train_dir))  # Number of action classes

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

# Train the model with reduced batch size
batch_size = 2  # Reduce batch size for lower RAM usage
epochs = 7
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_val, y_val))

# Save the trained model
model.save('action_recognition_model.h5')

# Function to label recognized actions
def label_recognized_action(frame, action_label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'Action: {action_label}'
    org = (10, 30)
    font_scale = 1
    color = (0, 0, 255)  # Red color for text
    thickness = 2
    cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

# Load detected frames from the YOLO output
def load_detected_frames(output_folder, max_frames):
    detected_frames = []

    for filename in os.listdir(output_folder):
        if filename.endswith(".jpg"):
            frame = cv2.imread(os.path.join(output_folder, filename))
            detected_frames.append(frame)

    return np.array(detected_frames)

x_test = load_detected_frames(output_folder, max_frames)

# Predict actions on the detected frames
predicted_probabilities = model.predict(x_test)
predicted_actions = np.argmax(predicted_probabilities, axis=1)


# Iterate through detected frames, label them, and save to recognized_actions_folder
for i, frame in enumerate(x_test):
    recognized_frame = frame.copy()
    action_label = class_names[predicted_actions[i]]  # Get action label
    label_recognized_action(recognized_frame, action_label)  # Label the frame
    recognized_frame_path = os.path.join(recognized_actions_folder, f"recognized_action_{i:04d}.jpg")
    cv2.imwrite(recognized_frame_path, recognized_frame)

# Display and save the labeled recognized frames
for i, frame in enumerate(x_test):
    action_label = class_names[predicted_actions[i]]  # Get action label
    cv2.imshow(f"Recognized Action: {action_label}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Evaluate the action recognition model on the detected frames
y_test = label_binarizer.transform(y_test)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
