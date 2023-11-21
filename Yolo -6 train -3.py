import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Define the paths to your data directories
train_dir = "D:/Python Codes/Training"
val_dir = "D:/Python Codes/Validation"
test_dir = "D:/Python Codes/Testing"

# Define hyperparameters
batch_size = 8
epochs = 10
input_shape = (64, 320, 240, 3)  # Adjust the input shape as needed
num_classes = len(os.listdir(train_dir))  # Number of action classes

# Data preprocessing function
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (input_shape[0], input_shape[1]))
        frames.append(frame)

    frames = np.array(frames, dtype=np.float32) / 255.0  # Convert to float32 and normalize pixel values
    return frames

# Load and preprocess data with padding
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

    return np.array(videos, dtype=np.float32), np.array(label_ids)

# Load training and validation data with padding
max_frames = 64  # Adjust the maximum number of frames as needed
x_train, y_train = load_data(train_dir, max_frames)
x_val, y_val = load_data(val_dir, max_frames)


# Define the 3D CNN model
model = models.Sequential([
    layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Conv3D(64, (3, 3, 3), activation='relu'),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_val, y_val))

# Save the trained model
model.save('action_recognition_model_3.h5')

# Evaluate the model on the test set
x_test, y_test = load_data(test_dir, max_frames)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
