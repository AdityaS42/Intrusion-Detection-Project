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
batch_size = 1  # Process one video at a time
epochs = 10
input_shape = (720, 1280, 3)  # Fixed resolution (720x1280)
num_classes = len(os.listdir(train_dir))  # Number of action classes

# Data preprocessing function
def preprocess_video(video_path, target_height, target_width):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)

    frames = np.array(frames, dtype=np.float32) / 255.0  # Convert to float32 and normalize pixel values
    return frames

# Load and preprocess data without padding
def load_data(data_dir):
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
        frames = preprocess_video(video_path, input_shape[0], input_shape[1])
        videos.append(frames)

    return np.array(videos, dtype=np.float32), np.array(label_ids, dtype=np.int32)

# Load training and validation data without padding
x_train, y_train = load_data(train_dir)
x_val, y_val = load_data(val_dir)

# Define the 3D CNN model
model = models.Sequential([
    layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(None, input_shape[0], input_shape[1], input_shape[2])),
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
model.save('action_recognition_model_2.h5')

# Evaluate the model on the test set
x_test, y_test = load_data(test_dir)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
