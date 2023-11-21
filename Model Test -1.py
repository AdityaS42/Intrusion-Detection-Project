import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Define the paths to your data directories
train_dir = "D:/Python Codes/Training"
val_dir = "D:/Python Codes/Validation"
test_dir = "D:/Python Codes/Testing"

# Define hyperparameters
batch_size = 32
epochs = 10
input_shape = (64, 64, 64, 3)  # Adjust the input shape as needed

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

    frames = np.array(frames)
    frames = frames / 255.0  # Normalize pixel values
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

    return np.array(videos), np.array(label_ids)

# Load training and validation data with padding
max_frames = 64  # Adjust the maximum number of frames as needed
x_train, y_train = load_data(train_dir, max_frames)
x_val, y_val = load_data(val_dir, max_frames)

# Load test data with padding
x_test, y_test = load_data(test_dir, max_frames)

# One-hot encode the labels
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_val = label_binarizer.transform(y_val)
y_test = label_binarizer.transform(y_test)

# Define the class names
class_names = sorted(os.listdir(train_dir))

# Define the 3D CNN model
model = models.Sequential([
    layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Conv3D(64, (3, 3, 3), activation='relu'),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  # Output layer matches the number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use categorical cross-entropy for one-hot encoded labels
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_val, y_val))

# Save the trained model
model.save('action_recognition_model.h5')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
