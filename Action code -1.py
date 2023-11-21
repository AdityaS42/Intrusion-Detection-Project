import cv2
import os
import numpy as np
import tensorflow as tf

# Folder containing detected frames
detected_frames_folder = "detected_frames"

# Load the coordinates of the bounding boxes from the saved file
bounding_boxes = []
with open("D:/Python Codes/Test/Yolo/bounding_boxes.txt", "r") as bb_file:
    for line in bb_file:
        x, y, w, h = map(int, line.strip().split())
        bounding_boxes.append((x, y, w, h))

# Load Action Recognition Model
action_model = tf.keras.models.load_model("D:/Python Codes/Test/Yolo/action_recognition_model_240p.h5")
action_input_shape = (64, 240, 320, 1)  # Adjust this shape as needed

# Create a folder to store the motion images
motion_images_folder = "action_models"
os.makedirs(motion_images_folder, exist_ok=True)

frame_buffer = []  # To store frames for action recognition
frame_buffer_size = 64  # Number of frames in the buffer

for filename in os.listdir(detected_frames_folder):
    if filename.endswith(".jpg"):
        frame_path = os.path.join(detected_frames_folder, filename)
        frame = cv2.imread(frame_path)

        # Get bounding box coordinates
        if bounding_boxes:
            x, y, w, h = bounding_boxes.pop(0)  # Get the first bounding box

            # Crop and resize the detected human region to 240x320 for action recognition
            human_region = frame[y:y + h, x:x + w]
            human_region = cv2.cvtColor(human_region, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            human_region = cv2.resize(human_region, (action_input_shape[2], action_input_shape[1]))
            human_region = np.expand_dims(human_region, axis=-1)  # Add a single channel for grayscale
            human_region = human_region / 255.0  # Normalize pixel values

            frame_buffer.append(human_region)

            # Check if the frame buffer is full
            if len(frame_buffer) == frame_buffer_size:
                # Predict action using the action recognition model
                frame_buffer_batch = np.array(frame_buffer)[np.newaxis, ...]
                action_prediction = action_model.predict(frame_buffer_batch)
                action_class = np.argmax(action_prediction)

                # Draw bounding box and label with action class on the original frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"Action: {action_class}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save the motion image to the "action_models" folder
                motion_image_path = os.path.join(motion_images_folder, f"motion_image_{filename[:-4]}.jpg")
                cv2.imwrite(motion_image_path, frame)

                # Clear the frame buffer
                frame_buffer.clear()

        cv2.imshow("Action Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
