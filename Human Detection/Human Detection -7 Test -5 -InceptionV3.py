import cv2
import os
import numpy as np
import tensorflow as tf

# Load YOLO model for object detection (if needed)
yolo_net = cv2.dnn.readNet("D:/Python Codes/Library/Yolo/yolov3.weights", "D:/Python Codes/Library/Yolo/yolov3.cfg")

# Load COCO class names (if needed)
with open("D:/Python Codes/Library/Yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Folder containing input videos
input_folder = "D:/Python Codes/Test/Sample Projects/Detection videos"

# Create folders to save detected frames
detected_human_folder = "detected_human_frames"
detected_motion_folder = "detected_motion_frames"
os.makedirs(detected_human_folder, exist_ok=True)
os.makedirs(detected_motion_folder, exist_ok=True)

# Define the region of interest (ROI) coordinates
roi_x, roi_y, roi_width, roi_height = 0, 0, 1280, 520  # Adjust these values for your setup

# Define the jump detection region just above the base ROI
jump_roi_y = roi_y - 10  # A few pixels above the base ROI

# Load the pre-trained InceptionV3 model without weights and include_top=False
base_model = tf.keras.applications.InceptionV3(weights=None, include_top=False)

# Add your custom classification head
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)  # You can adjust the number of units as needed
predictions = tf.keras.layers.Dense(6, activation='softmax')(x)  # 6 is the number of your custom classes

# Create the final model
i3d_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Define a list of action labels corresponding to the model's classes
action_labels = ["Run", "Jump", "Climb", "Stand", "Walk", "Sit"]  # Update with your action labels

# Minimum height for bounding boxes to avoid tracking just the head
min_bbox_height = 100

# Reduce the action confidence threshold
action_confidence_threshold = 0.1

# Initialize a buffer to store the last N frames
buffer_size = 20  # Adjust this based on your preference
frame_buffer = []

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

            # Initialize action label and bounding boxes
            current_action = "Unknown"
            bounding_boxes = []

            # Process detections
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = scores.argmax()
                    confidence = scores[class_id]
                    if confidence > 0.5 and classes[class_id] == "person":
                        # Human detection logic here
                        human_detected = True

                        # Extract bounding box coordinates
                        center_x, center_y, width, height = map(int, obj[0:4] * np.array([roi_width, roi_height, roi_width, roi_height]))
                        top_left_x = center_x - width // 2
                        top_left_y = center_y - height // 2

                        # Check if the height of the bounding box is above the minimum threshold
                        if height >= min_bbox_height:
                            bounding_boxes.append((top_left_x, top_left_y, top_left_x + width, top_left_y + height))

            # Draw bounding boxes
            for box in bounding_boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # Action Recognition (if a human is detected)
            if bounding_boxes:
                # Resize the frame to 299x299 for action recognition
                frame_for_recognition = cv2.resize(roi, (299, 299))
                frame_for_recognition = tf.keras.applications.inception_v3.preprocess_input(frame_for_recognition)
                frame_for_recognition = np.expand_dims(frame_for_recognition, axis=0)

                # Perform action recognition inference
                predictions = i3d_model.predict(frame_for_recognition)
                predicted_class = np.argmax(predictions)
                action_confidence = np.max(predictions)

                # Debug: Print predicted action and confidence
                print(f"Predicted Action: {action_labels[predicted_class]}, Confidence: {action_confidence}")

                # Check if action confidence is above the threshold
                if action_confidence >= action_confidence_threshold:
                    current_action = action_labels[predicted_class]

            # Append the current action to the buffer
            frame_buffer.append(current_action)

            # Ensure the buffer size does not exceed the specified size
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)

            # Compute the action prediction based on the majority class in the buffer
            action_prediction = max(set(frame_buffer), key=frame_buffer.count)

            # Display the current action label on the frame
            cv2.putText(frame, f"Action: {action_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the processed frame
            cv2.imshow("Processed Frame", frame)
            cv2.waitKey(1)  # Add a delay to control the frame display speed

            # Save the frame with bounding boxes
            frame_number += 1
            detected_frame_path = os.path.join(detected_human_folder, f"{filename[:-4]}_frame_{frame_number:04d}.jpg")
            cv2.imwrite(detected_frame_path, frame)

        cap.release()

cv2.destroyAllWindows()
