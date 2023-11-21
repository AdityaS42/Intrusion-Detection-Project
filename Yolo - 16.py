import cv2
import os
import numpy as np
import tensorflow as tf

# Load YOLO model for human detection
yolo_net = cv2.dnn.readNet("D:/Python Codes/Library/Yolo/yolov3.weights", "D:/Python Codes/Library/Yolo/yolov3.cfg")

# Load COCO class names
with open("D:/Python Codes/Library/Yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load Action Recognition Model 
action_model = tf.keras.models.load_model("D:/Python Codes/Test/Yolo/action_recognition_model_240p.h5")
action_input_shape = (64, 240, 320, 1)  # Adjust this shape as needed

# Folder containing input videos
input_folder = "D:/Python Codes/Test/Sample Projects/Detection videos"

# Create a folder to save the detected frames
output_folder = "detected_frames"
os.makedirs(output_folder, exist_ok=True)

# Create a folder to store the motion images
motion_images_folder = "action_models"
os.makedirs(motion_images_folder, exist_ok=True)

frame_buffer = []  # To store frames for action recognition
frame_buffer_size = 64  # Number of frames in the buffer

for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        video_path = os.path.join(input_folder, filename)
        cap = cv2.VideoCapture(video_path)

        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Define the desired frame size
            target_frame_size = (640, 480)  # You can adjust this size as needed
            frame = cv2.resize(frame, target_frame_size)

            # Adjust the Region of Interest (ROI) based on the resized frame
            original_height, original_width = frame.shape[:2]
            roi_x, roi_y, roi_width, roi_height = int(0.25 * original_width), int(0.1 * original_height), int(0.5 * original_width), int(0.8 * original_height)

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
                        center_x = int(detection[0] * roi.shape[1])
                        center_y = int(detection[1] * roi.shape[0])
                        w = int(detection[2] * roi.shape[1])
                        h = int(detection[3] * roi.shape[0])
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # Adjust the coordinates to the original frame
                        x += roi_x
                        y += roi_y

                        # Crop and resize the detected human region to 240x320 for action recognition
                        human_region = frame[y:y+h, x:x+w]
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

                            # Draw bounding box and label with action class
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = f"Action: {action_class}"
                            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Save the motion image to the "action_models" folder
                            motion_image_path = os.path.join(motion_images_folder, f"motion_image_{frame_number:04d}.jpg")
                            cv2.imwrite(motion_image_path, frame)

                            # Clear the frame buffer
                            frame_buffer.clear()

            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_number += 1

        cap.release()

cv2.destroyAllWindows()
