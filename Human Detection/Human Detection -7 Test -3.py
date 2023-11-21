import cv2
import os
import numpy as np

# Load YOLO model
yolo_net = cv2.dnn.readNet("D:/Python Codes/Library/Yolo/yolov3.weights", "D:/Python Codes/Library/Yolo/yolov3.cfg")

# Load COCO class names
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
roi_x, roi_y, roi_width, roi_height = 0, 0, 1280, 480  # Adjust these values for your setup

for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        video_path = os.path.join(input_folder, filename)
        cap = cv2.VideoCapture(video_path)

        frame_number = 0

        # Initialize previous frame for motion detection
        prev_frame = None

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

            # Flag to check if a human is detected
            human_detected = False

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

                        # Ensure the detected object is within the ROI
                        if y + h > roi_y + roi_height:
                            continue

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Save the detected human frame
                        detected_human_frame_path = os.path.join(detected_human_folder, f"{filename[:-4]}_frame_{frame_number:04d}.jpg")
                        cv2.imwrite(detected_human_frame_path, frame)

                        human_detected = True

            # Motion Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_frame is None:
                prev_frame = gray
                continue

            # Extract optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            x_flow, y_flow = flow[:, :, 0], flow[:, :, 1]

            # Flag to check if there is motion
            motion_detected = np.any(np.abs(x_flow) > 1) or np.any(np.abs(y_flow) > 1)

            if motion_detected:
                # Save the detected motion frame
                detected_motion_frame_path = os.path.join(detected_motion_folder, f"{filename[:-4]}_frame_{frame_number:04d}.jpg")
                cv2.imwrite(detected_motion_frame_path, frame)

            frame_number += 1
            prev_frame = gray

        cap.release()

cv2.destroyAllWindows()
