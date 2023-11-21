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

# Create a folder to save the detected frames
output_folder = "detected_frames"
os.makedirs(output_folder, exist_ok=True)

# Define the region of interest (ROI) coordinates
roi_x, roi_y, roi_width, roi_height = 300, 100, 400, 300

# Create a VideoCapture object for the input video
for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        video_path = os.path.join(input_folder, filename)
        cap = cv2.VideoCapture(video_path)

        frame_number = 0
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

    detected_person_bbox = None  # Store the bounding box of the detected person

    # Process detections (same as before)
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

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                detected_person_bbox = (x, y, x + w, y + h)  # Store the bounding box

                # Save the detected frame
                detected_frame_path = os.path.join(output_folder, f"{frame_number:04d}.jpg")
                cv2.imwrite(detected_frame_path, frame)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_number += 1

    # Convert the current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None and detected_person_bbox is not None:
        # Extract the ROI for motion detection based on the detected person's bounding box
        x1, y1, x2, y2 = detected_person_bbox
        person_roi = gray[y1:y2, x1:x2]

        # Check if the person ROI is not empty
        if person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
            # Calculate optical flow within the person ROI
            flow = cv2.calcOpticalFlowFarneback(prev_frame, person_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Calculate motion magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Threshold the magnitude to detect motion (adjust this threshold)
            motion_mask = magnitude > 1  # You can adjust this threshold

            # Create a color representation of magnitude
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            # Convert HSV to BGR
            motion_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Apply the motion mask to highlight motion areas
            motion_frame[motion_mask] = [0, 0, 255]  # Highlight motion areas in red

            cv2.imshow("Motion Detection", motion_frame)  # Display the motion detection window

    prev_frame = gray

cap.release()
cv2.destroyAllWindows()
