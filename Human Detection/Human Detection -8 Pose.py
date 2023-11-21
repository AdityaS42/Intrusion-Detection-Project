import cv2
import os
import numpy as np

# Load YOLO model for object detection (if needed)
yolo_net = cv2.dnn.readNet("D:/Python Codes/Library/Yolo/yolov3.weights", "D:/Python Codes/Library/Yolo/yolov3.cfg")

# Load COCO class names (if needed)
with open("D:/Python Codes/Library/Yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Folder containing input videos
input_folder = "D:/Python Codes/Test/Sample Projects/Detection videos"

# Create folders to save detected frames
detected_human_folder = "detected_human_frames"
os.makedirs(detected_human_folder, exist_ok=True)

# Define the region of interest (ROI) coordinates
roi_x, roi_y, roi_width, roi_height = 0, 0, 1280, 520  # Adjust these values for your setup

# Load the pre-trained Caffe-based pose estimation model
proto_file = "D:/Python Codes/Library/openpose-master/openpose-master/models/pose/coco/pose_deploy_linevec.prototxt"
caffemodel_file = "D:/Python Codes/Library/caffe-master/models/bvlc_reference_caffenet/deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(proto_file, caffemodel_file)

# Minimum height for bounding boxes to avoid tracking just the head
min_bbox_height = 100

# Reduce the action confidence threshold
action_confidence_threshold = 0.1

# Initialize a buffer to store the last N frames
buffer_size = 20  # Adjust this based on your preference
frame_buffer = []

# Define a list of action labels corresponding to the model's classes
action_labels = ["Run", "Jump", "Climb", "Stand", "Walk", "Sit"]  # Update with your action labels

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

            # Pose Detection (if a human is detected)
            if bounding_boxes:
                # Use the pre-trained Caffe-based pose estimation model
                input_blob = cv2.dnn.blobFromImage(roi, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
                net.setInput(input_blob)
                output = net.forward()

                # Process pose estimation output
                # (You can display or use the pose information here)
                # Extract keypoints and draw them on the frame

                # For example, you can iterate through the keypoints and draw them:
                for i in range(0, output.shape[1]):
                    confidence = output[0, i, :, 0]
                    h, w = frame.shape[:-1]

                    # Filter keypoints by confidence threshold
                    if confidence > 0.2:  # Adjust the confidence threshold as needed
                        x = int(output[0, i, 0, 0] * w)
                        y = int(output[0, i, 0, 1] * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

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
