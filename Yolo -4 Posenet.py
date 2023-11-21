import cv2
import os
import numpy as np
import tensorflow as tf
import posenet

# Load YOLO model for object detection
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

# Load PoseNet model
model = posenet.load_model("D:/Python Codes/Library/tfjs-models-master/posenet/src/posenet_model.ts")

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

            # Get detection results for persons
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

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Save the detected frame
                        detected_frame_path = os.path.join(output_folder, f"{filename[:-4]}_frame_{frame_number:04d}.jpg")
                        cv2.imwrite(detected_frame_path, frame)

                        # Perform human pose estimation using PoseNet
                        person_roi = frame[y:y + h, x:x + w]
                        input_image, output_scale = posenet.pad_to_input_size(person_roi, model.stride)
                        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                        input_image = np.expand_dims(input_image, axis=0)

                        # Run PoseNet inference
                        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model.predict(input_image)

                        # Process the PoseNet results and draw keypoints on the frame
                        keypoints, _ = posenet.decode_multiple_poses(
                            heatmaps_result.squeeze(axis=0),
                            offsets_result.squeeze(axis=0),
                            displacement_fwd_result.squeeze(axis=0),
                            displacement_bwd_result.squeeze(axis=0),
                            output_stride=model.output_stride,
                            max_pose_detections=1,
                            min_pose_score=0.15)

                        frame_with_keypoints = posenet.draw_skel_and_kp(
                            person_roi, keypoints, model.output_stride)

                        # Place the frame with keypoints back in the original frame
                        frame[y:y + h, x:x + w] = frame_with_keypoints

            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_number += 1

        cap.release()

cv2.destroyAllWindows()
