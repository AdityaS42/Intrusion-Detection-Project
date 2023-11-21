import cv2
import os

# Load the HOG detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Path to the folder containing video files
video_folder = 'C:/Users/ADITYA SURVE/Downloads/Detection videos'

# Specify absolute paths to the PoseNet model files
prototxt_path = 'D:/Python Codes/openpose-master/openpose-master/models/pose/coco'
caffemodel_path = 'path_to_pose_iter_440000.caffemodel'

# Load PoseNet model from OpenCV
net = cv2.dnn.readNetFromTensorflow(prototxt_path, caffemodel_path)

# Process each video file in the folder
for filename in os.listdir(video_folder):
    if filename.endswith('.mp4'):
        video_path = os.path.join(video_folder, filename)
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect humans in the frame
            detected, _ = hog.detectMultiScale(frame)

            for (x, y, w, h) in detected:
                # Prepare input blob for PoseNet model
                input_blob = cv2.dnn.blobFromImage(frame[y:y+h, x:x+w], 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)

                # Set the input to the PoseNet model
                net.setInput(input_blob)
                output = net.forward()

                # Process output and draw skeleton (stick figure)
                # Replace this with your code for processing output and drawing skeleton

                # Draw rectangle around detected human
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Human Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
