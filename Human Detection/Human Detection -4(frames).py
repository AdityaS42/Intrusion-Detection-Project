import cv2
import os

# Load YOLO model
yolo_net = cv2.dnn.readNet("D:/Python Codes/Yolo/yolov3.weights", "D:/Python Codes/Yolo/yolov3.cfg")

# Load COCO class names
with open("D:/Python Codes/Yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load the video
video_path = "C:/Users/ADITYA SURVE/Downloads/Detection videos/Timeline 2.mp4"
cap = cv2.VideoCapture(video_path)

# Create a folder to save the extracted frames
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save the frame as an image
    output_path = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
    cv2.imwrite(output_path, frame)

    # Prepare input blob for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
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
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()
