import cv2
import imageio

# Load YOLO model with CUDA backend and target
yolo_net = cv2.dnn.readNet("D:/Python Codes/Yolo/yolov3.weights", "D:/Python Codes/Yolo/yolov3.cfg")
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load COCO class names
with open("D:/Python Codes/Yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load the video
cap = cv2.VideoCapture("C:/Users/ADITYA SURVE/Downloads/Detection videos/Timeline 2.mp4")

# Define output video writer using imageio
output_path = "output.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = imageio.get_writer(output_path, fps=fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare input blob for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)

    # Get detection results
    layer_names = yolo_net.getUnconnectedOutLayersNames()
    detections = yolo_net.forward(layer_names)

    # Process detections and draw bounding boxes
    for out in detections:
        for detection in out:
            # Process detections and draw bounding boxes
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

    # Write the processed frame to the output video
    out.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for imageio

# Release the video capture and writer
cap.release()
out.close()
