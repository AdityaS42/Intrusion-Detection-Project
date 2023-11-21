import cv2
import os

# Load YOLO model
yolo_net = cv2.dnn.readNet("D:/Python Codes/Yolo/yolov3.weights", "D:/Python Codes/Yolo/yolov3.cfg")

# Load COCO class names
with open("D:/Python Codes/Yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Path to the folder containing images
image_folder = 'path_to_folder_with_images'

# Process each image file in the folder
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        frame = cv2.imread(image_path)

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
        if cv2.waitKey(0) & 0xFF == ord("q"):  # Wait for any key press to move to the next image
            break

cv2.destroyAllWindows()
