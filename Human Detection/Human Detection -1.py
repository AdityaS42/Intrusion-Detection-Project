import cv2
import os

# Load the HOG detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Path to the folder containing video files
video_folder = 'C:/Users/ADITYA SURVE/Downloads/Detection videos'

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

            # Draw rectangles around detected humans
            for (x, y, w, h) in detected:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Human Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()

