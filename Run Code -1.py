import subprocess

# Step 1: Run the human detection code
human_detection_script = "D:/Python Codes/Test/Yolo/Yolo -1 (Boundary boxes).py"  # Replace with the actual filename of your human detection code
subprocess.run(["python", human_detection_script])

# Step 2: Run the action recognition code
action_recognition_script = "D:/Python Codes/Test/Yolo/Action code -1.py"  # Replace with the actual filename of your action recognition code
subprocess.run(["python", action_recognition_script])
