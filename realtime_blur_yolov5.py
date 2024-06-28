import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import torch

from pathlib import Path

#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load a pre-trained model
model = torch.hub.load()
names = model.names

# classes ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# Open the Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Can't open the Webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't get frame.")
        break
    
    # Pass the frame to YOLOv5 model, perfroming object detection
    results = model(frame)

    # Draw results in image
    annotated_frame = results.render()[0]

    # Show results in screen
    cv2.imshow('YOLOv5 Webcam', annotated_frame)

    # Quit : 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close all
cap.release()
cv2.destroyAllWindows()
