import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Load a pre-trained model
model = YOLO()
names = model.names

# classes ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# Open the Webcam
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error opening video"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Blur ratio
blur_ratio = 50
additional_blur_ratio = 100  # Ratio for processing additional blur effect

# Video writer
video_writer = cv2.VideoWriter("test_video_1.avi", cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))

# Class ID to print on screen
display_class_ids = [0, 1, 2, 3]  # Hardhat, Mask, NO-Hardhat, NO-Mask

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=names)

    if boxes is not None:
        # Partition blurring and printing on screen
        for box, cls in zip(boxes, clss):
            if int(cls) in display_class_ids:  # Show it only class IDs
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])
                obj = im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))

                # Processing additional blur effect
                additional_blur_height = int((box[3] - box[1]) * 0.9)  # Add n% of boundary box heght
                additional_blur_obj = cv2.blur(im0[int(box[3] - additional_blur_height): int(box[3]), int(box[0]): int(box[2])], (additional_blur_ratio, additional_blur_ratio))
                im0[int(box[3] - additional_blur_height): int(box[3]), int(box[0]): int(box[2])] = additional_blur_obj

                im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])] = blur_obj

    cv2.imshow("ultralytics", im0)
    video_writer.write(im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
