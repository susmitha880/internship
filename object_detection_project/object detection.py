import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture("http://172.20.206.165:8080/video")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to open webcam")
        break

    # Run detection
    results = model(frame)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Object Detection", annotated_frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 