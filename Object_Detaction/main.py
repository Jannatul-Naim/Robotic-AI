from ultralytics import YOLO
import cv2

# Load YOLOv9 model
model = YOLO("yolov9s.pt")   # general-purpose detector

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(frame, conf=0.5, stream=True)

    # Loop through all detections
    for r in results:
        for box in r.boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Class and confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf:.2f}"

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

            # Label text
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 255), 2)

    cv2.imshow("YOLOv9 Multi-Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
