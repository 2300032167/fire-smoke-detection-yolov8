import cv2
from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Confidence threshold
CONF_THRESHOLD = 0.5

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            confidence = float(box.conf[0])

            # Apply confidence filtering
            if confidence > CONF_THRESHOLD:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label text
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

                # Print detection in terminal
                print(f"Detected: {class_name} | Confidence: {confidence:.2f}")

    cv2.imshow("Fire and Smoke Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()