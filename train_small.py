from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="Flame/data.yaml",
    epochs=8,
    imgsz=320,
    batch=4
)