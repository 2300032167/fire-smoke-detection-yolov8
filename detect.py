from ultralytics import YOLO

model = YOLO("C:/Users/china/runs/detect/train12/weights/best.pt")

model.predict(
    source="test1.jpg",
    conf=0.1,
    show=True,
    save=True,
    project="results",
    name="output",
    exist_ok=True
)