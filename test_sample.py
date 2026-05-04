from ultralytics import YOLO


MODEL = YOLO('./pretrained/best.pt')
MODEL.predict('path/to/data/',save=True, imgsz = 640)
