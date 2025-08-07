from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',  
        ]

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf)
                if conf >= self.conf_threshold:
                    class_id = int(box.cls)
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown object"
                    xyxy = box.xyxy[0].tolist()  # bounding box coordinates
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': xyxy
                    })
        return detections
