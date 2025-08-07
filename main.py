import cv2
import time
from object_detection_platform.detector import ObjectDetector
from object_detection_platform.speaker import VoiceSpeaker

def main():
    # Configuration
    MODEL_PATH = 'yolov8n.pt'
    CONFIDENCE_THRESHOLD = 0.5
    COOLDOWN = 5  # seconds between voice announcements
    ENABLE_VOICE_FEEDBACK = True

    # Initialize detector and speaker
    detector = ObjectDetector(model_path=MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD)
    speaker = VoiceSpeaker(enable_voice=ENABLE_VOICE_FEEDBACK)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_announced = {}

    print("Starting object detection. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from webcam.")
            time.sleep(0.1)
            continue

        detections = detector.detect(frame)

        # Draw bounding boxes and announce detected objects
        current_time = time.time()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            confidence = det['confidence']

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

            # Voice feedback with cooldown
            last_time = last_announced.get(class_name, 0)
            if (current_time - last_time) > COOLDOWN:
                speaker.speak(f"A {class_name} detected with confidence {confidence:.2f}")
                last_announced[class_name] = current_time

        cv2.imshow('Object Detection Platform', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    speaker.stop()

if __name__ == "__main__":
    main()
