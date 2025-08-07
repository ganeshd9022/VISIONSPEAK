import cv2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        cv2.imshow('Webcam Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key pressed. Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
