from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('/Users/jawadkon/Downloads/best.pt')  # Replace 'best.pt' with the path to your trained YOLOv8 model

# Set up video capture (e.g., webcam or IP camera)
url = "http://192.168.1.88:81"  # Replace with your IP camera URL
cap = cv2.VideoCapture(url)  # For webcam, use 0

# Process frames from the video feed
while True:
    success, frame = cap.read()
    if not success:
        break

    # Perform inference on the frame
    results = model(frame)

    # Render the detected objects on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()