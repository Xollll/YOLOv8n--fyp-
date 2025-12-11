import cv2
from ultralytics import YOLO

model = YOLO("leaf_model.pt")

# Replace with your ESP32-CAM IP address
esp32_url = "http://192.168.8.52:81/stream"  

cap = cv2.VideoCapture(esp32_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8n ESP32-CAM", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
