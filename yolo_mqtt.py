import base64
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from flask import Flask, Response, jsonify
import threading
import json
import time

# === CONFIGURATION ===
USE_WEBCAM = True               # True = Webcam mode, False = ESP32 mode
WEBCAM_INDEX = 0

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_FRAMES = "esp32cam/frame"
TOPIC_DETECTIONS = "yolo/detections"

MODEL_PATH = "leaf_model.pt"
FLASK_PORT = 5000

# === INIT YOLO ===
model = YOLO(MODEL_PATH)

# === GLOBALS ===
output_frame = None
frame_lock = threading.Lock()

latest_detection = {"status": "no_data"}
latest_lock = threading.Lock()


# ---------------------------------------------------------
# MQTT HANDLERS (used only if USE_WEBCAM = False)
# ---------------------------------------------------------
def on_connect(client, userdata, flags, rc):
    print("✅ Connected to MQTT broker")
    if not USE_WEBCAM:
        client.subscribe(TOPIC_FRAMES)


def on_message(client, userdata, msg):
    if USE_WEBCAM:
        return  # ignore MQTT frames in webcam mode

    global output_frame
    try:
        img_data = base64.b64decode(msg.payload)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        process_frame(frame)

    except Exception as e:
        print("❌ Error processing ESP32 frame:", e)


# MQTT client
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()


# ---------------------------------------------------------
# YOLO PROCESS FUNCTION (shared by webcam + ESP32)
# ---------------------------------------------------------
def process_frame(frame):
    global output_frame, latest_detection

    results = model(frame)

    detections = []
    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        conf = float(box.conf)
        detections.append({"label": label, "confidence": round(conf)})

    # ---- Store latest detection for Flutter ----
    with latest_lock:
        if len(detections) == 0:
            latest_detection = {"status": "no_data"}
        else:
            latest_detection = {
                "status": "ok",
                "label": detections[0]["label"],
                "confidence": detections[0]["confidence"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

    # ---- Optional MQTT publish ----
    mqtt_client.publish(TOPIC_DETECTIONS, json.dumps(detections))

    # Draw boxes for streaming
    processed = results[0].plot()

    with frame_lock:
        output_frame = processed.copy()

    print("📡 Detections:", detections)


# ---------------------------------------------------------
# WEBCAM THREAD
# ---------------------------------------------------------
def webcam_loop():
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Webcam not found. Change WEBCAM_INDEX.")
        return

    print("🎥 Webcam started!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame")
            continue

        process_frame(frame)
        time.sleep(0.03)  # ~30 FPS


# ---------------------------------------------------------
# FLASK SERVER
# ---------------------------------------------------------
app = Flask(__name__)


# MJPEG Stream
def generate():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            _, buffer = cv2.imencode(".jpg", output_frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---- NEW API FOR FLUTTER ----
@app.route("/latest_detection")
def latest_detection_api():
    with latest_lock:
        return jsonify(latest_detection)


def run_flask():
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)


# ---------------------------------------------------------
# START SYSTEM
# ---------------------------------------------------------
threading.Thread(target=run_flask, daemon=True).start()

if USE_WEBCAM:
    threading.Thread(target=webcam_loop, daemon=True).start()
    print("🟢 Running in WEBCAM mode...")
else:
    print("🟢 Running in ESP32-MQTT mode...")

print(f"🌐 Stream available at: http://localhost:{FLASK_PORT}/video_feed")

while True:
    time.sleep(1)
