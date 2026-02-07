from ultralytics import YOLO
import cv2
import csv
from datetime import datetime
import numpy as np
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load crowd video
cap = cv2.VideoCapture("../Videos/crowd.mp4")

# Heatmap
heatmap = None

# CSV file path
csv_path = "../Data/crowd_data.csv"

# Create CSV once
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "total", "zone1", "zone2", "zone3", "zone4"])

# Zone safety logic
def zone_status(count):
    if count <= 2:
        return "SAFE", (0, 255, 0)
    elif count <= 5:
        return "MODERATE", (0, 255, 255)
    else:
        return "DANGEROUS", (0, 0, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    if heatmap is None:
        heatmap = np.zeros(frame.shape[:2], dtype=np.float32)

    results = model(frame)
    annotated = results[0].plot()

    zone1 = zone2 = zone3 = zone4 = 0
    total = 0

    for box in results[0].boxes:
        if int(box.cls[0]) == 0:  # person class
            total += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            heatmap[cy, cx] += 1

            if cx < w // 2 and cy < h // 2:
                zone1 += 1
            elif cx >= w // 2 and cy < h // 2:
                zone2 += 1
            elif cx < w // 2 and cy >= h // 2:
                zone3 += 1
            else:
                zone4 += 1

    # Heatmap overlay
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype("uint8"), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(annotated, 0.6, heatmap_color, 0.4, 0)

    # Zone statuses
    z1_status, z1_color = zone_status(zone1)
    z2_status, z2_color = zone_status(zone2)
    z3_status, z3_color = zone_status(zone3)
    z4_status, z4_color = zone_status(zone4)

    # Draw zone borders
    cv2.rectangle(overlay, (0, 0), (w // 2, h // 2), z1_color, 3)
    cv2.rectangle(overlay, (w // 2, 0), (w, h // 2), z2_color, 3)
    cv2.rectangle(overlay, (0, h // 2), (w // 2, h), z3_color, 3)
    cv2.rectangle(overlay, (w // 2, h // 2), (w, h), z4_color, 3)

    # Display info
    cv2.putText(overlay, f"Total: {total}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(overlay, f"Z1: {zone1} {z1_status}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, z1_color, 2)

    cv2.putText(overlay, f"Z2: {zone2} {z2_status}", (w // 2 + 20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, z2_color, 2)

    cv2.putText(overlay, f"Z3: {zone3} {z3_status}", (20, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, z3_color, 2)

    cv2.putText(overlay, f"Z4: {zone4} {z4_status}", (w // 2 + 20, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, z4_color, 2)

    # Save zone data
    current_time = datetime.now().strftime("%H:%M:%S")
    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([current_time, total, zone1, zone2, zone3, zone4])

    cv2.imshow("Smart Crowd Monitoring System", overlay)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
