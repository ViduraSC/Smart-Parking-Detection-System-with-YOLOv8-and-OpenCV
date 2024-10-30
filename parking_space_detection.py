import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# Configuration
MODEL_PATH = 'yolov8s.pt'
VIDEO_PATH = 'parking1.mp4'
CLASS_FILE = 'coco.txt'

# Load YOLOv8 model and class labels
model = YOLO(MODEL_PATH)
with open(CLASS_FILE, "r") as f:
    class_list = f.read().splitlines()

# Define parking areas as a list of tuples
PARKING_AREAS = [
    [(49, 364), (20, 424), (71, 424), (92, 365)],
    [(105, 353), (86, 428), (137, 427), (146, 358)],
    [(159, 354), (150, 427), (204, 425), (203, 353)],
    [(217, 352), (219, 422), (273, 418), (261, 347)],
    [(274, 345), (286, 417), (338, 415), (321, 345)],
    [(336, 343), (357, 410), (409, 408), (382, 340)],
    [(396, 338), (426, 404), (479, 399), (439, 334)],
    [(458, 333), (494, 397), (543, 390), (495, 330)],
    [(509, 335), (557, 388), (603, 383), (549, 324)],
    [(564, 323), (615, 381), (654, 372), (596, 315)],
    [(616, 316), (666, 369), (703, 363), (642, 312)],
    [(674, 311), (730, 360), (764, 355), (707, 308)],
]

class ParkingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parking Detection System")
        self.setStyleSheet("background-color: white;")  # White background
        self.cap = cv2.VideoCapture(VIDEO_PATH)

        # Timer for updating video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # UI Elements
        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("border: 1px solid white;")

        # Current available spaces label
        self.available_label = QLabel("Available Parking Lots: ", self)
        self.available_label.setAlignment(Qt.AlignCenter)  # Center alignment
        self.available_label.setStyleSheet("font-size: 30px; color: black;")  # Black text

        self.start_button = QPushButton("Start", self)
        self.stop_button = QPushButton("Stop", self)
        self.quit_button = QPushButton("Quit", self)

        # Style buttons
        for button in [self.start_button, self.stop_button, self.quit_button]:
            button.setStyleSheet("background-color: lightskyblue; color: black; font-size: 18px; padding: 10px; border-radius: 10px;")
            button.setFixedHeight(50)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.available_label)  # Position above buttons
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)

        # Button actions
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.quit_button.clicked.connect(self.quit)

    def start(self):
        self.timer.start(30)  # 30 ms refresh rate

    def stop(self):
        self.timer.stop()

    def quit(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.close()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        frame = cv2.resize(frame, (1020, 500))
        detections = self.detect_objects(frame)
        self.draw_parking_areas(frame, detections)

        # Convert frame to QImage and display it
        qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def detect_objects(self, frame):
        results = model.predict(frame)[0].boxes.data
        return pd.DataFrame(results).astype("float")

    def draw_parking_areas(self, frame, detections):
        area_counts = [0] * len(PARKING_AREAS)

        for _, row in detections.iterrows():
            x1, y1, x2, y2, _, class_id = map(int, row)
            class_name = class_list[class_id]

            if 'car' in class_name:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                for i, area in enumerate(PARKING_AREAS):
                    if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                        area_counts[i] += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        occupied = sum(min(1, count) for count in area_counts)
        available = len(PARKING_AREAS) - occupied

        # Update available spaces with color coding
        self.update_available_spaces(available)

    def update_available_spaces(self, available):
        color = "red" if available <= 6 else "green"  # Color based on availability
        self.available_label.setText(f"Available Parking Lots: <span style='color: {color};'>{available}</span>")

# Run the application
if __name__ == "__main__":
    app = QApplication([])
    window = ParkingApp()
    window.show()
    app.exec()
