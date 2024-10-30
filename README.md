# Smart-Parking-Detection-System-with-YOLOv8-and-OpenCV

## Overview

The **Parking Detection System** is an application designed to monitor parking spaces in real-time using a YOLOv8 object detection model. This system captures video input from a camera, detects parked cars, and provides information about the availability of parking lots. The application has a user-friendly interface built using PyQt5, allowing users to start, stop, and quit the detection process easily.

## Features

- Real-time video feed processing from a parking area.
- Object detection using YOLOv8 model to identify parked cars.
- Displays the number of available parking lots with color coding based on availability.
- User-friendly interface with start, stop, and quit buttons.

## Requirements

- Python 3.7 or later
- Required Python packages:
  - OpenCV
  - Pandas
  - NumPy
  - Ultralytics (for YOLOv8)
  - PyQt5

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ViduraSC/Smart-Parking-Detection-System-with-YOLOv8-and-OpenCV.git
   cd ParkingDetectionSystem
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install opencv-python pandas numpy ultralytics PyQt5
   ```

4. Download the YOLOv8 model weights and class file:
   - Place the `yolov8s.pt` file in the project directory.
   - Create a `coco.txt` file containing the class labels (e.g., 'car', 'truck', etc.) used by YOLOv8.

5. Prepare a video file of the parking area:
   - Rename your video file to `parking1.mp4` and place it in the project directory.

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Click the **Start** button to begin detecting parking lots.
3. Click the **Stop** button to pause the detection.
4. Click the **Quit** button to exit the application.

## Acknowledgements

- This project uses the YOLOv8 model developed by Ultralytics for object detection.
- The GUI is built using PyQt5, providing a modern and responsive interface.
