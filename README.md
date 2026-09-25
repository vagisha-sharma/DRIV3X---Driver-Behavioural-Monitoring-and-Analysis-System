# DRIV3X — Driver Behavioural Monitoring and Analysis System

DRIV3X is an AI-powered computer vision system designed to monitor driving behaviour from video and identify potentially unsafe driving events.

The system uses **YOLOv8, OpenCV, and Streamlit** to process driving videos, detect vehicles and traffic signs, analyze driving behaviour, and generate structured event information.

## Features

* **Lane Departure Detection** — identifies when a vehicle moves beyond the defined lane boundaries.
* **Tailgating Detection** — monitors unsafe following distances between vehicles.
* **Lead Vehicle Monitoring** — identifies and analyzes the vehicle directly ahead.
* **Traffic/Sign Violation Detection** — detects relevant traffic-sign and signal violations.
* **Object Detection & Tracking** — uses YOLO-based detection and tracking to identify vehicles and maintain object identities.
* **Event Severity** — records detected events along with severity information.
* **Timestamped Events** — associates detected behaviours with their occurrence in the video.
* **JSON Event Reporting** — stores detected events and related information in structured JSON format.
* **Streamlit Interface** — provides a simple interface for uploading and analyzing driving videos.

## How It Works

```text
Driving Video
      ↓
Streamlit Video Upload
      ↓
Video Processing
      ↓
YOLOv8 Object Detection
      ↓
Object Tracking
      ↓
 ┌─────────────────────────────┐
 │ Behaviour Analysis          │
 │                             │
 │ • Lane Departure            │
 │ • Tailgating                │
 │ • Lead Vehicle Monitoring   │
 │ • Sign/Signal Violations    │
 └─────────────────────────────┘
      ↓
Event & Severity Analysis
      ↓
Timestamped JSON Report
```

## Technologies Used

* **Python**
* **YOLOv8 / Ultralytics**
* **OpenCV**
* **Streamlit**
* **NumPy**
* **Pandas**
* **PyTorch**

## Project Structure

```text
DRIV3X/
│
├── app.py
├── main.py
├── config.toml
├── events.json
├── requirements.txt
│
├── behaviour/
│   ├── lane_departure.py
│   ├── lead_vehicle.py
│   ├── sign_violation.py
│   ├── tailgating.py
│   └── tracker.py
│
├── core/
│   └── pipeline.py
│
├── vision/
│   ├── lanes.py
│   ├── objects.py
│   └── signs.py
│
├── debug/
│   ├── debug_lanes.py
│   ├── debug_objects.py
│   └── debug_signs.py
│
├── models/
│   └── signs_best.pt
│
├── testing/
│   └── Test driving videos
│
└── yolov8n.pt
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/vagisha-sharma/DRIV3X---Driver-Behavioural-Monitoring-and-Analysis-System.git
cd DRIV3X---Driver-Behavioural-Monitoring-and-Analysis-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application opens a web interface where a compatible driving video can be uploaded for analysis.

### Supported Video Formats

* MP4
* AVI
* MOV

## Output

DRIV3X records detected driving events with information such as:

* Behaviour/event type
* Timestamp
* Severity
* Detection confidence
* Object/track information
* Relevant positional information

The project also includes structured JSON event output for storing detection results.

## Testing

The repository includes sample driving videos under the `testing/` directory that can be used to test the system.

The primary Streamlit application also supports uploading external driving videos through the interface.

## Models

The project uses:

* **YOLOv8 Nano (`yolov8n.pt`)** for object detection.
* **Custom trained sign-detection model (`models/signs_best.pt`)** for traffic/sign detection.

## Future Improvements

Potential improvements include:

* Improving detection accuracy across different road and weather conditions.
* Expanding the range of driving behaviours and traffic violations detected.
* Improving object tracking robustness.
* Adding more detailed analytics and visualizations.
* Optimizing processing speed for real-time or near-real-time applications.
* Deploying the application as a scalable web service.

## Disclaimer

DRIV3X is an academic/project implementation intended for experimentation and demonstration of computer vision techniques for driver behaviour monitoring. Detection results may vary depending on video quality, camera position, lighting, road conditions, and other environmental factors.

## Author

**Vagisha Sharma**

[GitHub](https://github.com/vagisha-sharma)
