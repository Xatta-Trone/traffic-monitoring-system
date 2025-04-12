# 🚗 Real-Time Traffic Monitoring System 🎥

## 📝 Overview
A computer vision-based system that tracks vehicles, calculates their speeds, and logs GPS coordinates in real-time using YOLOv11.



### Vehichles
https://github.com/user-attachments/assets/1637e15c-10b3-41cb-9d4d-78570824b695


### Train
https://github.com/user-attachments/assets/fe75fa8f-6eb2-499b-b981-176086035fe3


### Pedestrian-Bicyclists
https://github.com/user-attachments/assets/d976aef9-aa1d-430a-8e78-5ea377b83f35



## 🌟 Features
- 🎯 Real-time vehicle detection and tracking
- 🚦 Speed calculation with perspective transformation
- 📍 GPS coordinate mapping
- 📊 Data logging in CSV format
- 🎨 Visual tracking with bounding boxes

## 🛠️ Technologies Used
- Python 3.8+
- YOLOv8
- OpenCV
- NumPy
- Supervision

## 📋 Prerequisites
```bash
pip install ultralytics
pip install supervision
pip install opencv-python
pip install numpy
```

## 🚀 Quick Start
1. Clone the repository
```bash
git clone https://github.com/xatta-trone/traffic-monitoring-system.git
cd traffic-monitoring-system
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the program
```bash
python main_homography.py
```

## 📸 Usage
1. Launch the program
2. Select ROI (Region of Interest) by clicking 4 points
3. Select the crossing point
4. System will start tracking vehicles and calculating speeds
5. Data is saved in CSV format with timestamps

## 📊 Output Format
The system generates a CSV file with the following information:
- Timestamp
- Object ID
- Object Type
- Speed (MPH)
- Current Latitude
- Current Longitude
- Trajectory Points

## 📊 Sample Data Preview
Here's a snapshot of the actual tracking data output:

| Timestamp           | Object_ID | Object_Type | Speed_MPH | Current_Lat | Current_Long | Trajectory_Points |
|--------------------|-----------|-------------|-----------|-------------|--------------|------------------|
| 2025-04-07 01:47:24| 9         | car         | 4.9       | 29.881146   | -97.931758   | (29.881146, -97.931758) |
| 2025-04-07 01:47:26| 9         | car         | 5.3       | 29.881126   | -97.931749   | (29.881126, -97.931749) |
| 2025-04-07 01:47:27| 9         | car         | 6.4       | 29.881113   | -97.931744   | (29.881113, -97.931744) |
| 2025-04-07 01:47:30| 17        | car         | 4.8       | 29.881164   | -97.931764   | (29.881164, -97.931764) |
| 2025-04-07 01:47:31| 17        | car         | 4.5       | 29.881137   | -97.931754   | (29.881137, -97.931754) |

> Note: This data shows actual vehicle tracking results with their timestamps, speeds, and GPS coordinates.

## 🗺️ Coordinate System
The system uses the following GPS boundaries:
```
Top-left: 29.881259, -97.931718
Top-right: 29.881222, -97.931814
Bottom-right: 29.881081, -97.931782
Bottom-left: 29.881109, -97.931691
```

## 📝 License
MIT License

## 👥 Contributors
- Xatta Trone (@xatta-trone)

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/Xatta-Trone/traffic-monitoring-system/issues).

## ⭐ Show your support 
Give a ⭐️ if this project helped you!
