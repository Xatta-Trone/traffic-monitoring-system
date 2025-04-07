# 🚗 Real-Time Traffic Monitoring System 🎥

## 📝 Overview
A computer vision-based system that tracks vehicles, calculates their speeds, and logs GPS coordinates in real-time using YOLOv11.




https://github.com/user-attachments/assets/1637e15c-10b3-41cb-9d4d-78570824b695



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
python main.py
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
