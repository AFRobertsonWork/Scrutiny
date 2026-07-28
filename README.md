# Scrutiny

> Multi-modal Camera Data Collection System for Livestock Monitoring

---

## Overview

Scrutiny is a ROS 2 based multi-camera data acquisition platform developed at the **UK Agri-Tech Centre**.

The system provides a reliable and operator-friendly interface for capturing synchronised data from multiple sensing modalities for livestock research and machine learning applications.

Current sensing modalities include:

- ZED Stereo RGB Camera
- ZED Depth Camera
- Time-of-Flight (ToF) Camera

The recorded datasets will be used to support the development and validation of computer vision models for livestock health assessment.

---

# Features

## Live Camera Dashboard

The operator dashboard provides real-time monitoring of all connected sensors.

Current features include:

- Live ZED RGB feed
- Live ZED Depth visualisation
- Live Time-of-Flight visualisation
- Camera health indicators
- Stream timeout detection
- Frame age monitoring

---

## Data Recording

The system supports one-button recording of synchronised ROS2 bag files.

Features:

- Automatic timestamped recording folders
- Simultaneous recording of all configured topics
- Recording status indicator
- Safe recording shutdown

---

## System Monitoring

The dashboard continuously monitors:

- Available disk space
- System uptime
- CPU temperature
- GPU temperature
- Camera status
- Recording status

---

## Logging

Integrated runtime logging includes:

- Camera startup
- Stream availability
- Recording events
- System warnings
- Error reporting
- Periodic heartbeat monitoring

---

# Hardware

Current supported hardware:

| Device | Purpose |
|---------|----------|
| ZED Camera | RGB + Stereo Depth |
| Synexens ToF Camera | Short-range depth sensing |
| NVIDIA Jetson / ZED Box | Edge computing platform |

---

# Software Stack

- Ubuntu
- ROS 2
- Python
- C++
- OpenCV
- PyQt5
- cv_bridge
- ROS2 Image Transport
- rosbag2

---

# Repository Structure

```
Scrutiny
│
├── src
│   │
│   ├── scrutiny_sleek_gui
│   │   ├── launch
│   │   ├── scrutiny_sleek_gui
│   │   ├── package.xml
│   │   └── setup.py
│   │
│   └── scrutiny_tof_driver
│       ├── src
│       ├── CMakeLists.txt
│       └── package.xml
│
├── build
├── install
└── log
```

---

# Building

Clone the repository

```bash
git clone <repository-url>
```

Build the workspace

```bash
colcon build
```

Source the workspace

```bash
source install/setup.bash
```

---

# Running

Launch the complete camera system

```bash
ros2 launch scrutiny_sleek_gui scrutiny_system.launch.py
```

---

# ROS Topics

### Subscribed

| Topic | Description |
|--------|-------------|
| /zed/zed_node/left/color/rect/image | RGB Camera |
| /zed/zed_node/depth/depth_registered | ZED Depth |
| /scrutiny/tof/depth8 | Time-of-Flight |

---

# Dashboard

The operator interface displays

- ZED RGB
- ZED Depth
- ToF Heatmap
- Runtime Log
- System Status
- Disk Space
- Temperature
- Recording Controls

---

# Recording

Recorded datasets are automatically stored inside

```
~/bags
```

Each recording is timestamped

```
scrutiny_YYYYMMDD_HHMMSS
```

---

# Current Status

## Version

V1.0

### Completed

- Multi-camera dashboard
- ZED RGB visualisation
- ZED Depth processing
- Time-of-Flight driver
- ROS2 launch system
- Live operator dashboard
- Rosbag recording
- Runtime logging
- Temperature monitoring
- Disk monitoring
- Safe shutdown

---

# Planned Features

Future development includes

- Object measurement module
- Object size validation
- RGB alignment stencil
- Dataset quality validation
- Camera calibration tools
- CNN inference
- Automatic health assessment
- Dataset upload utilities
- Configuration management
- Additional sensor support

---

# Design Goals

The project has been designed with the following objectives:

- Reliable long-term operation
- Simple operator workflow
- Modular ROS2 architecture
- Easy hardware integration
- High-quality synchronised datasets
- Future machine learning deployment

---

# Coding Standards

Current development follows

- ROS2 package structure
- Parameterised configuration
- Exception handling
- Modular GUI components
- Git version control
- Incremental feature development

---

# Authors

**Aditya Jadhav**

Robotics Engineer

UK Agri-Tech Centre

---

# Acknowledgements

This project makes use of

- ROS 2
- OpenCV
- NVIDIA Jetson Platform
- Stereolabs ZED SDK
- PyQt5

---

# License

Internal project.

Developed for the UK Agri-Tech Centre.

Distribution outside the organisation should follow UK Agri-Tech Centre policies.
