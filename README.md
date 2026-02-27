# Video-to-2D Mapping Prototype (Computer Vision)

## Overview
A small computer vision prototype that explores how **video footage** can be converted into a **rough 2D spatial representation**.  
The project focuses on demonstrating a **basic pipeline**, not mapping accuracy.

---

## What It Does
Given a short indoor video, the system:
- samples video frames  
- detects visual features  
- estimates camera motion using optical flow  
- builds an estimated **2D camera path**  
- accumulates edges into a simple map-like visualization  

---

## Project Structure
video_to_map_prototype/
├── main.py
├── requirements.txt
├── video/
│ └── test.mp4
├── src/
│ ├── config.py
│ ├── video_io.py
│ ├── motion.py
│ ├── mapper.py
│ └── viz.py
└── outputs/

---

## Requirements
- Python 3.9+
- OpenCV
- NumPy
- Matplotlib
