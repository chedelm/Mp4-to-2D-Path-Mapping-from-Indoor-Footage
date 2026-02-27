# Video-to-2D Mapping Prototype (Computer Vision)

## Overview
A small beginner computer vision prototype that explores how video footage can be converted into a rough 2D spatial representation.  
The project focuses on demonstrating a basic pipeline, not mapping accuracy.

---

## What It Does
Given a short indoor video, the system:
- samples video frames  
- detects visual features  
- estimates camera motion using optical flow  
- builds an estimated **2D camera path**  
- accumulates edges into a simple map-like visualization  

---

## Requirements
- Python 3.9+
- OpenCV
- NumPy
- Matplotlib

## How to run:

From project root, type python main.py --video <path location of the video>
