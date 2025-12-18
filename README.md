# Real-time Object Color Detection (YOLOv8 + Rule-based Color)

## 📌 Overview
This project implements a **real-time object detection and color recognition system**
using a webcam.

Pipeline:
- Detect object inside a fixed **ROI (square guide)** using **YOLOv8**
- Smooth bounding box for stable visualization
- Crop the object region
- Detect **dominant color** using **rule-based HSV + KMeans (no ML training required)**

The system is optimized for **real-time performance** (30–60 FPS) on consumer hardware.

---

## 🎯 Features
- Real-time object detection via webcam
- ROI-based detection (user-guided)
- Bounding box smoothing (EMA)
- Robust color detection:
  - Black / White / Gray
  - Red, Pink, Purple
  - Yellow, Orange, Green, Blue
- Works with:
  - Phones
  - Books
  - Tissue packs
  - Plastic / reflective objects
- No color dataset training required

---

## 🧠 Color Detection Strategy
Color detection is **rule-based**, not ML-based:

1. Convert crop to HSV
2. Use **Hue-based rules** for semantic colors
3. Use **neutral detection** for black/white/gray
4. Use **KMeans + LAB distance** as fallback
5. Center-crop to avoid background / hand interference

This approach is:
- Faster than ML
- More stable under lighting changes
- Easier to explain in Image Processing courses

---

## 📂 Project Structure
```text
project/
│
├── main.py                     # Webcam + YOLO pipeline
├── requirements.txt
├── README.md
│
├── color/
│   ├── __init__.py
│   ├── dominant_color.py       # Color detection logic
│   └── palette.py              # Reference color palette
│
└── models/
    └── yolov8n.pt              # YOLOv8 model (auto-downloaded)
