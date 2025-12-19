# COLOR_DETECTOR

COLOR_DETECTOR is a real-time color detection system using computer vision and deep learning. It detects people in a video stream and analyzes the dominant color in the detected region.

## Features

- Real-time person detection using YOLOv8.
- Cropping and smoothing bounding boxes for robust tracking.
- Dominant color extraction using KMeans clustering.
- Color classification based on a predefined palette.
- Modular design for easy extension.

## Installation

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

## DSP Pipeline Overview

1. **Video Capture**
	- OpenCV captures frames from the camera (`cv2.VideoCapture`).

2. **Object Detection**
	- YOLOv8 model (`ultralytics.YOLO`) detects people in each frame.
	- Bounding boxes are extracted for detected persons.

3. **Bounding Box Processing**
	- Bounding boxes are smoothed for stability (`smooth_bbox`).
	- Cropping is performed with padding to focus on the region of interest (`crop_with_padding`).

4. **Color Extraction**
	- The cropped image is resized and converted to HSV.
	- Pixels are filtered to remove highlights, skin, and background noise.
	- KMeans clustering is applied to find the dominant color (`dominant_bgr_kmeans`).

5. **Color Classification**
	- The dominant color is compared to a palette in LAB color space for perceptual accuracy.
	- The closest color name is selected from the palette.

6. **Visualization**
	- The detected bounding box and color label are drawn on the frame.
	- The processed frame is displayed in real time.

## Color Palette

Supported colors (BGR):

- Black, White, Gray
- Red, Pink, Purple
- Orange, Yellow
- Green, Blue, Cyan
- Brown

## File Structure

- `main.py`: Main pipeline and video processing.
- `color/dominant_color.py`: Dominant color extraction logic.
- `color/palette.py`: Color palette definitions.
- `color/__init__.py`: Color detection interface.
- `requirements.txt`: Python dependencies.

## Requirements

- ultralytics>=8.0.0
- opencv-python>=4.8.0
- numpy>=1.23
- Pillow>=9.0

## Notes

- The YOLOv8 model file (`yolov8n.pt`) must be present in the project directory.
- Camera index may need adjustment depending on your hardware.
