# Object Tracking with YOLO and SORT

This Python script utilizes YOLO (You Only Look Once) for object detection and SORT (Simple Online and Realtime Tracking) for object tracking. It can be used to detect and track various objects in a video stream.

## Dependencies

- Python 3.x
- numpy
- ultralytics
- OpenCV (cv2)
- cvzone
- math
- sort

## Installation

1. Clone this repository:

```
# Note make sure to install 
driver https://www.nvidia.com/download/index.aspx
cuda tool kit https://developer.nvidia.com/cudnn
cuDN https://developer.nvidia.com/cudnn
git clone https://github.com/your_username/your_repository.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Make sure you have a video file named `cars.mp4` in the same directory as the script.
2. Run the script:

```
python object_tracking.py
```

3. Press 'q' to exit the video stream.

## Description

The script performs the following steps:

1. Imports necessary libraries including YOLO, OpenCV, and SORT.
2. Initializes a SORT tracker with specified parameters.
3. Defines the main function which:
   - Loads YOLO model weights.
   - Reads video frames from `cars.mp4`.
   - Detects objects using YOLO.
   - Filters detected objects based on class labels and confidence threshold.
   - Tracks filtered objects using SORT.
   - Displays the video stream with tracked objects and counts their occurrences.


## Reference

for more details please check
```bash
https://www.youtube.com/@murtazasworkshop
https://www.youtube.com/watch?v=WgPbbWmnXJ8
```

## Notes

- You may need to adjust the paths to the YOLO model weights (`yolov8l.pt`) and mask image (`mask0.png`) as per your directory structure.
- Ensure that your environment has sufficient resources to run the script, especially for real-time processing.
---

