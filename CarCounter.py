import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Tracking
tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]


def main():
    classNames = [
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    total_Counts = []
    try:
        cap = cv2.VideoCapture('cars.mp4')

        model = YOLO("../Yolo-Weights/yolov8l.pt")
        # cap.set(3,1280)
        # cap.set(4,480)
        mask = cv2.imread("mask0.png")

        while True:
            success, img = cap.read()
            imgRegion = cv2.bitwise_and(img, mask)
            results = model(imgRegion, stream=True)
            detections = np.empty((0, 5))

            for r in results:
                # getting bounding box
                boxes = r.boxes
                for box in boxes:
                    # x1, y1, x2, y2 = box.xyxy[0]
                    # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # print(x1, y1, x2, y2)
                    # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    bbox = int(x1), int(y1), int(w), int(h)
                    # print(x1, y1, w, h)
                    # cvzone.cornerRect(img, bbox,l=5)
                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # print(conf)
                    # Get the Class ID

                    cls = box.cls[0]
                    className = classNames[int(cls)]
                    if className == "car" or className == "truck" or className == "motorbike" or className == "bus" and conf >= 0.3:
                        # cvzone.putTextRect(img, f' {classNames[int(cls)]} {conf} Id:{Id} ', (max(0, x1), max(35, y1)), scale=1,
                        #                    thickness=1)
                        # cvzone.cornerRect(img, bbox, l=5, rt=5)
                        # cvzone.putTextRect(imgRegion, f' {classNames[int(cls)]} {conf} {Id}', (max(0, x1), max(35, y1)),
                        #                    scale=1,
                        #                    thickness=1)
                        # cvzone.cornerRect(imgRegion, bbox, l=5)
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))

                    # print(f"\r[+] Confidence Value: {conf}", end="")

            # cv2.imshow("ImageRegion,", imgRegion)
            resultsTracker = tracker.update(detections)
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
            for marker in resultsTracker:
                x1, y1, x2, y2, Id = marker
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                bbox = int(x1), int(y1), int(w), int(h)
                cx, cy = x1 + w // 2, y1 + h // 2
                # Center position
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
                    if total_Counts.count(Id) == 0:
                        total_Counts.append(Id)
                        print("[+] Vehicle Tracker:", total_Counts)
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

                print(marker)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                cvzone.putTextRect(img, f' {str(Id)}', (max(0, x1), max(35, y1)),
                                   scale=2, thickness=3, offset=10)
            cvzone.putTextRect(img, f'Counts : {str(len(total_Counts))}', (50, 50),
                               scale=2, thickness=3, offset=10)
            cv2.imshow("Image,", img)
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    try:
        print("[+] Program Initiated ..")
        main()
    except KeyboardInterrupt:
        print("[+] Operation Cancelled")
