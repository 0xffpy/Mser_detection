import cv2
import numpy as np
from time import time


def mser_create(image, mser_object, length=100000, ratio=10):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, ksize=(9, 9), sigmaX=0)
    regions, _ = mser_object.detectRegions(image_blur)
    boxes = []
    for p in regions:
        x_max, y_max = np.amax(p, axis=0)
        x_min, y_min = np.amin(p, axis=0)
        if not (abs(x_max - x_min) > length or abs(y_min - y_max) > length):
            if not (abs(y_min - y_max) / abs(x_max - x_min) > ratio
                    or abs(x_max - x_min) / abs(y_min - y_max) > ratio):
                boxes.append((x_min, y_min, x_max, y_max))
    return np.array(boxes)


def non_max_suppression_fast(boxes, overlapThresh=0.25):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")


def main():
    video_object = cv2.VideoCapture()
    mser_object = cv2.MSER_create()

    while True:
        frame, ret = video_object.read()
        if not ret:
            break
        image = cv2.resize(frame, (1080, 720))
        box = non_max_suppression_fast(mser_create(image, mser_object))
        if not box:
            '''code to trigger the sony camera is here 
               so if there is a blob detected the camera will
               take a photo and send it to the ground station'''
            pass

        if cv2.waitKey(5) & 0XFF == ord('q'):
            break


def main_1():
    mser_object = cv2.MSER_create()
    for i in range(825, 925):
        # time_ = time()
        image = cv2.imread(f'Pictures/DSC00{str(i)}.JPG')
        image = cv2.resize(image, (1080, 720))
        time_ = time()

        box = non_max_suppression_fast(mser_create(image, mser_object))
        for coor in box:
            x_min, y_min, x_max, y_max = coor
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 0),
                          thickness=1)
        print("Time it took: ", time() - time_)
        cv2.imshow("Pic", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main_1()
