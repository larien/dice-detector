import cv2
import numpy as np


def load():
    return cv2.imread("dados.jpg", cv2.IMREAD_COLOR)


def dice_pattern():
    return np.ones((3,3),np.uint8)


def binary(original):
    _, treated = cv2.threshold(original,250,250,cv2.THRESH_BINARY_INV)
    return cv2.morphologyEx(treated, cv2.MORPH_OPEN, dice_pattern())


def set_info(image, x, y, w, h):
    area = image[y:y+h,x:x+w].copy()

    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(area)
    
    cv2.putText(image,str(len(keypoints)),(int(x+(w/2)),int(y+(h/2))), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),2,cv2.LINE_AA)
    
    return image


def contour(original, gray, treated):
    contours, _ = cv2.findContours(treated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape
    min_x, min_y  = width, height
    max_x = max_y = 0

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        print("x = ", x, " y= ", y, " w = ", w, " h = ", h)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        original = set_info(original, x, y, w, h)

    return original


if __name__ == "__main__":
    image = load()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary_image = binary(gray_image)

    detected_dices = contour(image, gray_image, binary_image)

    print("Azul é vida!")

    cv2.imshow("Dices",  detected_dices)

    cv2.waitKey(0)
