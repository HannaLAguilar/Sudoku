import numpy as np
import matplotlib.pyplot as plt
import cv2


def pre_processing0(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    img_canny = cv2.Canny(img_blur, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    img_dila = cv2.dilate(img_canny, kernel, iterations=1)
    img_threshold = cv2.erode(img_dila, kernel, iterations=1)
    return img_threshold


def pre_processing1(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return img_threshold


def all_contours(img, img_color):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = img_color.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    return contours, img_contours


def right_contour_(contours):
    right_contour = None
    points = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            perimeter = cv2.arcLength(cnt, True)
            box = cv2.approxPolyDP(cnt, perimeter*0.1, True)
            if area > max_area and len(box) == 4:
                right_contour = [cnt]
                points = box
                max_area = area
    return right_contour, points


def reorder_points(points):
    points = points.reshape(-1, 2)
    sum = points.sum(axis=1)
    pt1 = points[np.argmin(sum)]
    pt4 = points[np.argmax(sum)]
    diff = np.diff(points, axis=1)
    pt2 = points[np.argmin(diff)]
    pt3 = points[np.argmax(diff)]

    ordered_points = np.array([pt1, pt2, pt3, pt4])
    return ordered_points


def split_box_numbers(img):
    rows = np.vsplit(img, 9)
    num_boxes = []
    for row in rows:
        num_col = np.hsplit(row, 9)
        for num in num_col:
            num_boxes.append(num)
    return num_boxes

# plt.figure(), plt.imshow(rows[0], cmap='gray')


