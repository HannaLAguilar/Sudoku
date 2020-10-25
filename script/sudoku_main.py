import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import *


img_path = '../imgs/1.jpg'
img_original = cv2.imread(img_path)[:, :, ::-1]
width, height = 450, 450
img = cv2.resize(img_original, (width, height))
img_black = np.zeros((height, width, 3), np.uint8)
img_threshold0 = pre_processing0(img)
img_threshold1 = pre_processing1(img)

img_stack = np.hstack((img_threshold0, img_threshold1))
plt.figure()
plt.imshow(img_stack, cmap='gray')

contours0, img_contours0 = all_contours(img_threshold0, img)
contours1, img_contours1 = all_contours(img_threshold1, img)

img_contours_stack = np.hstack((img_contours0, img_contours1))
plt.figure()
plt.imshow(img_contours_stack)

sudoku_contour, points = right_contour_(contours1)

img_sudoku_contours = img.copy()
cv2.drawContours(img_sudoku_contours, sudoku_contour, -1, 255, 2)
plt.figure()
plt.imshow(img_sudoku_contours)

