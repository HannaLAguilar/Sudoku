import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import *

# Image
img_path = '../imgs/1.jpg'
img_original = cv2.imread(img_path)[:, :, ::-1]
width, height = 450, 450
img = cv2.resize(img_original, (width, height))

# Image preprocessing
img_threshold0 = pre_processing0(img)
img_threshold1 = pre_processing1(img)
img_stack = np.hstack((img_threshold0, img_threshold1))
plt.figure()
plt.imshow(img_stack, cmap='gray')

# All Contours
contours0, img_contours0 = all_contours(img_threshold0, img)
contours1, img_contours1 = all_contours(img_threshold1, img)
img_contours_stack = np.hstack((img_contours0, img_contours1))
plt.figure()
plt.imshow(img_contours_stack)

# Sudoku contour
sudoku_contour, points = right_contour_(contours0)
img_sudoku_contours = img.copy()
cv2.drawContours(img_sudoku_contours, sudoku_contour, -1, 255, 2)
plt.figure()
plt.imshow(img_sudoku_contours)

# WarpPespective
pts1 = reorder_points(points)



