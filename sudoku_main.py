import matplotlib.pyplot as plt
from utils import *
from model_cnn import *
from p import photo_

# Image preprocessing
img_path = 'imgs/4.jpg'
img_original = cv2.imread(img_path)[:, :, ::-1]
width, height = 450, 450
img = cv2.resize(img_original, (width, height))
img_threshold = pre_processing(img)

# Contours
contours, img_contours = all_contours(img_threshold, img)  # all contours
sudoku_contour, points = right_contour_(contours)  # sudoku contour
img_sudoku_contours = img.copy()
img_points = img.copy()
cv2.drawContours(img_sudoku_contours, sudoku_contour, -1, 255, 2)
cv2.drawContours(img_points, points, -1, 255, 15)

# WarpPerspective
pts1 = np.float32(reorder_points(points))
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
img_warp = cv2.warpPerspective(img, matrix, (width, height))
img_warp_gray = cv2.cvtColor(img_warp, cv2.COLOR_RGB2GRAY)

# Split numbers
numbers_box = split_box_numbers(img_warp_gray)
numbers = get_numbers(numbers_box, Classifier(), 'classifier_digit.pt')
print(numbers)
photo_(img_path, numbers)
l = [ii for ii in numbers.ravel()]

# Visualization
img_stack = ([img, img_threshold, img_contours, img_points],
             [img_warp_gray, displayNumbers(np.zeros_like(img), l), np.zeros_like(img), np.zeros_like(img)])
img_stack = stackImages(img_stack, 0.7)[:, :, ::-1]
cv2.imshow('Stacked Images', img_stack)
