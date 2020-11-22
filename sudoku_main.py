import matplotlib.pyplot as plt
from utils import *
from model_cnn import *

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

img_box = pre_processing(img_warp)

# Split numbers
numbers_box = split_box_numbers(img_warp_gray)

# Visualization
# img_stack = ([img, img_threshold, img_contours, img_points],
#              [img_warp_gray, img, img, img])
# img_stack = stackImages(img_stack, 0.7)[:, :, ::-1]
# cv2.imshow('Stacked Images', img_stack)

plt.figure(), plt.imshow(img_warp_gray, cmap='gray')
# i = 22
# img = numbers_box[i]
# plt.figure(), plt.imshow(img, cmap='gray')
# # img = img[5:img.shape[0] - 5, 5:img.shape[1] - 5]
# img = get_digit(img)
# plt.figure(), plt.imshow(img, cmap='gray')
# pred, img, prob = prediction(img, Network(), 'classifier_digit.pt')
# plt.figure(), plt.imshow(img, cmap='gray')
# print(pred)
# print(prob)


l = []
l2 = []
for i in range(len(numbers_box)):
    img = numbers_box[i]
    img = get_digit(img)
    # img = preprocessImage(img)
    # img = cv2.dilate(img, kernel=np.ones((5,5),np.uint8), iterations=1)
    # img = cv2.equalizeHist(img)
    # pred, img, prob = prediction(img, Classifier(), 'classifier_digit99.pt')
    pred, img, prob = prediction(img, Network(), 'classifier_digit.pt')
    l.append(pred)
    l2.append(prob)
l = np.array(l).reshape(9, 9)
l2 = np.array(l2).reshape(9, 9)

photo = None
if img_path == 'imgs/1.jpg':
    photo = np.array([[2, 1, 0, 0, 6, 0, 9, 0, 0],
                      [0, 0, 0, 0, 0, 9, 1, 0, 0],
                      [4, 0, 9, 3, 1, 0, 0, 5, 8],
                      [0, 0, 1, 0, 0, 5, 0, 4, 0],
                      [9, 0, 4, 0, 3, 0, 8, 0, 5],
                      [0, 5, 0, 2, 0, 0, 6, 0, 0],
                      [3, 8, 0, 0, 4, 0, 5, 0, 6],
                      [0, 0, 6, 7, 0, 0, 0, 0, 2],
                      [0, 0, 7, 0, 8, 0, 3, 0, 9]])
elif img_path == 'imgs/2.jpg':
    photo = np.array([[5, 3, 0, 6, 0, 0, 0, 4, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 3, 0, 2, 0],
                      [0, 8, 6, 2, 0, 0, 5, 0, 7],
                      [0, 0, 7, 0, 0, 0, 4, 0, 0],
                      [3, 0, 5, 0, 0, 7, 2, 8, 0],
                      [0, 9, 0, 3, 0, 0, 0, 0, 0],
                      [0, 0, 4, 0, 0, 0, 0, 0, 0],
                      [0, 6, 0, 0, 0, 2, 0, 7, 8]])
elif img_path == 'imgs/3.jpg':
    photo = np.array([[0, 0, 4, 0, 0, 0, 6, 7, 1],
                      [0, 9, 1, 0, 0, 3, 0, 4, 0],
                      [0, 0, 0, 0, 8, 1, 0, 0, 5],
                      [0, 0, 0, 3, 9, 6, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 9, 0, 0],
                      [2, 5, 0, 0, 0, 8, 0, 0, 0],
                      [9, 0, 0, 0, 6, 7, 0, 0, 0],
                      [4, 0, 0, 2, 0, 0, 0, 3, 0],
                      [0, 2, 6, 0, 0, 0, 7, 0, 0]])
elif img_path == 'imgs/4.jpg':
    photo = np.array([[8, 0, 0, 0, 1, 0, 0, 0, 9],
                      [0, 5, 0, 8, 0, 7, 0, 1, 0],
                      [0, 0, 4, 0, 9, 0, 7, 0, 0],
                      [0, 6, 0, 7, 0, 1, 0, 2, 0],
                      [5, 0, 8, 0, 6, 0, 1, 0, 7],
                      [0, 1, 0, 5, 0, 2, 0, 9, 0],
                      [0, 0, 7, 0, 4, 0, 6, 0, 0],
                      [0, 8, 0, 3, 0, 9, 0, 4, 0],
                      [3, 0, 0, 0, 5, 0, 0, 0, 8]])
elif img_path == 'imgs/5.jpg':
    photo = np.array([[0, 4, 0, 0, 0, 2, 0, 1, 9],
                      [0, 0, 0, 3, 5, 1, 0, 8, 6],
                      [3, 1, 0, 0, 9, 4, 7, 0, 0],
                      [0, 9, 4, 0, 0, 0, 0, 0, 7],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [2, 0, 0, 0, 0, 0, 8, 9, 0],
                      [0, 0, 9, 5, 2, 0, 0, 4, 1],
                      [4, 2, 0, 1, 6, 9, 0, 0, 0],
                      [1, 6, 0, 8, 0, 0, 0, 7, 0]])



else:
    pass
print(l)
print('DIFERENCIA:', 81 - np.sum(photo == l))
print(l2.round(3))

# # img = preprocessImage(img)
# # plt.figure(), plt.imshow(img, cmap='gray')
# # _, img = centeringImage(img)
# # plt.figure(), plt.imshow(img, cmap='gray')
