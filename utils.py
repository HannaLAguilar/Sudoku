import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image


def pre_processing(img):
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
            box = cv2.approxPolyDP(cnt, perimeter * 0.1, True)
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


def only_digit(img, offset=10):
    # img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
    thresh = cv2.threshold(img, 0, 50, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 7 and (15 <= h <= 80) and ((0 < x < img.shape[1]) and (0 < y < img.shape[0])):
            rect = [x, y, w, h]
            (x, y, w, h) = rect
            digit = img[y - offset:y + h + offset, x - offset:x + w + offset]
    if len(digit):
        return digit
    else:
        return img


def prediction(img, model_cnn, model_state, cut=True):
    # image processing
    if cut:
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
    img = Image.fromarray(np.uint8(img))
    transform = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)
    # img_np = img_tensor.numpy().reshape(32, 32)

    # model
    device = torch.device('cpu')
    model = model_cnn
    model.to(device)
    state_dict = torch.load(model_state)
    model.load_state_dict(state_dict)

    # predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        output_sof = torch.softmax(output, dim=1)
        prob, pred = torch.max(output_sof, 1)
        prob, pred = prob.item(), pred.item()
    if prob >= 0.55:
        pred = pred
    else:
        pred = 0
    return prob, pred


def get_numbers(numbers_box, model_cnn, model_state):
    numbers = []
    for img in numbers_box:
        img = only_digit(img)
        prob, pred = prediction(img, model_cnn, model_state)
        numbers.append(pred)
    numbers = np.array(numbers).reshape(9, 9)
    return numbers


def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img



def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2RGB)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2RGB)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver
