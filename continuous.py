#!/usr/bin/env python

import cv2
import numpy as np
import pytesseract

TESSERACT_CONFIG="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def ocr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    (h, w) = img.shape[:2]
    
    for x in range(w):
        if img[0][x] == 0:
            cv2.floodFill(img, None, (x, 0), 255)
        if img[h - 1][x] == 0:
            cv2.floodFill(img, None, (x, h - 1), 255)
    
    for y in range(h):
        if img[y][0] == 0:
            cv2.floodFill(img, None, (0, y), 255)
        if img[y][w - 1] == 0:
            cv2.floodFill(img, None, (w - 1, y), 255)

    images = []

    for j in range(0, 4):
        for i in range(0, 4):
            tl = (i * (w // 4), j * (h // 4))
            br = ((i + 1) * (w // 4), ((j + 1) * (h // 4)))
            mod_mask = [
                tl,
                (tl[0] + (w // 6), tl[1]),
                (tl[0], tl[1] + (h // 6))
            ]
            cv2.drawContours(img, [np.array(mod_mask)], 0, 255, -1)
            
            subimg = img[tl[1]:br[1], tl[0]:br[0]]
            images.append(subimg[::])

    cv2.imshow('Binary', img)

    blank = np.zeros((h // 4, w * 4), dtype=np.uint8)
    offset = 0
    for img in images:
        blank[0:h // 4, offset:offset + img.shape[1]] = img
        offset += img.shape[1]
    color = cv2.cvtColor(blank, cv2.COLOR_GRAY2RGB)
    cv2.imshow('Horizontal', color)
    
    cv2.imwrite('eh.png', color)
    text = pytesseract.image_to_string(color, config=TESSERACT_CONFIG)
    text = "".join(text.split())
    if len(text) == 16:
        print(text)


def process(img):
    cv2.imshow('original', img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.erode(img, kernel)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find Contours
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # I should figure out wth is going on with this line...
    contours = contours[0] if len(contours) == 2 else contours[1]

    square_threshold = 0.9
    min_area = 8000
    max_area = 25000

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    square_display = img[::].copy()
    sq_contours = []
    boxes = []

    # Sometimes this catches my finger as a box,
    # I might want to add a filter for white pixel
    # counts rather than just size and squareness
    # of the bounding box
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        square_display = cv2.rectangle(square_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        squareness = min(w, h) / max(w, h)
        if squareness < square_threshold:
            continue

        area = w * h
        if area < min_area or area > max_area:
            continue

        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        sq_contours.append(c)
        boxes.append((x, y, w, h))

    cv2.imshow("squares", square_display)

    if len(boxes) > 12:
        hull = cv2.convexHull(np.vstack(list(c for c in sq_contours)))
        epsilon = 0.1 * cv2.arcLength(hull, True)
        quad = cv2.approxPolyDP(hull, epsilon, True)
        if len(quad) == 4:
            quad = quad.reshape(4, 2)
            rect = np.zeros((4, 2), dtype = "float32")

            # Points are arranged counter clockwise starting from
            # the top left ending on the top right

            # the top-left point has the smallest sum whereas the
            # bottom-right has the largest sum
            s = quad.sum(axis = 1)
            rect[0] = quad[np.argmin(s)]
            rect[2] = quad[np.argmax(s)]

            # compute the difference between the points -- the top-right
            # will have the maximum difference and the bottom-left will
            # have the minimum difference
            diff = np.diff(quad, axis = 1)
            rect[1] = quad[np.argmax(diff)]
            rect[3] = quad[np.argmin(diff)]

            dest = np.float32([
                [0, 0],
                [0, 499],
                [499, 499],
                [499, 0]
            ])
            
            matrix = cv2.getPerspectiveTransform(rect, dest)
            out = cv2.warpPerspective(img, matrix, (500, 500), flags=cv2.INTER_LINEAR)
            cv2.imshow('Board', out)
            #cv2.imwrite('board.png', out)
            ocr(out)
            cv2.drawContours(img, [quad], 0, (0, 255, 0), 1)

    # if len(boxes) > 12:
    #     (h, w, _) = img.shape
    #     min_x, max_x = w, 0
    #     min_y, max_y = h, 0
    #     for (x, y, w, h) in boxes:
    #         min_x = min(min_x, x)
    #         max_x = max(max_x, x + w)
    #         min_y = min(min_y, y)
    #         max_y = max(max_y, y + h)
    #     print((min_x, max_x, min_y, max_y))
    #     subimg = img[min_y:max_y, min_x:max_x]
    #     subimg = cv2.copyMakeBorder(subimg, 20, 20, 20, 20, cv2.BORDER_CONSTANT, 0)
    #     cv2.imwrite("board.png", subimg)
    #     cv2.imshow('Board', subimg)
    #     #cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    return img

def main():
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    window = cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Input', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty('Input', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        frame = process(frame)
        
        cv2.imshow('Input', frame)
        cv2.setWindowProperty('Input', cv2.WND_PROP_TOPMOST, 1)

        c = cv2.waitKey(1)
        if c < 0:
            continue
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
