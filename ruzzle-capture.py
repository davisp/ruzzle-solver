#!/usr/bin/env python

import cv2
import numpy as np
import pytesseract

TESSERACT_CONFIG = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def process(img):
    # Crop out center square region
    (h, w, _) = img.shape
    min_dim = min(h, w)
    box_size = int(0.8 * min_dim)
    (cx, cy) = w // 2, h // 2
    left = cx - (box_size // 2)
    top = cy - (box_size // 2)
    img = img[top:top + box_size, left:left + box_size]

    # Basic de-noising
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #img = cv2.GaussianBlur(img, (5, 5), 1)

    return img

def add_target(img):
    (h, w, _) = img.shape

    # Horizontal lines
    for i in range(1, 4):
        l = (0, i * h // 4)
        r = (w, i * h // 4)
        img = cv2.line(img, l, r, (0, 255, 0), 2)
    
    # Vertical lines
    for i in range(1, 4):
        t = (i * w // 4, 0)
        b = (i * w // 4, h)
        img = cv2.line(img, t, b, (0, 255, 0), 2)

    return img

def ocr_boxes(img):
    (h, w, _) = img.shape
    for j in range(0, 4):
        for i in range(0, 4):
            tl = (i * w // 4, j * h // 4)
            br = ((i + 1) * w // 4, (j + 1) * h // 4)
            sub = img[tl[1]:br[1], tl[0]:br[0]]
            sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
            (sh, sw) = sub.shape

            # Crop out letters by blacking out rectangles
            rects = [
                ((0, 0), (sw - 1, sh // 6)),
                ((0, 0), (sw // 6, sh - 1)),
                ((5 * sw // 6, 0), (sw - 1, sh - 1)),
                ((0, 5 * sh // 6), (sw - 1, sh - 1))
            ]
            for rect in rects:
                cv2.rectangle(sub, rect[0], rect[1], 0, -1)
 
            # Crop out score modifier
            mod_triangle = np.array([(0, 0), (sw // 2, 0), (0, sh // 2)])
            cv2.fillPoly(sub, [mod_triangle], 0)

            # Basic denoising
            #sub = cv2.GaussianBlur(sub, (5, 5), 1)
            sub = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sub = cv2.filter2D(sub, -1, kernel)
            sub = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Flood fill outer black area            
            for x in range(0, sw):
                cv2.floodFill(sub, None, (x, 0), 255)
                cv2.floodFill(sub, None, (x, sh - 1), 255)
            for y in range(0, sh):
                cv2.floodFill(sub, None, (0, y), 255)
                cv2.floodFill(sub, None, (sw - 1, y), 255)

            cv2.imwrite("%d-%d.png" % (j, i), sub)

            boxes = pytesseract.image_to_boxes(sub, config=TESSERACT_CONFIG)
            boxes = boxes.strip().splitlines()
            if len(boxes) == 1:
                box = boxes[0].split(' ')
                ltl = (int(box[1]), h - int(box[2]))
                lbr = (int(box[3]), h - int(box[4]))
                sub = cv2.rectangle(sub, ltl, lbr, (0, 0, 255), 2)

            img[tl[1]:br[1], tl[0]:br[0]] = cv2.cvtColor(sub, cv2.COLOR_GRAY2RGB)

    return img

def main():
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    processing = False
    capture = False
    frame = None
    
    ret, frame = cap.read()
    window = cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Input', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty('Input', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        frame = process(frame)
        
        if processing:
            frame = ocr_boxes(frame)

        if capture:
            cv2.imwrite("capture.png", frame)
            capture = False

        frame = add_target(frame)
        cv2.imshow('Input', frame)
        cv2.setWindowProperty('Input', cv2.WND_PROP_TOPMOST, 1)

        c = cv2.waitKey(1)
        if c < 0:
            continue
        if c == 27:
            break
        if c == 32:
            processing = not processing
        if c == 99:
            capture = True
        
    
    #cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
