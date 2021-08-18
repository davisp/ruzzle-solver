#!/usr/bin/env python

import threading
import time

import cv2
import numpy as np
import pytesseract


OCR_SIZE = (500, 500)
TESSERACT_CONFIG="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEBUG = True

class BoardExtractor(object):
    def __init__(self, frame):
        # Constants
        self.shape = frame.shape
        self.target = self.get_rect(frame.shape, 0.75)
        self.hull_boundary = self.get_rect(frame.shape, 0.9)

        self.masked = None

        # Hackish initialization for OpenCV windows to
        # show on top with focus
        window = cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    def run(self, cap):
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if frame.shape != self.shape:
                raise RuntimeError("Image capture changed sizes!")

            # Clear mask so it disappears in the UI if we
            # fail to detect one.
            self.masked = None
            self.process(frame)

            camera = cv2.rectangle(frame, self.target[0], self.target[1], (0, 255, 0), 1)
            if self.masked is not None:
                camera = cv2.addWeighted(camera, 0.8, self.masked, 0.2, 0.0)

            fps = 1 / (time.time() - start_time)
            cv2.putText(camera, "%0.2f" % fps, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            cv2.imshow('Camera', camera)
            cv2.setWindowProperty('Camera', cv2.WND_PROP_TOPMOST, 1)

            c = cv2.waitKey(1)
            if c < 0:
                continue
            if c == 27:
                break

    def process(self, source):
        gray = self.preprocess(source)

        contours = self.find_contours(gray)
        if not contours:
            return

        hull = self.find_board_hull(contours)
        if hull is None:
            return
        
        # UI Feedback step
        self.draw_mask(hull)

        gray_board = self.extract_board(gray, hull, OCR_SIZE)
        letters = self.extract_letters(gray_board)

        color_board = self.extract_board(source, hull, OCR_SIZE)
        modifiers = self.extract_modifiers(color_board)


    def preprocess(self, source):
        img = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 1)
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(img, kernel)
        return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def find_contours(self, img):
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # I should figure out wth is going on with this line...
        contours = contours[0] if len(contours) == 2 else contours[1]
    
        def contour_filter(c):
            M = cv2.moments(c)
            if M['m00'] < 5000:
                return False
    
            c_x = M['m10'] // M['m00']
            c_y = M['m01'] // M['m00']
    
            if not self.inside_rect(c_x, c_y, self.target):
                return False
            
            return True

        return list(filter(contour_filter, contours))

    def find_board_hull(self, contours):
        raw_hull = cv2.convexHull(np.vstack(list(c for c in contours)))
        epsilon = 0.1 * cv2.arcLength(raw_hull, True)
        hull = cv2.approxPolyDP(raw_hull, epsilon, True)
        hull.resize(4, 2)

        if len(hull) != 4:
            return
    
        if not self.hull_in_rect(hull, self.hull_boundary):
            return

        return hull

    def draw_mask(self, hull):
        self.masked = np.zeros((self.shape[0], self.shape[1], 3), dtype = "uint8")
        self.masked = cv2.drawContours(self.masked, [hull], 0, (0, 255, 0), -1)

    def extract_board(self, img, hull, size):
        rect = np.zeros((4, 2), dtype = "float32")
    
        # Points are arranged counter clockwise starting from
        # the top left ending on the top right
    
        # the top left point has the smallest sum whereas the
        # bottom right has the largest sum
        s = hull.sum(axis = 1)
        rect[0] = hull[np.argmin(s)]
        rect[2] = hull[np.argmax(s)]
    
        # Compute the difference between the points, the top right
        # will have the maximum difference and the bottom left will
        # have the minimum difference
        diff = np.diff(hull, axis = 1)
        rect[1] = hull[np.argmax(diff)]
        rect[3] = hull[np.argmin(diff)]
    
        dest = np.float32([
            [0, 0],
            [0, size[1] - 1],
            [size[0] - 1, size[1] - 1],
            [size[0] - 1, 0]
        ])
        
        matrix = cv2.getPerspectiveTransform(rect, dest)
        return cv2.warpPerspective(img, matrix, size, flags=cv2.INTER_LINEAR)

    def extract_letters(self, board):
        threshed = cv2.threshold(board, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        (h, w) = threshed.shape[:2]
        
        for (tl, br) in self.tile_corners(threshed):
            self.flood_fill_from_border(threshed, tl, br)
    
        letters = []
        for (tl, br) in self.tile_corners(threshed):
            subimg = threshed[tl[1]:br[1], tl[0]:br[0]]
            letters.append(subimg[::])
        
        horizontal = np.zeros((letters[0].shape[0], letters[0].shape[1] * len(letters)), dtype=np.uint8)
        offset = 0
        for img in letters:
            horizontal[0:h // 4, offset:offset + img.shape[1]] = img
            offset += img.shape[1]

        # Squeeze letters closer together to improve OCR
        # quality. The somewhat unintuitive rotation here
        # is for performance. Iterating over x and then y to
        # check each column took roughly 400ms per frame.
        cols = [0] * horizontal.shape[1]
        test = cv2.rotate(horizontal, cv2.ROTATE_90_CLOCKWISE)
        for y in range(test.shape[0]):
            if (test[y] == 255).all():
                cols[y] = 1

        # Collapse white gaps right to left
        # to simplify column deletions. 
        i = len(cols) - 1
        while i >= 0:
            if cols[i] == 0:
                i -= 1
                continue
            j = i - 1
            while j > 0 and cols[j] == 1:
                j -= 1
            # If we don't have at least 10
            # columns we skip squashing the gap
            if j > i - 10:
                i = j;
                continue
            # Otherwise delete all but 10 columns
            horizontal = np.delete(horizontal, slice(j, i - 10), 1)
            i = j - 1

        # Returning "color" images here for pytesseract's benefit
        return cv2.cvtColor(horizontal, cv2.COLOR_GRAY2BGR)

    def extract_modifiers(self, board):
        modifiers = self.possible_modifiers(board)
        if not modifiers:
            return
        return

        scores = []
        for (tl, br) in self.tile_corners(board):
            # Look for a detected modifier in the top
            # left quadrant of the letter tile
            new_br_x = tl[0] + (br[0] - tl[0]) // 2
            new_br_y = tl[1] + (br[1] - tl[1]) // 2
            br = (new_br_x, new_br_y)
            found = None
            for (mod_x, mod_y) in modifiers:
                if self.inside_rect(mod_x, mod_y, (tl, br)):
                    if found is None:
                        found = modifiers[(mod_x, mod_y)]
                    else:
                        # Multiple detections for the same tile invalidate
                        # all detections.
                        found = None
                        break
            if found:
                scores.append(found)
            else:
                scores.append("")

    def possible_modifiers(self, color_board):
        hsv = cv2.cvtColor(color_board, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(color_board, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        hough_args = {
            "param1": 100,
            "param2": 15,
            "minRadius": 1,
            "maxRadius": 30
        }
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 25, **hough_args)

        if circles is None:
            return

        modifiers = {}
        circles = np.uint16(np.around(circles))
        for idx, c in enumerate(circles[0]):
            center = (c[0], c[1])
            radius = c[2]
    
            cont = cv2.ellipse2Poly(center, (radius, radius), 0, 0, 360, 10)
            blank = np.zeros_like(gray)
            cv2.drawContours(blank, [cont], 0, 255, -1)
            points = np.where(blank == 255)
    
            total = 0
            count = 0
    
            for (hue, sat, val) in hsv[points[0], points[1]]:
                pixels += 1
                if sat >= 110 and val >= 110:
                    total += hue * 2
                    count += 1
    
            if count < 25:
                continue
    
            hue = total / count
            if hue > 30 and hue < 40:
                modifiers[center] = "2W"
            elif hue > 95 and hue < 115:
                modifiers[center] = "2L"
            elif hue > 210 and hue < 220:
                modifiers[center] = "3L"
            elif hue > 275 and hue < 285:
                modifiers[center] = "3W"
        
        return modifiers

    def get_rect(self, shape, percent):
        (h, w) = shape[:2]
        c_x, c_y = w // 2, h // 2

        dim = min(h, w)
        length = int(dim * percent)

        tl = (c_x - length // 2, c_y - length // 2)
        br = tl[0] + length, tl[1] + length

        return (tl, br)
        
    def inside_rect(self, x, y, rect):
        if x < rect[0][0] or x > rect[1][0]:
            return False
        if y < rect[0][1] or y > rect[1][1]:
            return False
        return True
        
    def hull_in_rect(self, hull, rect):
        for (x, y) in hull:
            if not self.inside_rect(x, y, rect):
                return False
        return True

    def tile_corners(self, img):
        (h, w) = img.shape[:2]
        for j in range(0, 4):
            for i in range(0, 4):
                tl = (i * (w // 4), j * (h // 4))
                br = ((i + 1) * (w // 4) - 1, ((j + 1) * (h // 4)) - 1)
                yield (tl, br)

    def flood_fill_from_border(self, img, tl, br):
        for x in range(tl[0], br[0]):
            if img[tl[1]][x] == 0:
                cv2.floodFill(img, None, (x, tl[1]), 255)
            if img[br[1]][x] == 0:
                cv2.floodFill(img, None, (x, br[1]), 255)
        
        for y in range(tl[1], br[1]):
            if img[y][tl[0]] == 0:
                cv2.floodFill(img, None, (tl[0], y), 255)
            if img[y][br[0]] == 0:
                cv2.floodFill(img, None, (br[0], y), 255)

def main():
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    ret, frame = cap.read()

    extractor = BoardExtractor(frame)
    extractor.run(cap)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
