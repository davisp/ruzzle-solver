#!/usr/bin/env python

import cv2
import numpy as np


OCR_SIZE = (500, 500)


class BoardExtractor(object):
    def __init__(self, frame):
        # Constants
        self.shape = frame.shape
        self.target = self.get_rect(frame.shape, 0.75)
        self.hull_boundary = self.get_rect(frame.shape, 0.9)

        # Images for display
        self.source = frame
        self.treshed = None
        self.board = None
        self.masked = None

        # Various analysis data
        self.contours = []
        self.hull = None

        # Hackish initialization for OpenCV windows to
        # show on top with focus
        window = cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    def run(self, cap):
        while True:
            ret, frame = cap.read()
            if frame.shape != self.shape:
                raise RuntimeError("Image capture changed sizes!")

            self.source = frame
            self.process()
            self.show()

            c = cv2.waitKey(1)
            if c < 0:
                continue
            if c == 27:
                break

    def process(self):
        # Clear analysis state
        self.threshed = None
        self.board = None
        self.masked = None
        self.contours = []
        self.hull = None

        img = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 1)
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(img, kernel)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.threshed = img[::].copy()
    
        # Find Contours
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # I should figure out wth is going on with this line...
        contours = contours[0] if len(contours) == 2 else contours[1]
    
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] < 5000:
                continue
    
            c_x = M['m10'] // M['m00']
            c_y = M['m01'] // M['m00']
    
            if not self.inside_rect(c_x, c_y, self.target):
                continue
                
            self.contours.append(c)
    
        if not self.contours:
            return
    
        raw_hull = cv2.convexHull(np.vstack(list(c for c in self.contours)))
        epsilon = 0.1 * cv2.arcLength(raw_hull, True)
        self.hull = cv2.approxPolyDP(raw_hull, epsilon, True)
        self.hull.resize(4, 2)

        if len(self.hull) != 4:
            return
    
        if not self.hull_in_rect(self.hull, self.hull_boundary):
            return
    
        # Tracking mask
        self.masked = np.zeros((self.shape[0], self.shape[1], 3), dtype = "uint8")
        self.masked = cv2.drawContours(self.masked, [self.hull], 0, (0, 255, 0), -1)
    
        self.board = self.extract_board(img, self.hull, OCR_SIZE)

    def show(self):
        # Display Basic UI
        camera = self.source[::].copy()
        camera = cv2.rectangle(camera, self.target[0], self.target[1], (0, 255, 0), 1)

        if self.masked is not None:
            camera = cv2.addWeighted(camera, 0.8, self.masked, 0.2, 0.0)

        cv2.imshow('Camera', camera)
        cv2.setWindowProperty('Camera', cv2.WND_PROP_TOPMOST, 1)

        if self.threshed is not None:
            threshed = cv2.cvtColor(self.threshed, cv2.COLOR_GRAY2BGR)
            for c in self.contours:
                threshed = cv2.drawContours(threshed, [c], 0, (0, 255, 0), 1)
            cv2.imshow('Analyzed', threshed)

        if self.board is not None:
            cv2.imshow('Board', self.board)

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
