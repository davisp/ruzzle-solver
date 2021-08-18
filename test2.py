#!/usr/bin/env python

import cv2
import numpy as np


img = cv2.imread('cb2.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 25, param1=100, param2=15, minRadius=1, maxRadius=30)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for idx, c in enumerate(circles[0]):
        center = (c[0], c[1])
        radius = c[2]

        cont = cv2.ellipse2Poly(center, (radius, radius), 0, 0, 360, 10)
        blank = np.zeros_like(img)
        cv2.drawContours(blank, [cont], 0, 255, -1)
        points = np.where(blank == 255)

        total = 0
        count = 0

        for (hue, sat, val) in hsv[points[0], points[1]]:
            if sat >= 110 and val >= 110:
                total += hue * 2
                count += 1

        if count < 25:
            continue

        hue = total / count
        if hue > 30 and hue < 40:
            cv2.putText(img, "Yellow", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif hue > 95 and hue < 115:
            cv2.putText(img, "Green", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif hue > 210 and hue < 220:
            cv2.putText(img, "Blue", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif hue > 275 and hue < 285:
            cv2.putText(img, "Red", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        print("%d %r %f %r" % (idx, center, radius, total / count))

        cv2.circle(img, center, radius, (0, 0, 0), 1)
    # for i in circles[0, :]:
    #     center = (i[0], i[1])
    #     # circle center
    #     cv2.circle(img, center, 1, (0, 100, 100), 3)
    #     # circle outline
    #     radius = i[2]
    #     print (center, radius)
    #     cv2.circle(img, center, radius, (255, 0, 255), 1)

cv2.imshow("hsv", hsv)
cv2.imshow("detected circles", img)
cv2.waitKey(0)
