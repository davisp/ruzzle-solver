#!/usr/bin/env python

import sys

import cv2
import numpy as np

import calamari_ocr.ocr as ocr
import calamari_ocr.ocr.voting
import calamari_ocr.proto

MODEL_ROOT="/Users/davisp/Data/calamari_models-1.0/antiqua_modern/"
MODEL_PATHS = [MODEL_ROOT + "%d.ckpt" % i for i in range(5)]
PREDICTOR = ocr.MultiPredictor(MODEL_PATHS)
VOTER = ocr.voting.voter_from_proto(calamari_ocr.proto.VoterParams())


def predict(imgs):
    for result in PREDICTOR.predict_raw(imgs, progress_bar=False):
        pred = VOTER.vote_prediction_result(result)
        print(pred)
        print(pred.sentence)


def process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    (h, w) = img.shape[:2]
    
    for x in range(w):
        cv2.floodFill(img, None, (x, 0), 255)
        cv2.floodFill(img, None, (x, h - 1), 255)
    
    for y in range(h):
        cv2.floodFill(img, None, (0, y), 255)
        cv2.floodFill(img, None, (w - 1, y), 255)

    images = []

    for j in range(0, 4):
        for i in range(0, 4):
            tl = (i * (w // 4), j * (h // 4))
            br = ((i + 1) * (w // 4), ((j + 1) * (h // 4)))
            mod_mask = [
                tl,
                (tl[0] + (w // 8), tl[1]),
                (tl[0], tl[1] + (h // 8))
            ]
            print(mod_mask)
            cv2.drawContours(img, [np.array(mod_mask)], 0, 255, -1)
            
            subimg = img[tl[1]:br[1], tl[0]:br[0]]
            images.append(subimg[::])


    predict(images)

    cv2.imwrite('stuff.png', img)

    return img

def main():
    if len(sys.argv) != 2:
        print("usage: %s image.png" % sys.argv[0])
        exit(1)

    frame = cv2.imread(sys.argv[1])

    window = cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Input', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty('Input', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    frame = process(frame)        
    cv2.imshow('Input', frame)
    cv2.setWindowProperty('Input', cv2.WND_PROP_TOPMOST, 1)

    while True:
        c = cv2.waitKey(1)
        if c < 0:
            continue
        if c == 27:
            break        
    
    #cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
