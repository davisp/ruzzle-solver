#!/usr/bin/env python

import threading
import time

import cv2
import numpy as np
import pytesseract


OCR_SIZE = (500, 500)
TESSERACT_CONFIG="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEBUG = 0

class BoardExtractor(object):
    def __init__(self, frame):
        # Constants
        self.shape = frame.shape
        self.target = self.get_rect(frame.shape, 0.75)
        self.hull_boundary = self.get_rect(frame.shape, 0.9)

        self.masked = None

        self.letters = []
        self.modifiers = []

        self.lock = threading.Lock()
        self.processing = False
        self.ocr_result = None

        # Hackish initialization for OpenCV windows to
        # show on top with focus
        window = cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    def run(self, cap):
        global DEBUG
        
        prev_result = None
        
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

            if DEBUG > 0:
                fps = 1 / (time.time() - start_time)
                cv2.putText(camera, "Debug: %d" % DEBUG, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(camera, "%0.2f" % fps, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            cv2.imshow('Camera', camera)
            cv2.setWindowProperty('Camera', cv2.WND_PROP_TOPMOST, 1)

            with self.lock:
                if self.ocr_result is not None:
                    print(self.ocr_result)
                    if prev_result is None:
                        prev_result = self.ocr_result
                    elif prev_result == self.ocr_result:
                        cv2.destroyWindow('Camera')
                        cv2.waitKey(1)
                        return prev_result
                    else:
                        prev_result = self.ocr_result

            c = cv2.waitKey(1)
            if c < 0:
                continue
            if c == 27:
                break
            if c == 68 or c == 100:
                DEBUG = (DEBUG + 1) % 4

    def process(self, source):
        gray = self.preprocess(source)

        contours = self.find_contours(gray)
        if not contours:
            return

        if DEBUG >= 1:
            img = gray.copy()
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for c in contours:
                cv2.drawContours(img, [c], 0, (0, 255, 0), 2)
            cv2.imshow("Contours", img)
        else:
            cv2.destroyWindow("Contours")

        hull = self.find_board_hull(contours)
        if hull is None:
            return
        
        # UI Feedback step
        self.draw_mask(hull)

        with self.lock:
            processing = self.processing

        gray_board = self.extract_board(gray, hull, OCR_SIZE)
        letters = self.extract_letters(gray_board)

        if DEBUG >= 2:
            cv2.imshow('Gray Board', gray_board)
            cv2.imshow('Letters', letters)
        else:
            cv2.destroyWindow('Gray Board')
            cv2.destroyWindow('Letters')

        color_board = self.extract_board(source, hull, OCR_SIZE)
        modifiers = self.extract_modifiers(color_board)

        if processing and len(self.letters) >= 20:
            return

        self.letters.append(letters)
        if modifiers is not None:
            self.modifiers.append(modifiers)

        if not processing and len(self.letters) == 20:
            self.start_ocr()

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
    
        if DEBUG >= 3:
            cv2.imshow("Filled Board", threshed)
        else:
            cv2.destroyWindow("Filled Board")
    
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
        circles = self.get_circles(board)
        if circles is None:
            return

        hsv_board = self.get_hsv_masked(board)

        modifiers = []
        for (tl, br) in self.tile_corners(board):
            # Only look for circles in the top left corner
            # of letter tiles
            new_br_x = tl[0] + (br[0] - tl[0]) // 2
            new_br_y = tl[1] + (br[1] - tl[1]) // 2
            br = (new_br_x, new_br_y)

            circle = None
            for (cx, cy, r) in circles[0]:
                if self.inside_rect(cx, cy, (tl, br)):
                    if circle is None:
                        circle = (cx, cy, r)
                    else:
                        circle = None
                        break

            if circle is None:
                modifiers.append("")
                continue
            
            modifier = self.classify_modifier(hsv_board, circle)
            if modifier is None:
                modifiers.append("")
                continue
            
            modifiers.append(modifier)

        return modifiers

    def get_circles(self, board):
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        hough_args = {
            "param1": 100,
            "param2": 15,
            "minRadius": 10,
            "maxRadius": 30
        }
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 25, **hough_args)
        if circles is None:
            return
        return np.int32(np.around(circles))

    def get_hsv_masked(self, board):
        hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 90, 90), (180, 255, 255))
        return cv2.bitwise_and(hsv, hsv, mask=mask)

    def classify_modifier(self, board, circle):
        (h, w) = board.shape[:2]
        (cx, cy, r) = circle

        tl = (max(cx - r, 0), max(cy - r, 0))
        br = (min(cx + r, w), max(cy + r, 0))

        cont = cv2.ellipse2Poly((cx, cy), (r, r), 0, 0, 360, 10)
        mask = np.zeros(board.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [cont], 0, 255, -1)
        hist = cv2.calcHist([board], [0], mask, [180], [0, 180])

        # Zero out zero hue value. This is a bit awkward because
        # technically 0 is bright red. However, the thresholded HSV
        # mask has to set pixels to some value and that's zero. By
        # not ignoring it we end up picking red quite often due
        # to the circle mask not lining up exactly.
        hist[0] = 0

        if np.sum(hist) < 100:
            # Not enough pixels to sustain a valid classification
            return

        hue = np.argmax(hist)
        if hist[hue] == 0:
            return
    
        if hue < 5:
            return "3W"
        elif hue > 10 and hue < 20:
            return "2W"
        elif hue > 40 and hue < 80:
            return "2L"
        elif hue > 100 and hue < 140:
            return "3L"
        elif hue > 165:
            return "3W"

    def start_ocr(self):
        combined = self.combine_letters()
        modifiers = list(self.modifiers[:])

        self.processing = True
        t = threading.Thread(
                target=self.do_ocr,
                args=(combined, modifiers),
                daemon=True
            )
        t.start()
        
        self.letters = []
        self.modifiers = []

    def do_ocr(self, combined, modifiers):
        data = pytesseract.image_to_string(combined, config=TESSERACT_CONFIG)
        
        found = {}
        for line in data.splitlines():
            line = ''.join(line.split())
            if len(line) != 16:
                continue
            found.setdefault(line, 0)
            found[line] += 1
        if not found:
            with self.lock:
                self.processing = False
                return

        (score, text) = sorted((s, t) for (t, s) in found.items())[-1]
        if score < 10:
            with self.lock:
                self.processing = False
                return

        found = [{} for _ in modifiers[0]]
        for row in modifiers:
            for idx, mod in enumerate(row):
                found[idx].setdefault(mod, 0)
                found[idx][mod] += 1

        modifiers = []
        for col in found:
            (score, mod) = sorted((s, m) for (m, s) in col.items())[-1]
            if score > 5:
                modifiers.append(mod)
            else:
                modifiers.append("")

        combined = [mod + char for (mod, char) in zip(modifiers, text)]
        combined = "".join(combined)

        with self.lock:
            self.processing = False
            self.ocr_result = combined

    def combine_letters(self):
        max_w = max(i.shape[1] for i in self.letters)
        total_h = sum(i.shape[0] for i in self.letters)
        combined = 255 * np.ones((total_h, max_w, 3), dtype="uint8")
        offset = 0
        for img in self.letters:
            combined[offset:offset+img.shape[0], 0:img.shape[1]] = img
            offset += img.shape[0]
        return combined

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

SCORES = {
    "A":  1, "B":  4, "C":  4, "D":  2, "E":  1, "F":  4,
    "G":  3, "H":  4, "I":  1, "J": 10, "K":  0, "L":  1,
    "M":  3, "N":  1, "O":  1, "P":  4, "Q": 10, "R":  1,
    "S":  1, "T":  1, "U":  2, "V":  0, "W":  4, "X":  0,
    "Y":  4, "Z":  0
}

class Letter(object):
    def __init__(self, char):
        self.char = char
        self.children = {}
        self.word = None

DICT = Letter(None)

def load_dict():
    print("Loading dictionary...")
    seen = set()
    #with open("/usr/share/dict/web2") as handle:
    with open("dictionary") as handle:
        for line in handle:
            line = line.strip().upper()
            if line in seen:
                continue
            seen.add(line)
            curr = DICT
            for c in line:
                if c not in curr.children:
                    curr.children[c] = Letter(c)
                curr = curr.children[c]
            curr.word = line

class Cell(object):
    def __init__(self, idx, c, mult, scope):
        if mult is not None and scope is None:
            raise RuntimeError("Invalid mult/scope", mult, scope)
        if mult is None and scope is not None:
            raise RuntimeError("Invalid mult/scope", mult, scope)
        self.coords = (idx // 4, idx % 4)
        self.char = c
        self.mult = mult
        self.scope = scope
        self.visited = False

    def __repr__(self):
        if self.mult:
            return "%d%s%s" % (self.mult, self.scope[0].upper(), self.char)
        else:
            return "  %s" % self.char

    def neighbors(self):
        i, j = self.coords
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                if x == 0 and y == 0:
                    continue
                if x + i >= 0 and x + i < 4 and y + j >= 0 and y + j < 4:
                    yield (x + i, y + j)

class Solution(object):
    def __init__(self, word, path, board):
        self.word = word
        self.path = path
        self.score = self.calculate(self.path, board)
    
    def __repr__(self):
        BOARD_TEMPLATE = """\
        * ---- * ---- * ---- * ---- *
        | %s | %s | %s | %s |
        * ---- * ---- * ---- * ---- *
        | %s | %s | %s | %s |
        * ---- * ---- * ---- * ---- *
        | %s | %s | %s | %s |
        * ---- * ---- * ---- * ---- *
        | %s | %s | %s | %s |
        * ---- * ---- * ---- * ---- *
        """
        args = ["    "] * 16
        for idx, p in enumerate(self.path):
            args[p[0] * 4 + p[1]] = " %2d " % (idx + 1)
        formatted = BOARD_TEMPLATE % tuple(args)
        return "%d %s\n%s" % (self.score, self.word, formatted)
    
    def display(self):
        info = 255 * np.ones((100, OCR_SIZE[0], 3), dtype="uint8")
        self.center_text(info, "%d - %s" % (self.score, self.word))

        board = 255 * np.ones((OCR_SIZE[1], OCR_SIZE[0], 3), dtype="uint8")
        for (tl, br) in self.tile_corners(board):
            board = cv2.rectangle(board, tl, br, (0, 0, 0), 2)

        centers = list(self.tile_centers(board))
        for idx, p in enumerate(self.path):
            if idx == 0:
                color = (0, 255, 0)
            elif idx == len(self.word) - 1:
                color = (0, 0, 255)
            else:
                color = (0, 0, 0)
            label = "%d" % (idx + 1)
            (cx, cy) = centers[p[0] * 4 + p[1]]
            (tx, ty) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            tl = (cx - tx // 2, cy + ty // 2)
            cv2.putText(board, label, tl, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if idx > 0:
                self.add_arrow(board, self.path[idx - 1], p)

        display = np.concatenate((info, board), 0)
        cv2.imshow('Solution', display)
        cv2.setWindowProperty('Solution', cv2.WND_PROP_TOPMOST, 1)

        while True:
            c = cv2.waitKey(1)
            if c < 0:
                continue
            elif c == 27:
                exit(0)
            else:
                return
    
    def calculate(self, path, board):
        score = 0
        word_mult = 1
        for (i, j) in path:
            cell = board.data[i][j]
            if SCORES[cell.char] == 0:
                print("MISSING SCORE: %s" % cell.char)
            if cell.mult is not None and cell.scope == "letter":
                score += SCORES[cell.char] * cell.mult
            elif cell.mult is not None and cell.scope == "word":
                word_mult *= cell.mult
                score += SCORES[cell.char]
            else:
                score += SCORES[cell.char]

        score *= word_mult

        if len(path) > 4:
            score += 5 * (len(path) - 4)
        return score

    def center_text(self, img, text, color=(0, 0, 0)):
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        x = (img.shape[1] - textsize[0]) // 2
        y = (img.shape[0] + textsize[1]) // 2
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def add_arrow(self, img, start, end, color=(0, 0, 0), thickness=2, offset=30):
        centers = list(self.tile_centers(img))
        (sx, sy) = centers[start[0] * 4 + start[1]]
        (ex, ey) = centers[end[0] * 4 + end[1]]
        length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5

        if length <= 0:
            return
        
        dx = (ex - sx) / length
        dy = (ey - sy) / length

        start = (int(sx + dx * offset), int(sy + dy * offset))
        end = (int(ex - dx * offset), int(ey - dy * offset))
        
        return cv2.arrowedLine(img, start, end, color, thickness)
        

    def tile_corners(self, img):
        (h, w) = img.shape[:2]
        for j in range(0, 4):
            for i in range(0, 4):
                tl = (i * (w // 4), j * (h // 4))
                br = ((i + 1) * (w // 4) - 1, ((j + 1) * (h // 4)) - 1)
                yield (tl, br)

    def tile_centers(self, img):
        for (tl, br) in self.tile_corners(img):
            yield (tl[0] + (br[0] - tl[0]) // 2, tl[1] + (br[1] - tl[1]) // 2)

class Board(object):
    def __init__(self, desc):
        self.desc = desc
        self.data = self.parse(desc)
    
    def __repr__(self):
        rows = []
        for row in self.data:
            curr = []
            for col in row:
                curr.append("%s" % col)
            rows.append(" ".join(curr))
        return "\n".join(rows)

    def search(self):
        solutions = []
        for i in range(4):
            for j in range(4):
                seed = self.data[i][j]
                words = DICT.children[seed.char]
                for (word, path) in self.dfs(seed, words, [(i, j)]):
                    solutions.append(Solution(word, path, self))
        return solutions

    def dfs(self, cell, words, path):
        cell.visited = True
        if words.word is not None:
            if len(words.word) > 1:
                yield (words.word, path)
        for (x, y) in cell.neighbors():
            neighbor = self.data[x][y]
            if neighbor.visited:
                continue
            if neighbor.char in words.children:
                new_words = words.children[neighbor.char]
                new_path = path + [(x, y)]
                for solution in self.dfs(neighbor, new_words, new_path):
                    yield solution
        cell.visited = False

    def parse(self, desc):
        board = [
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None]
        ]
        i = 0
        desc = list(desc)
        while len(desc):
            if i > 15:
                raise RuntimeError("Invalid board description: %s" % desc)
            c = desc.pop(0)
            mult = None
            scope = None
            if "1" <= c <= "9":
                mult = int(c)
                scope = desc.pop(0)
                if scope.upper() == "L":
                    scope = "letter"
                elif scope.upper() == "W":
                    scope = "word"
                else:
                    raise RuntimeError("Invalid scope", scope)
                c = desc.pop(0)
            board[i // 4][i % 4] = Cell(i, c, mult, scope)
            i += 1

        if i < 16:
            raise RuntimeError("Description too short")

        return board

def main():
    load_dict()
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    ret, frame = cap.read()

    extractor = BoardExtractor(frame)
    descr = extractor.run(cap)
    #descr = "TL2LI2LCS3LETEU2WEARASOH"

    if descr is None:
        return

    window = cv2.namedWindow('Solution', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Solution', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty('Solution', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    board = Board(descr)
    solutions = board.search()
    solutions.sort(key=lambda k: k.score, reverse=True)
    seen = set()
    for s in solutions:
        if s.word in seen:
            continue
        seen.add(s.word)
        s.display()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
