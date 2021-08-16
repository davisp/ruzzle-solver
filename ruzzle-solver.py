#!/usr/bin/env python

import cmd

SCORES = {
    "A":  1,
    "B":  4,
    "C":  4,
    "D":  2,
    "E":  1,
    "F":  4,
    "G":  0, # MISSING
    "H":  4,
    "I":  1,
    "J": 10,
    "K":  0, # MISSING
    "L":  1,
    "M":  3,
    "N":  1,
    "O":  1,
    "P":  4,
    "Q": 10,
    "R":  1,
    "S":  1,
    "T":  1,
    "U":  2,
    "V":  0, # MISSING
    "W":  4,
    "X":  0, # MISSING
    "Y":  4,
    "Z":  0  # MISSING
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
    with open("all_words") as handle:
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
    
    def calculate(self, path, board):
        score = 0
        word_mult = 1
        for (i, j) in path:
            cell = board.data[i][j]
            if SCORES[cell.char] == 0:
                print("MISSING SCORE VALUE HERE!")
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

class SolverShell(cmd.Cmd):
    intro = "Solve Ruzzles"
    prompt = "> "

    def default(self, line):
        try:
            board = Board(line.strip().upper())
            print(board)
        except Exception as e:
            print("Error parsing: '%s'" % line)
            print(e)
            return
        solutions = board.search()
        solutions.sort(key=lambda k: k.score, reverse=True)
        seen = set()
        for s in solutions:
            if s.word in seen:
                continue
            seen.add(s.word)
            print(s)
        
        print("\nFound %d solutions" % len(seen))


def main():
    load_dict()
    SolverShell().cmdloop()

if __name__ == "__main__":
    main()
