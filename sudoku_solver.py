""" Example
import numpy as np

sudoku_ = np.array([[1, 8, 6, 0, 0, 0, 0, 0, 5],
                   [0, 0, 5, 0, 0, 0, 1, 0, 4],
                   [0, 9, 0, 0, 0, 1, 0, 0, 0],
                   [0, 5, 0, 1, 0, 2, 0, 8, 0],
                   [0, 0, 1, 4, 0, 0, 7, 0, 0],
                   [8, 0, 0, 5, 9, 7, 6, 1, 0],
                   [2, 1, 0, 0, 4, 0, 0, 0, 6],
                   [6, 0, 0, 9, 0, 0, 0, 4, 0],
                   [5, 0, 0, 7, 1, 0, 0, 3, 8]])
"""


def solve(sudoku):
    find = find_empty(sudoku)
    if not find:
        return True
    else:
        y, x = find
    for n in range(1, 10):
        if is_possible(sudoku, y, x, n):
            sudoku[y][x] = n
            if solve(sudoku):
                return True
            sudoku[y][x] = 0
    return False


def find_empty(sudoku):
    for y in range(9):
        for x in range(9):
            if sudoku[y][x] == 0:
                return y, x  # row, col
    return None


def is_possible(sudoku, y, x, n):
    for i in range(9):  # check row=y
        if sudoku[y][i] == n and y != i:
            return False
    for i in range(9):  # check col=x
        if sudoku[i][x] == n and x != i:
            return False
    x0 = (x // 3) * 3  # check box (local coord)
    y0 = (y // 3) * 3
    for i in range(3):
        for j in range(3):
            if sudoku[y0 + i][x0 + j] == n and (i, j) != (y, x):
                return False
    return True



