import numpy as np

sudoku = np.array([[1, 8, 6, 0, 0, 0, 0, 0, 5],
                   [0, 0, 5, 0, 0, 0, 1, 0, 4],
                   [0, 9, 0, 0, 0, 1, 0, 0, 0],
                   [0, 5, 0, 1, 0, 2, 0, 8, 0],
                   [0, 0, 1, 4, 0, 0, 7, 0, 0],
                   [8, 0, 0, 5, 9, 7, 6, 1, 0],
                   [2, 1, 0, 0, 4, 0, 0, 0, 6],
                   [6, 0, 0, 9, 0, 0, 0, 4, 0],
                   [5, 0, 0, 7, 1, 0, 0, 3, 8]])


def posible(sudoku, y, x, n):
    for i in range(9):
        if sudoku[y][i] == n:
            return False
    for i in range(9):
        if sudoku[i][x] == n:
            return False
    x0 = (x // 3) * 3  # local coord
    y0 = (y // 3) * 3
    for i in range(3):
        for j in range(3):
            if sudoku[y0 + i][x0 + j] == n:
                return False
    return True


def solve(sudoku):
    for y in range(9):
        for x in range(9):
            if sudoku[y][x] == 0:
                for n in range(1, 10):
                    if posible(sudoku, y, x, n):
                        sudoku[y][x] = n
                        solve(sudoku)
                        sudoku[y][x] = 0
                return
    print(sudoku)



