import sudoku38



def solve_sudoku(board):
    result = sudoku38.solveSudoku(board)
    if result is None:
        raise RuntimeError("Failed to solve this board")
    return result

def print_sudoku(board):
    for i in range(len(board)):
        if i % 27 == 0 and i != 0:
            print("\n-",end="")
            print("-" * 24)
        if i % 9 == 0 and i != 0:
            print()
        if i % 3 == 0 and i != 0:
            print("|", end=" ")
        if board[i] == 0:
            print("□", end="  ")
        else:
            print(board[i], end="  ")
        if (i + 1) % 9 == 0 and (i + 1) % 27 != 0:
            print("|", end=" ")
    print()

# input_board = [
#     3, 0, 6, 0, 4, 0, 7, 0, 0,
#     0, 0, 0, 0, 0, 8, 0, 1, 0,
#     0, 8, 9, 0, 7, 2, 0, 0, 0,
#     0, 3, 0, 0, 6, 0, 0, 0, 0,
#     8, 0, 2, 0, 0, 0, 0, 4, 1,
#     0, 0, 0, 0, 0, 4, 8, 0, 3,
#     0, 0, 0, 9, 0, 6, 0, 2, 0,
#     6, 2, 5, 4, 0, 7, 9, 0, 8,
#     0, 4, 8, 3, 0, 5, 1, 6, 7
# ]
input_board = [
    9, 0, 6, 0, 4, 5, 0, 8, 1,
    0, 0, 0, 0, 2, 0, 4, 3, 0,
    4, 0, 0, 8, 0, 1, 0, 2, 6,
    0, 6, 8, 9, 1, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 4, 0, 6, 3,
    0, 0, 0, 6, 0, 7, 1, 9, 0,
    6, 0, 0, 4, 7, 0, 3, 1, 2,
    2, 1, 0, 5, 0, 0, 6, 0, 0,
    0, 0, 4, 1, 0, 0, 0, 0, 0
    ]
# print_sudoku(input_board)

result = solve_sudoku(input_board)
for i in range(9):
    row = result[i * 9: (i + 1) * 9]
    print(row)

print_sudoku(result)