import sudoku.sudoku38 as sudoku38
from typing import List


class SudokuSolver:
    def __init__(self, solver=sudoku38):
        self.solver = solver
    
    def _validate_grid(self, grid:list) -> bool:
        """
        Validates if the grid is correct shape and values.

        Args:
            grid (list): A list of integers representing the Sudoku grid.

        Returns:
            bool: True if the grid is valid, False otherwise.
        """

        if len(grid) != 81:
            return False
        
        for cell in grid:
            if not isinstance(cell, int) or cell < 0 or cell > 9:
                return False
            
        return True

    def solve(self, grid:list) -> list:
        """
        Wrapper for sudoku38 module's solveSudoku method

        Args:
            grid (list): A list of integers representing unsolved sudoku grid. Empty cells represented by 0s.

        Returns:
            solved_grid (list): A list of integers representing solved grid.
        """

        if not self._validate_grid(grid):
            raise ValueError("Invalid Sudoku grid")
        

        try:
            solved_grid = self.solver.solveSudoku(grid)
            return solved_grid
        except (RuntimeError, Exception) as e:
            print("Failed to solve this board:", e)

        

def print_sudoku(board):
    nested_list = [board[i:i+9] for i in range(0, len(board), 9)]
    for i in range(len(nested_list)):
        if i % 3 == 0   and i != 0:
            print("-" * 30)
        for j in range(len(board[0])):
            if j % 3 == 0   and j != 0:
                print("| ", end="")
            if board[i][j] == 0:
                print("□", " ", end="")
            else:
                print(board[i][j], " ", end="")
        print()

