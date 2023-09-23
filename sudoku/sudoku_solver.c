#include <Python.h>

// Function to set a bit at the specified row, column, and value in the Sudoku board
void setBit(int b[3][9], int row, int col, int k) {
    b[0][row] |= (1 << k); // Set the kth bit in the row
    b[1][col] |= (1 << k); // Set the kth bit in the column
    b[2][3 * (row / 3) + col / 3] |= (1 << k); // Set the kth bit in the subgrid
}

// Function to clear a bit at the specified row, column, and value in the Sudoku board
void clearBit(int b[3][9], int row, int col, int k) {
    b[0][row] &= ~(1 << k); // Clear the kth bit in the row
    b[1][col] &= ~(1 << k); // Clear the kth bit in the column
    b[2][3 * (row / 3) + col / 3] &= ~(1 << k); // Clear the kth bit in the subgrid
}

// Function for counting the number of zero bits in a number
int num_zero(int n) {
    int count = 0;
    for (int i = 1; i <= 9; i++) {
        if ((n & (1 << i)) == 0) {
            count++;
        }
    }
    return count;
}

// Recursive helper function to solve the Sudoku puzzle
int solveSudokuHelper(int a[9][9], int b[3][9]) {
    int bestSet = 0, bestSize = 9, bestRow = -1, bestCol = -1, bestSubgrid = -1;

    // Find the subgrid with the fewest empty cells
    for (int subgrid = 0; subgrid < 9; subgrid++) {
        int subgridRowStart = (subgrid / 3) * 3;
        int subgridColStart = (subgrid % 3) * 3;
        int emptyCells = 0;
        for (int i = subgridRowStart; i < subgridRowStart + 3; i++) {
            for (int j = subgridColStart; j < subgridColStart + 3; j++) {
                if (a[i][j] == 0) {
                    emptyCells++;
                }
            }
        }
        if (emptyCells == 0) {
            continue;  // Skip subgrid if it is already filled
        }
        if (emptyCells < bestSize) {
            bestSize = emptyCells;
            bestSubgrid = subgrid;
        }
    }

    if (bestSubgrid == -1) {
        return 1; // Sudoku puzzle is solved
    }

    int subgridRowStart = (bestSubgrid / 3) * 3;
    int subgridColStart = (bestSubgrid % 3) * 3;
    bestSize = 9;

    // Find the best cell in the subgrid to fill
    for (int i = subgridRowStart; i < subgridRowStart + 3; i++) {
        for (int j = subgridColStart; j < subgridColStart + 3; j++) {
            if (a[i][j] == 0) {
                int set = b[0][i] | b[1][j] | b[2][3 * (i / 3) + j / 3];
                int size = num_zero(set);
                if (size < bestSize) {
                    bestRow = i;
                    bestCol = j;
                    bestSet = set;
                    bestSize = size;
                    if (size == 0)
                        return 0; // No possible values to fill the cell
                    if (size <= 1)
                        goto done; // Found a cell with only one possible value
                }
            }
        }
    }

done:
    if (bestRow == -1)
        return 1; // Sudoku puzzle is solved
    for (int k = 1; k <= 9; k++) {
        if ((bestSet & (1 << k)) == 0) {
            a[bestRow][bestCol] = k; // Fill the cell with the value k
            setBit(b, bestRow, bestCol, k);
            if (solveSudokuHelper(a, b))
                return 1; // Found a solution
            clearBit(b, bestRow, bestCol, k);
        }
    }
    a[bestRow][bestCol] = 0; // Clear the cell
    return 0; // No solution found
}

// Python interface function to solve the Sudoku puzzle
static PyObject* solveSudoku(PyObject* self, PyObject* args) {
    PyObject* inputList;
    if (!PyArg_ParseTuple(args, "O", &inputList)) {
        return NULL;
    }

    int a[9][9];
    int b[3][9] = {{0}};

    // Convert the input Sudoku board from Python list to C array
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            PyObject* cell = PyList_GetItem(inputList, i * 9 + j);
            if (!PyLong_Check(cell)) {
                PyErr_SetString(PyExc_TypeError, "Invalid Sudoku input format");
                return NULL;
            }

            a[i][j] = PyLong_AsLong(cell);
            setBit(b, i, j, a[i][j]);
        }
    }

    // Solve the Sudoku puzzle
    if (solveSudokuHelper(a, b)) {

        // Convert the solved Sudoku board from C array to Python list
        PyObject* result = PyList_New(81);
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                PyObject* cell = PyLong_FromLong(a[i][j]);
                PyList_SetItem(result, i * 9 + j, cell);
            }
        }
        return result; // Return the solved Sudoku board
    }
    else {
        PyErr_SetString(PyExc_RuntimeError, "No solution exists for the given Sudoku.");
        return NULL;
    }
}

// Method definition for the Sudoku module
static PyMethodDef SudokuMethods[] = {
    {"solveSudoku", solveSudoku, METH_VARARGS, "Solve Sudoku puzzle given a list of 81 integers"},
    {NULL, NULL, 0, NULL}
};

// Module definition for the Sudoku module
static struct PyModuleDef sudokumodule = {
    PyModuleDef_HEAD_INIT,
    "sudoku38",
    "A module for solving sudoku",
    -1,
    SudokuMethods
};

// Module initializationfunction for the Sudoku module
PyMODINIT_FUNC PyInit_sudoku38(void) {
    return PyModule_Create(&sudokumodule); // Create and return the Sudoku module
}
