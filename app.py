from sudoku.sudoku import SudokuSolver
import cv2
from CNN.model.model import ModelFactory
import time
from typing import Type
from process.sudoku_drawer import SudokuDrawer
from process.image_preprocessor import ImagePreprocessor
from process.digit_recognizer import DigitRecognizer
from process.detections import GoodDetections

class SudokuCameraProcessor:
    def __init__(self, image_preprocessor:Type[ImagePreprocessor], 
                 digit_recognizer:Type[DigitRecognizer], 
                 sudoku_drawer:Type[SudokuDrawer], sudoku_solver):
        self.image_preprocessor = image_preprocessor
        self.digit_recognizer = digit_recognizer
        self.sudoku_drawer = sudoku_drawer
        self.sudoku_solver = sudoku_solver
        self.cap = cv2.VideoCapture(0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def run(self):
        detections = GoodDetections(maxlen = 5)#int(self.fps * 0.3)) # store the last 0.5 seconds' results
        
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = self.cap.read()

            start = time.perf_counter()

            preprocessed_frame = self.image_preprocessor.preprocess(frame)
            corners = self.image_preprocessor.find_contours(preprocessed_frame, frame)

            if corners:
                warped = self.image_preprocessor.warp_image(frame, corners)
                preprocessed_warped = self.image_preprocessor.preprocess(warped)

                vertical_lines, horizontal_lines = self.image_preprocessor.get_grid_lines(preprocessed_warped)
                mask = self.image_preprocessor.create_grid_mask(vertical_lines, horizontal_lines)
                numbers = cv2.bitwise_and(preprocessed_warped, mask)

                cells = self.image_preprocessor.split_into_cells(numbers)
                cells_processed = self.image_preprocessor.clean_cells(cells)
                digits = self.digit_recognizer.recognize_digits(cells_processed)

                sudoku_time = time.perf_counter()
                board = self.sudoku_solver.solve(digits)
                end = time.perf_counter()

                good_detections = detections.update(corners, digits, board)
                corners, grid, solution = detections.get_last_good_detection()

                end2 = time.perf_counter()

                solve_time = (end2 - sudoku_time)
                predict_time = end - start


                if solution is not None:
                    empty_cells = self.sudoku_drawer.empty_cell_digits(grid, solution)
                    warped_with_digits = self.sudoku_drawer.draw_digits(warped, empty_cells)
                    frame = self.sudoku_drawer.unwarp_image(warped_with_digits, frame, corners, predict_time, solve_time)
                    cv2.imshow("Original", frame)

                cv2.imshow("Warped screen", warped)

            cv2.imshow("Original", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    factory = ModelFactory()
    model = factory.load_model(model_path="CNN\model\model.pt")
    image_preprocessor = ImagePreprocessor(blur_size=11)
    sudoku_drawer = SudokuDrawer()
    sudoku_solver = SudokuSolver()
    digit_recognizer = DigitRecognizer()


    sudoku_camera_processor = SudokuCameraProcessor(image_preprocessor, digit_recognizer, sudoku_drawer, sudoku_solver)

    sudoku_camera_processor.run()

if __name__ == "__main__":
    main()