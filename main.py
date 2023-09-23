from collections import deque
from sudoku.sudoku import SudokuSolver
import cv2
import numpy as np
from multiprocessing import Pool
import os
import operator
import time
import torch

start_import = time.perf_counter()
from CNN.model.model import ModelFactory
factory = ModelFactory()
model = factory.load_model(model_path="CNN\model\model.pt")


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("20230725_192900.mp4")



def warped_preprocess(image, blur_size=11):
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(grayscale, (blur_size, blur_size), 0)

    # Apply adaptive thresholding to create a binary image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Invert the image
    inverted = ~thresh

    # Detect lines in the image using Hough Line Transform
    lines = cv2.HoughLinesP(inverted, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a black image with the same dimensions as the original
    mask = np.zeros_like(inverted)

    # Draw the detected lines onto the mask
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Subtract the mask from the original image to remove the lines
    result = cv2.subtract(inverted, mask)

    return result

def preprocess(image, blur_size=9, remove_lines=False, line_ksize=40):
    # convert image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur image for smoothing
    blur = cv2.GaussianBlur(grayscale, (blur_size, blur_size), 0)
    # adaptive thresholding to create a binary image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blur_size, 2)
    # inverting that image for model
    inverted = ~thresh

    if remove_lines:
        # define horizontal line erosion
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_ksize,1))
        # erode and dilate the image to remove horizontal lines
        inverted = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel)
        # define vertical line erosion
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_ksize))
        # erode and dilate the image to remove vertical lines
        inverted = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel)



    # morphological opening for removing small particles like dots or moire lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    # dilating to increase border size
    result = cv2.dilate(morph, kernel, iterations=1)
    return result

def draw_extreme_corners(pts, original):
    cv2.circle(original, tuple(pts), 3, (255, 255, 0), cv2.FILLED)

def find_extreme_corners(polygon, limit_fn, compare_fn):
    # if we are trying to find bottom left corner, we know that it will have the smallest (x - y) value
    section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in polygon]),
                          key=operator.itemgetter(1))

    return polygon[section][0]

def find_contours(img, original):
    # find contours on thresholded image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort by the largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = None

    # make sure this is the one we are looking for
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, closed=True)
        num_corners = len(approx)

        if num_corners == 4 and area > 1000:
            polygon = approx
            break

    if polygon is not None:
        # find its extreme corners
        top_left = find_extreme_corners(polygon, min, np.add)  # has smallest (x + y) value
        top_right = find_extreme_corners(polygon, max, np.subtract)  # has largest (x - y) value
        bot_left = find_extreme_corners(polygon, min, np.subtract)  # has smallest (x - y) value
        bot_right = find_extreme_corners(polygon, max, np.add)  # has largest (x + y) value

        # if its not a square, we don't want it
        if bot_right[1] - top_right[1] == 0:
            return []
        if not (0.95 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.05):
            return []

        cv2.drawContours(original, [polygon], 0, (0, 0, 255), 3)

        # draw corresponding circles
        [draw_extreme_corners(x, original) for x in [top_left, top_right, bot_right, bot_left]]

        return [top_left, top_right, bot_right, bot_left]

    return []

def warp_image(image, corners):
    # sort the corners in the same order
    corners = sorted(corners, key=lambda x: x[0])
    top_left, bottom_left = sorted(corners[:2], key=lambda x: x[1])
    top_right, bottom_right = sorted(corners[2:], key=lambda x: x[1])

    # creating an array with the four corners of the sudoku grid in the order of:
    # top-left, top-right, bottom-right, bottom-left
    corner_arr = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

    # dimensions of the warped grid
    side = max([
        np.sqrt(((top_right[0]-top_left[0])**2)+((top_right[1]-top_left[1])**2)),
        np.sqrt(((bottom_right[0]-bottom_left[0])**2)+((bottom_right[1]-bottom_left[1])**2)),
        np.sqrt(((bottom_right[0]-top_right[0])**2)+((bottom_right[1]-top_right[1])**2)),
        np.sqrt(((bottom_left[0]-top_left[0])**2)+((bottom_left[1]-top_left[1])**2))
    ])

    dst = np.array([[0,0], [side-1,0], [side-1, side-1], [0, side-1]], dtype="float32")

    # get the perspective transform matrix
    transmat = cv2.getPerspectiveTransform(corner_arr, dst)

    warped = cv2.warpPerspective(image, transmat, (int(side), int(side)))

    return warped

def get_grid_lines(img, length=10):
    horizontal = grid_line_helper(img, 1, length)
    vertical = grid_line_helper(img, 0, length)
    return vertical, horizontal


def split_into_cells(image):
    height, width = image.shape

    cell_height = height // 9
    cell_width = width // 9

    cells = []

    for i in range(9):
        row = []
        for j in range(9):
            cell = image[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
            cells.append(cell)
            # row.append(cell)
    
    return cells

def save_cells(cells, folder="cells"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(81):
        # for j in range(9):
        filepath = os.path.join(folder, f"{i}.png")
        cv2.imwrite(filepath, cells[i])

def grid_line_helper(img, shape_location, length=10):
    clone = img.copy()
    row_or_col = clone.shape[shape_location]
    size = row_or_col // length

    if shape_location == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))

    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))

    clone = cv2.erode(clone, kernel)
    clone = cv2.dilate(clone, kernel)

    return clone


def draw_lines(img, lines:np.ndarray):
    clone = img.copy()
    lines = np.squeeze(lines)

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        x2 = int(x0 - 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        y2 = int(y0 - 1000 * a)

        cv2.line(clone, (x1, y1), (x2, y2), (255,255,255), thickness=4)
    
    return clone

def create_grid_mask(vertical, horizontal):
    grid = cv2.add(horizontal,vertical)
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=2)
    pts = cv2.HoughLines(grid, .2, np.pi / 90, 200)
    lines = draw_lines(grid, pts)
    mask = cv2.bitwise_not(lines)
    return mask

###################################################################


def clean_cells(cells):
    """
    Cleans and preprocesses a list of grayscaled cells

    Args:
        cells (list): List of the grayscaled images

    Returns:
        list: List of cleaned and preprocessed images
    """
    # start = time.perf_counter()
    cleaned_cells = []
    i = 0

    for cell in cells:
        new_img, is_number = clean_helper(cell)

        if is_number:
            new_img = cv2.resize(new_img, (28,28))
            cleaned_cells.append(new_img)
            i+=1
        else:
            cleaned_cells.append(0)

    return cleaned_cells

#####################################################################

def clean_helper(img):
    """
    Cleans and preprocess individual cell
    Args:
        img (np.ndarray): Grayscaled image of a single cell

    Returns:
        tuple: (cleaned_image, IS_NUMBER_FLAG)
    """

    # Check if image is mostly empty
    if np.isclose(img, 0).sum() / (img.shape[0] * img.shape[1]) >= 0.99:
        return np.zeros_like(img), False
    
    
    height, width = img.shape
    mid = width // 2

    # Check if 60% of the central width of the image is mostly black
    empty_60_percent = np.isclose(img[:, int(mid - width * 0.2):int(mid + width * 0.2)], 0).sum() / (2 * width * 0.2 * height) >= 0.90
    if empty_60_percent:
        return np.zeros_like(img), False
    
    # Find contours and sort them by the area
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Extract the bounding rectangle of the largest contour (assumed to be the digit)
    x, y, w, h = cv2.boundingRect(contours[0])
    start_x = (width - w) // 2
    start_y = (height - h) // 2

    # Create a new image and copy the digit region to the center of the new image
    new_img = np.zeros_like(img)
    new_img[start_y:start_y + h, start_x:start_x + w] = img[y:y + h, x:x + w]
    return new_img, True

def display_images_in_grid(images):
    # Create a blank canvas
    canvas_size = 9 * 28
    canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    
    for i, image in enumerate(images):
        row = i // 9  # Calculate the row index
        col = i % 9   # Calculate the column index

        if np.array_equal(image, np.zeros((28, 28), dtype=np.uint8)):
            continue  # Skip if it's a zero
        
        # Resize the image to fit in a 28x28 square
        resized_image = cv2.resize(image, (28, 28))

        # Calculate the top-left corner coordinates for the image
        top = row * 28
        left = col * 28

        # Place the image on the canvas
        canvas[top:top+28, left:left+28] = resized_image
    
    return canvas

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model



def recognize_digits(processed_cells, model):
    non_zero_elements = [(i, cell/255.0) for i, cell in enumerate(processed_cells) if type(cell) != int]
    # non_zero_elements = [(i, cell) for i, cell in enumerate(processed_cells) if type(cell) != int]
    non_zero_indices, non_zero_cells = zip(*non_zero_elements)

    non_zero_cells = np.array(non_zero_cells)
    non_zero_cells_tensor = torch.Tensor(non_zero_cells).unsqueeze(1)
    with torch.no_grad():
        output = model(non_zero_cells_tensor)
        _, preds = torch.max(output,1)
        preds = preds.tolist()

    preds_with_zeros = [0] * len(processed_cells)
    for index, pred in zip(non_zero_indices, preds):
        preds_with_zeros[index] = pred

    return preds_with_zeros






def empty_cell_digits(sudoku, solution):
    return [j if i == 0 else 0 for i, j in zip(sudoku, solution)]
    # return np.where(sudoku == 0, solution, 0).tolist()


def draw_digits(warped, digits):
    # Determine the size of each square 
    square_size = warped.shape[0] // 9
    # reshape the array for convenience
    digit_grid = np.reshape(digits, (9,9))
    digit_grid = digit_grid.astype("int32")
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row in range(9):
        for col in range(9):
            digit = digit_grid[row,col]

            if digit != 0:
                # calculate the diagonal corners of current cell
                p1 = col * square_size, row * square_size # top left
                p2 = (col + 1) * square_size, (row + 1) * square_size # bottom right
                # calculating the text origin
                center = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                text_size, _ = cv2.getTextSize(str(digit), font, square_size / 300, 2)
                text_org = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

                # draw the digit into image
                cv2.putText(warped, str(digit), text_org, font, square_size/55, (0,0,255),2)



    return warped



def unwarp_image(warped_frame, original_frame, corners, detection_time, solving_time):
    corners = np.array(corners)
    height, width = warped_frame.shape[0], warped_frame.shape[1]
    
    corners_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]], dtype='float32')
    h, status = cv2.findHomography(corners_source, corners)

    unwarped_frame = cv2.warpPerspective(warped_frame, h, (original_frame.shape[1], original_frame.shape[0]))
    warped = cv2.warpPerspective(warped_frame, h, (original_frame.shape[1], original_frame.shape[0]))
    # mask = np.zeros_like(original_frame)
    cv2.fillConvexPoly(original_frame, corners, 0, 16)
    # masked_frame = cv2.bitwise_and(original_frame, mask)
    # result_frame = cv2.add(masked_frame, unwarped_frame)
    frame = cv2.add(original_frame, warped)
    frame_height, frame_width = frame.shape[:2]
    overlay_text = f"Whole process took: {detection_time:.4f} seconds"
    cv2.putText(frame, overlay_text, (int(frame_width*0.05), int(frame_height*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f"Solving sudoku took: {solving_time:.4f} seconds", (int(frame_width*0.05), int(frame_width*0.06)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

###########################################################


# class DetectionResult:
#     def __init__(self, grid, solution, corners, confidence):
#         self.grid = grid
#         self.solution = solution
#         self.corners = corners
#         self.confidence = confidence

# class DetectionBuffer:
#     def __init__(self, maxlen=15, confidence_threshold=0.8):
#         self.buffer = deque(maxlen=maxlen)
#         self.confidence_threshold = confidence_threshold

#     def update(self, corners_detected, grid, solution):
#         # calculate the confidence score
#         grid, solution = np.array(grid), np.array(solution)
#         confidence = self.calculate_confidence(corners_detected, grid, solution)
#         result = DetectionResult(grid, solution, corners_detected, confidence)

#         self.buffer.append(result)

#     def calculate_confidence(self, corners_detected, grid, solution):
#         if solution is None:
#             return 0
#         # List of conditions and their corresponding scores
#         conditions = [
#         # (corners_detected, 1),  # If corners are detected
#         (solution is not None and np.all(solution != 0), 1),  # If solution is available and does not contain any 0s
#         (all(np.array_equal(solution, result.solution) for result in self.buffer), 1),  # If the solution is the same as the last N solutions
#         (all(np.mean(grid == result.grid) > 0.5 for result in self.buffer), 1),  # If the majority of unsolved grid is the same in the last N detections
#     ]

#         # Calculate the total score
#         score = sum(score for condition, score in conditions if condition)

#         # Normalize the score to get a confidence between 0 and 1
#         confidence = score / len(conditions)
#         return confidence
#     def get_average_confidence(self):
#         # calculating the average grid, solution, confidence
#         grids = [result.grid for result in self.buffer]
#         average_grid = np.mean(grids, axis=0)

#         solutions = [result.solution for result in self.buffer if result.solution is not None]
#         if solutions:  # Check if solutions list is not empty
#             solutions = np.array(solutions)
#             average_solution = np.mean(solutions, axis=0)
#         else:
#             average_solution = None

#         average_confidence = sum(result.confidence for result in self.buffer) / len(self.buffer)

#         return average_grid, average_solution, average_confidence


#     # def get_average_confidence(self):
#     #     average_grid = np.mean([result.grid for result in self.buffer], axis=0)
#     #     # Get the list of solutions that are not None
#     #     solutions = [result.solution for result in self.buffer if result.solution is not None]
#     #     # If there are no solutions, return None for the average solution and confidence
#     #     if not solutions:
#     #         return None, None, None
#     #     # Otherwise, calculate the average solution and confidence
#     #     average_solution = np.mean(solutions, axis=0)
#     #     average_confidence = sum(result.confidence for result in self.buffer if result.confidence is not None) / len(self.buffer)
#     #     return average_grid, average_solution, average_confidence


#     def should_display(self) -> bool:
#         _, _, average_confidence = self.get_average_confidence()
#         print(f"Average confidence: {average_confidence}")
#         return average_confidence > self.confidence_threshold


class GoodDetections:
    def __init__(self, maxlen=15):
        self.buffer = deque(maxlen=maxlen)

    def update(self, corners_detected, grid, solution):
        if corners_detected and solution is not None and np.all(solution != 0):
            # This is a good detection
            self.buffer.append((corners_detected, grid, solution))

    def get_last_good_detection(self):
        if self.buffer:
            # Return the most recent good detection
            return self.buffer[-1]
        else:
            # No good detections have been stored yet
            return None, None, None


###########################################################

solver = SudokuSolver()

# modelson = "./last_model.pt"
# print(f"{fps} frames")


fps = cap.get(cv2.CAP_PROP_FPS)
# buffer = DetectionBuffer(maxlen=int(fps*2))
good_detections = GoodDetections(maxlen=int(fps*0.5))
# model = load_model(modelson)
KSIZE = 15
while True:#cap.isOpened():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

    if not ret:
        print('no video')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    # frame = cv2.imread("sudokuu.jpg")

    start = time.perf_counter()


    preprocessed_frame = preprocess(frame)
    corners = find_contours(preprocessed_frame,frame)

    if corners:
        warped = warp_image(frame, corners)
        preprocessed_warped = preprocess(warped)


        vertical_lines, horizontal_lines = get_grid_lines(preprocessed_warped)
        mask = create_grid_mask(vertical_lines, horizontal_lines)
        numbers = cv2.bitwise_and(preprocessed_warped, mask)

        
        cells = split_into_cells(numbers)
        cells_processed = clean_cells(cells)
        detect_time = time.perf_counter()
        preds = recognize_digits(cells_processed, model)

        end = time.perf_counter()
        sudoku_time = time.perf_counter()
        board = solver.solve(preds)#sudoku_solver(preds)

        good_detections.update(corners, preds, board)
        corners, grid, solution = good_detections.get_last_good_detection()
        
        end2 = time.perf_counter()
        solve_time = (end2 - sudoku_time)
        predict_time = end - start
        
        if solution is not None:
            # We have a good detection to display
            empty_cells = empty_cell_digits(grid, solution)
            warped_with_digits = draw_digits(warped, empty_cells)
            cv2.imshow("digited warped", warped_with_digits)

            frame = unwarp_image(warped_with_digits, frame, corners, predict_time, solve_time)
            cv2.imshow("Original", frame)
        






        # # buffer.update(corners, preds, board)

        
        # # if buffer.should_display():
        # #     average_grid, average_solution, average_confidence = buffer.get_average_confidence()
        # #     if average_solution is not None:
        # #         average_solution = average_solution.tolist()
        # #         empty_cells = empty_cell_digits(preds, average_solution)
        # #     else:
        # #         empty_cells = empty_cell_digits(preds, board)

        # # else:
        # #     empty_cells = empty_cell_digits(preds, board)
            
        # # if empty_cells is not None:
        # #     warped_with_digits = draw_digits(warped, empty_cells)
        # #     cv2.imshow("digit warp", warped_with_digits)
        # #     frame = unwarp_image(warped_with_digits, frame, corners, predict_time, solve_time)








        
        if type(board)==list:#and 0 not in board:
            empty_cells = empty_cell_digits(preds, board)
            warped_with_digits = draw_digits(warped, empty_cells)
            cv2.imshow("digited warped", warped_with_digits)

            frame = unwarp_image(warped_with_digits, frame, corners, predict_time, solve_time)
            
            # cv2.imshow("Overlayed", result)
            # frame = result



    
            




        cv2.imshow("Warped screen", warped,)
        
        

    cv2.imshow("Original", frame)

    # break



cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()