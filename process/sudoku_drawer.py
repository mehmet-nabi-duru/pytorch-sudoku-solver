import cv2
import numpy as np

class SudokuDrawer:
    def __init__(self):
        pass

    def draw_digits(self, warped, digits):
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

    def unwarp_image(self, warped_frame, original_frame, corners, detection_time, solving_time):
        corners = np.array(corners)
        height, width = warped_frame.shape[0], warped_frame.shape[1]
        
        corners_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]], dtype='float32')
        h, status = cv2.findHomography(corners_source, corners)

        unwarped_frame = cv2.warpPerspective(warped_frame, h, (original_frame.shape[1], original_frame.shape[0]))
        warped = cv2.warpPerspective(warped_frame, h, (original_frame.shape[1], original_frame.shape[0]))
        cv2.fillConvexPoly(original_frame, corners, 0, 16)
        frame = cv2.add(original_frame, warped)
        frame_height, frame_width = frame.shape[:2]
        overlay_text = f"Whole process took: {detection_time:.4f} seconds"
        cv2.putText(frame, overlay_text, (int(frame_width*0.05), int(frame_height*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Solving sudoku took: {solving_time:.4f} seconds", (int(frame_width*0.05), int(frame_width*0.06)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

    # Helper methods

    def empty_cell_digits(self, sudoku, solution):
        return [j if i == 0 else 0 for i, j in zip(sudoku, solution)]
        # return np.where(sudoku == 0, solution, 0).tolist()

    def display_images_in_grid(self, images):
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