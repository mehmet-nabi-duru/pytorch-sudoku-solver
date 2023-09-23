import cv2
import numpy as np
import operator

class ImagePreprocessor:
    def __init__(self, blur_size=9):
        self.blur_size = blur_size

    def preprocess(self, image, remove_lines=False, line_ksize=40):
        # convert image to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blur image for smoothing
        blur = cv2.GaussianBlur(grayscale, (self.blur_size, self.blur_size), 0)
        # adaptive thresholding to create a binary image
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.blur_size, 2)
        # inverting that image for model
        inverted = ~thresh

        # morphological opening for removing small particles like dots or moire lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        # dilating to increase border size
        result = cv2.dilate(morph, kernel, iterations=1)
        return result

    def find_contours(self, img, original):
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
            top_left = self.find_extreme_corners(polygon, min, np.add)  # has smallest (x + y) value
            top_right = self.find_extreme_corners(polygon, max, np.subtract)  # has largest (x - y) value
            bot_left = self.find_extreme_corners(polygon, min, np.subtract)  # has smallest (x - y) value
            bot_right = self.find_extreme_corners(polygon, max, np.add)  # has largest (x + y) value

            # if its not a square, we don't want it
            if bot_right[1] - top_right[1] == 0:
                return []
            if not (0.95 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.05):
                return []

            cv2.drawContours(original, [polygon], 0, (0, 0, 255), 3)

            # draw corresponding circles
            [self.draw_extreme_corners(x, original) for x in [top_left, top_right, bot_right, bot_left]]

            return [top_left, top_right, bot_right, bot_left]

        return []

        



    def split_into_cells(self, image):
        width, height = image.shape

        cell_height = height // 9
        cell_width = width // 9

        cells = []

        for i in range(9):
            for j in range(9):
                cell = image[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
                cells.append(cell)

        return cells

    

    def clean_cells(self, cells):
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
            new_img, is_number = self.clean_helper(cell)

            if is_number:
                new_img = cv2.resize(new_img, (28,28))
                cleaned_cells.append(new_img)
                i+=1
            else:
                cleaned_cells.append(0)

        return cleaned_cells
    
    
    def warp_image(self, image, corners):
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
    
    def get_grid_lines(self, img, length=10):
        horizontal = self.grid_line_helper(img, 1, length)
        vertical = self.grid_line_helper(img, 0, length)
        return vertical, horizontal
    
    def create_grid_mask(self, vertical, horizontal):
        grid = cv2.add(horizontal,vertical)
        grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
        grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=2)
        pts = cv2.HoughLines(grid, .2, np.pi / 90, 200)
        lines = self.draw_lines(grid, pts)
        mask = cv2.bitwise_not(lines)
        return mask
    

    # helpers

    def find_extreme_corners(self, polygon, limit_fn, compare_fn):
        # if we are trying to find bottom left corner, we know that it will have the smallest (x - y) value
        section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in polygon]),
                          key=operator.itemgetter(1))
        return polygon[section][0]

    def draw_extreme_corners(self, pts, original):
        cv2.circle(original, tuple(pts), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def grid_line_helper(self, img, shape_location, length):
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
    
    def draw_lines(self, img, lines:np.ndarray):
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
    
    def clean_helper(self, img):
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