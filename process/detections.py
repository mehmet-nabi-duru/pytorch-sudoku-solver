import numpy as np
from collections import deque

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