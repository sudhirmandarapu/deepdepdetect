import numpy as np


class Model:
    @staticmethod
    def min_max_normalized(data):
        col_max = np.max(data, axis=0)
        col_min = np.min(data, axis=0)
        return np.divide(data - col_min, col_max - col_min)
