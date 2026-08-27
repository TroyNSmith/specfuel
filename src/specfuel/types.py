"""specfuel types."""

import numpy as np

FLOAT_VECTOR = np.ndarray[tuple[int,], np.dtype[np.float64]]
INT_MATRIX = np.ndarray[tuple[int, int], np.dtype[np.int64]]
