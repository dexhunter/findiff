"""Peak traced memory of utils.extend_to_ND.

The input is a dense 32x32 line operator, the shape compact finite
differences produce after inverting the implicit system, extended to a
32^3 grid along axis 0. The matrix is built before tracing. One warmup,
then one traced call. Prints tracemalloc peak in bytes.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import tracemalloc

import numpy as np
from scipy.sparse import csr_matrix

from findiff.utils import extend_to_ND


def main():
    grid = 32
    dense = np.random.default_rng(0).standard_normal((grid, grid))
    line = csr_matrix(dense)
    shape = (grid, grid, grid)
    extend_to_ND(line, 0, shape)
    gc.collect()
    tracemalloc.start()
    extend_to_ND(line, 0, shape)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(peak)


if __name__ == "__main__":
    main()
