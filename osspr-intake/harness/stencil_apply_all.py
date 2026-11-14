"""Time StencilSet.apply_all on a 2D Laplacian stencil.

Setup (operator, grid, stencil build) is outside the timer. One warmup
call, then one measured call. Prints wall time in milliseconds.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time

import numpy as np

from findiff import Diff


def main():
    n = 200
    x = np.linspace(-1.0, 1.0, n)
    dx = float(x[1] - x[0])
    big_x, big_y = np.meshgrid(x, x, indexing="ij")
    field = big_x**3 + big_y**3
    operator = Diff(0, dx, 2) + Diff(1, dx, 2)
    stencil = operator.stencil(field.shape)

    stencil.apply_all(field)
    start = time.perf_counter()
    stencil.apply_all(field)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"{elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
