"""Time Stencil._system_matrix for a 5x5x5 generic 3D stencil.

Constructs the same offsets Stencil.__init__ would use and calls
_system_matrix directly. One warmup, then one measured call.
Prints wall time in milliseconds.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time

from findiff.stencils import Stencil


def _bare_stencil():
    stencil = Stencil.__new__(Stencil)
    stencil.ndims = 3
    stencil.max_order = 100
    stencil.offsets = [
        (i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)
    ]
    stencil.partials = {(2, 0, 0): 1, (0, 2, 0): 1, (0, 0, 2): 1}
    stencil.spacings = [1, 1, 1]
    return stencil


def main():
    stencil = _bare_stencil()
    stencil._system_matrix()
    start = time.perf_counter()
    stencil._system_matrix()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"{elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
