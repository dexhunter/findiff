"""Peak traced memory of Gradient.__call__ on a 3D scalar field.

Imports and the input field are allocated before tracing. One warmup
call, then one traced call. Prints tracemalloc peak in bytes.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import tracemalloc

import numpy as np

from findiff import Gradient


def main():
    field = np.random.default_rng(0).standard_normal((96, 96, 96))
    gradient = Gradient(h=(1.0, 1.0, 1.0), acc=2)
    gradient(field)
    gc.collect()
    tracemalloc.start()
    gradient(field)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(peak)


if __name__ == "__main__":
    main()
