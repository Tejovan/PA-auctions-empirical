"""
parallel.py

Process-level parallelism inside one combination: independent starts and restarts
of a search, Nelder–Mead starts, bootstrap resamples. Run-level parallelism (one
process per combination) is in experiments/runner.py; it hands each combination
the workers it does not use itself (workers // number of tasks).

Items and the function must be picklable (module-level functions; threshold
functions are classes, threshold_functions.py and threshold_forms.FormThreshold).
"""

import os
from concurrent.futures import ProcessPoolExecutor

# Each worker computes with one BLAS thread; the processes are the parallelism.
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS")


def _single_threaded():
    for variable in _THREAD_VARIABLES:
        os.environ.setdefault(variable, "1")


def parallel_map(function, items, workers=1):
    """[function(item) for item in items], in up to `workers` processes; order kept.

    Serial (no processes) for workers <= 1 or a single item."""
    items = list(items)
    if workers is None or workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ProcessPoolExecutor(max_workers=min(workers, len(items)),
                             initializer=_single_threaded) as pool:
        return list(pool.map(function, items))


def split_workers(workers, num_items):
    """(processes for the items, workers left for each item)."""
    workers = max(1, workers or 1)
    outer = min(workers, max(num_items, 1))
    return outer, max(1, workers // outer)
