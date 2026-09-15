"""Prevent nested BLAS threading inside the benchmark's process workers.

Python imports ``sitecustomize`` before executing a script.  Four basin workers should use
four cores in total, not four cores per worker.  Set LOWFLOW_ALLOW_BLAS_THREADS=1 to opt out.
"""

import os

if os.environ.get("LOWFLOW_ALLOW_BLAS_THREADS") != "1":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
