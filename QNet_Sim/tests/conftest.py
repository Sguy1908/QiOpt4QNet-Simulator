"""Test-session setup.

OpenJij parallelises each sampler call with OpenMP. On the tiny QUBOs used in
the tests the threads mostly contend: one SQA test took about 300 s with the
default thread count and about 3 s with a single thread. The variable must be
set before any module that loads the OpenMP runtime is imported, which is why
it lives here. An explicit OMP_NUM_THREADS in the environment still wins.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
