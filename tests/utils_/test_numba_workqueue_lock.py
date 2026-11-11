# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import sys
import textwrap

import pytest

# Runs in a subprocess: without the lock, numba's workqueue threading layer
# aborts the whole process ("Concurrent access has been detected").
_SCRIPT = textwrap.dedent(
    """
    from concurrent.futures import ThreadPoolExecutor

    import numba
    import numpy as np

    from vllm.utils.jit_monitor import numba_workqueue_threading_layer

    @numba.njit(parallel=True)
    def kernel(x):
        out = np.empty_like(x)
        for i in numba.prange(x.shape[0]):
            out[i] = np.sqrt(x[i]) + np.sin(x[i])
        return out

    x = np.random.default_rng(0).random(1 << 22)

    def run(_):
        with numba_workqueue_threading_layer():
            return kernel(x)

    run(None)  # compile
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(64)))
    assert all(np.array_equal(r, results[0]) for r in results)
    """
)


def test_numba_workqueue_threading_layer_serializes_threads():
    pytest.importorskip("numba")
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
