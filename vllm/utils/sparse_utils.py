# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Process-wide lock for sparse tensor validation.

PyTorch's ``torch.sparse.check_sparse_tensor_invariants()`` context manager
manipulates a **process-global** flag (save/enable/restore). When multiple
embedding-load operations run concurrently on a thread-pool executor, one
context can restore the flag to ``False`` while another thread is still inside
its guard, bypassing the invariant check.

All call sites that need sparse invariant checking MUST use
``sparse_invariants_checked()`` which serializes access behind a lock.
"""

import contextlib
import threading

import torch

_SPARSE_LOAD_LOCK = threading.Lock()


@contextlib.contextmanager
def sparse_invariants_checked():
    """Acquire the process-wide lock and enable sparse tensor invariant checks.

    This context manager wraps ``torch.sparse.check_sparse_tensor_invariants()``
    behind ``_SPARSE_LOAD_LOCK`` so concurrent callers cannot race the
    process-global save/restore flag.
    """
    with _SPARSE_LOAD_LOCK, torch.sparse.check_sparse_tensor_invariants():
        yield
