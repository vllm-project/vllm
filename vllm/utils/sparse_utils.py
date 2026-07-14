# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Process-wide lock for sparse tensor validation.

PyTorch's ``torch.sparse.check_sparse_tensor_invariants()`` context manager
manipulates a **process-global** flag (save/enable/restore). When multiple
embedding-load operations run concurrently on a thread-pool executor, one
context can restore the flag to ``False`` while another thread is still inside
its guard, bypassing the invariant check.

All call sites that use this context manager MUST acquire
``_SPARSE_LOAD_LOCK`` first to serialize access to the global flag.
"""

import threading

_SPARSE_LOAD_LOCK = threading.Lock()
