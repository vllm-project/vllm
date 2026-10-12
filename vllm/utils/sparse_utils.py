# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Safe deserialization helpers for client-supplied embedding tensors.

`check_sparse_tensor_invariants_threadsafe()` is a thread-safe wrapper for
sparse tensor invariant validation. `safe_to_dense()` bounds the memory a
deserialized tensor may allocate when densified.

PyTorch's `torch.sparse.check_sparse_tensor_invariants()` context manager
manipulates a **process-global** flag (save/enable/restore). When multiple
embedding-load operations run concurrently on a thread-pool executor, one
context can restore the flag to `False` while another thread is still inside
its guard, bypassing the invariant check.

All call sites MUST use `check_sparse_tensor_invariants_threadsafe()`
which serializes access behind a lock.
"""

import contextlib
import threading

import torch

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.utils.mem_constants import MiB_bytes

_SPARSE_LOAD_LOCK = threading.Lock()


class TensorDecodeBudget:
    """Reserve the dense bytes for a group of client-supplied tensors."""

    def __init__(self) -> None:
        self._reserved_bytes = 0
        self._lock = threading.Lock()

    def reserve(self, dense_bytes: int, *, parameter: str) -> None:
        max_bytes = envs.VLLM_MAX_EMBED_DECODE_BYTES
        if max_bytes <= 0:
            return

        with self._lock:
            total_bytes = self._reserved_bytes + dense_bytes
            if total_bytes > max_bytes:
                raise VLLMValidationError(
                    f"`{parameter}` parts would allocate {total_bytes} bytes "
                    f"({total_bytes / MiB_bytes:.0f} MiB) once densified, "
                    f"exceeding the {max_bytes} byte request limit. Set "
                    f"VLLM_MAX_EMBED_DECODE_BYTES to increase this limit.",
                    parameter=parameter,
                    value=total_bytes,
                )
            self._reserved_bytes = total_bytes


@contextlib.contextmanager
def check_sparse_tensor_invariants_threadsafe():
    with _SPARSE_LOAD_LOCK, torch.sparse.check_sparse_tensor_invariants():
        yield


def safe_to_dense(
    tensor: object,
    *,
    parameter: str,
    budget: TensorDecodeBudget | None = None,
) -> torch.Tensor:
    """Densify a client-supplied embedding tensor within a memory budget.

    `torch.load(..., weights_only=True)` prevents arbitrary code execution but
    not memory amplification: a sparse COO tensor carries its own declared
    shape, so the dense allocation is bounded by that shape rather than by the
    request. A 2.6 KB payload declaring `(30000, 30000)` materializes 3.4 GiB,
    and the shape can be picked to fit whatever host is being targeted. The
    invariant checks only reject *invalid* tensors (indices outside the
    declared shape); a single non-zero element inside a `(2**20, 2**20)` shape
    is perfectly valid and passes them.

    So reject oversized payloads based on the dense size they declare, before
    `to_dense()` allocates anything.
    """
    if not isinstance(tensor, torch.Tensor):
        raise VLLMValidationError(
            f"`{parameter}` payload did not deserialize to a torch.Tensor.",
            parameter=parameter,
        )

    max_bytes = envs.VLLM_MAX_EMBED_DECODE_BYTES
    if max_bytes > 0:
        dense_bytes = tensor.numel() * tensor.element_size()
        if budget is not None:
            budget.reserve(dense_bytes, parameter=parameter)
        elif dense_bytes > max_bytes:
            raise VLLMValidationError(
                f"`{parameter}` declares shape {tuple(tensor.shape)}, which "
                f"would allocate {dense_bytes} bytes "
                f"({dense_bytes / MiB_bytes:.0f} MiB) once densified, "
                f"exceeding the {max_bytes} byte limit. Set "
                f"VLLM_MAX_EMBED_DECODE_BYTES to increase this limit.",
                parameter=parameter,
                value=dense_bytes,
            )

    return tensor.to_dense()
