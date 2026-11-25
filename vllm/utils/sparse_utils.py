# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Safe deserialization helpers for client-supplied embedding tensors.

`check_sparse_tensor_invariants_threadsafe()` is a thread-safe wrapper for
sparse tensor invariant validation. `check_torch_archive_size()` bounds the
memory `torch.load` allocates for the tensor storages of a payload.
`load_torch_payload()` maps unreadable client bytes to a validation error.
`safe_to_dense()` bounds the memory a deserialized tensor may allocate when
densified.

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
import zipfile
from io import BytesIO
from typing import Final

import torch

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.utils.mem_constants import MiB_bytes

_SPARSE_LOAD_LOCK = threading.Lock()

# How much of torch's own reason is repeated back to the caller. The reason is
# built from bytes the caller sent, so it is bounded rather than echoed whole.
_MAX_EMBED_ERROR_REASON_CHARS: Final = 200


def _truncated_reason(exc: Exception) -> str:
    """Return `exc`'s message, capped with the omitted character count."""
    reason: Final = str(exc).strip() or type(exc).__name__
    omitted: Final = len(reason) - _MAX_EMBED_ERROR_REASON_CHARS
    if omitted <= 0:
        return reason
    return (
        f"{reason[:_MAX_EMBED_ERROR_REASON_CHARS]}"
        f"... ({omitted} more characters truncated)"
    )


@contextlib.contextmanager
def check_sparse_tensor_invariants_threadsafe():
    with _SPARSE_LOAD_LOCK, torch.sparse.check_sparse_tensor_invariants():
        yield


_ZIP_LOCAL_FILE_HEADER = b"PK\x03\x04"


def check_torch_archive_size(payload: bytes, *, parameter: str) -> None:
    """Bound the bytes `torch.load` inflates from a client-supplied payload.

    `torch.save` writes a zip archive, and `torch.load` reads each tensor
    storage entry into memory at the uncompressed size recorded in the
    archive. Entries may be deflate-compressed, so a few MiB of zeros can
    declare a storage of many GiB. That allocation happens inside
    `torch.load`, before `safe_to_dense()` sees the tensor, and a small view
    of a large storage keeps the whole storage alive. Reject archives whose
    entries add up to more than `VLLM_MAX_EMBED_DECODE_BYTES` before loading.
    """
    max_bytes = envs.VLLM_MAX_EMBED_DECODE_BYTES
    if max_bytes <= 0 or payload[:4] != _ZIP_LOCAL_FILE_HEADER:
        return

    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            inflated_bytes = sum(info.file_size for info in archive.infolist())
    except (zipfile.BadZipFile, ValueError) as exc:
        raise VLLMValidationError(
            f"`{parameter}` payload is not a readable torch.save archive.",
            parameter=parameter,
        ) from exc

    if inflated_bytes > max_bytes:
        raise VLLMValidationError(
            f"`{parameter}` payload inflates to {inflated_bytes} bytes "
            f"({inflated_bytes / MiB_bytes:.0f} MiB) when loaded, exceeding "
            f"the {max_bytes} byte limit. Set VLLM_MAX_EMBED_DECODE_BYTES to "
            f"increase this limit.",
            parameter=parameter,
            value=inflated_bytes,
        )


def load_torch_payload(
    payload: bytes,
    *,
    parameter: str,
) -> torch.Tensor:
    """Safely deserialize a tensor from client-supplied bytes.

    Payload storages are always mapped to CPU so checkpoint metadata cannot
    select a server device. Unreadable payloads are client validation errors.
    Memory exhaustion remains a server error because it can also be caused by
    server capacity.

    Args:
        payload: Bytes supplied by the client.
        parameter: Request parameter used in validation errors.

    Returns:
        The deserialized dense tensor.

    Raises:
        VLLMValidationError: If the payload cannot be deserialized as a tensor.
        MemoryError: If deserialization exhausts host memory.
        torch.OutOfMemoryError: If deserialization exhausts device memory.

    """
    check_torch_archive_size(payload, parameter=parameter)

    with check_sparse_tensor_invariants_threadsafe():
        try:
            tensor = torch.load(
                BytesIO(payload),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        except (torch.OutOfMemoryError, MemoryError):
            raise
        except Exception as exc:
            raise VLLMValidationError(
                f"`{parameter}` could not be deserialized as a torch tensor: "
                f"{_truncated_reason(exc)}",
                parameter=parameter,
            ) from exc

        return safe_to_dense(tensor, parameter=parameter)


def safe_to_dense(tensor: object, *, parameter: str) -> torch.Tensor:
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
        if dense_bytes > max_bytes:
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
