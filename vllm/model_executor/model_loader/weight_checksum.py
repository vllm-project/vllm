# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tensor digests of a model's weights and the reset that supports
verifying a weight update.

Nothing here depends on parallel ranks: the caller qualifies the digests with
its own rank prefix, so this stays a model-layer concern.
"""

import hashlib

import torch
import torch.nn as nn

_INTEGER_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

_NON_PERSISTENT_BUFFER_PATTERNS = (
    "cos_cached",
    "sin_cached",
    "cos_sin_cache",
    "inv_freq",
    "freqs_cis",
)


def _iter_checksum_targets(model: nn.Module):
    """Yield (name, tensor) for persistent weights: all parameters plus
    persistent buffers (quantizers/adapters sometimes store weights there)."""
    for name, tensor in model.named_parameters():
        if not tensor.is_floating_point() and tensor.dtype not in _INTEGER_DTYPES:
            continue
        yield name, tensor

    seen_buffers: set[int] = set()
    for module_name, module in model.named_modules():
        for buffer_name, tensor in module.named_buffers(recurse=False):
            if id(tensor) in seen_buffers:
                continue
            seen_buffers.add(id(tensor))
            if buffer_name in module._non_persistent_buffers_set:
                continue
            name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            if any(p in name for p in _NON_PERSISTENT_BUFFER_PATTERNS):
                continue
            if not tensor.is_floating_point() and tensor.dtype not in _INTEGER_DTYPES:
                continue
            yield name, tensor


def _randomize_tensor_inplace(tensor: torch.Tensor) -> None:
    """Fill ``tensor`` with random values without a same-sized temporary."""
    if tensor.is_floating_point():
        values = torch.rand_like(tensor, dtype=torch.float32).to(tensor.dtype)
    else:
        values = torch.randint(
            0,
            2,
            tensor.shape,
            device=tensor.device,
            dtype=tensor.dtype,
        )
    tensor.copy_(values)


def compute_tensor_digests(model: nn.Module) -> dict[str, str]:
    """Return one SHA-256 digest per checksum-covered tensor, keyed by name.

    Hashing needs host bytes, so each tensor is moved to CPU as one uint8 array
    and passed to hashlib as a buffer.
    """
    digests: dict[str, str] = {}
    for name, tensor in _iter_checksum_targets(model):
        # Reshape first: view(dtype) rejects a 0-dim tensor outright, and a few
        # models keep scalar buffers such as per-layer k_scale/v_scale. A
        # reshape also drops the separate .contiguous() call.
        flat = tensor.data.cpu().reshape(-1)
        cpu_uint8 = flat.view(torch.uint8).numpy()
        # Hash the array in place; .tobytes() would copy it a second time.
        raw = memoryview(cpu_uint8)
        digests[name] = hashlib.sha256(raw).hexdigest()
    return digests


def randomize_weights(model: nn.Module) -> None:
    """Randomize exactly the tensors covered by ``compute_tensor_digests``."""
    for _, tensor in _iter_checksum_targets(model):
        # Chunk so the staging buffer stays bounded for large weights.
        if tensor.numel() == 0:
            continue
        if tensor.is_contiguous():
            chunks = tensor.data.view(-1).split(64 * 1024 * 1024)
        elif tensor.ndim == 0:
            chunks = (tensor.data,)
        else:
            row_numel = tensor[0].numel()
            rows_per_chunk = max(1, (64 * 1024 * 1024) // row_numel)
            chunks = tensor.data.split(rows_per_chunk, dim=0)
        for chunk in chunks:
            _randomize_tensor_inplace(chunk)
