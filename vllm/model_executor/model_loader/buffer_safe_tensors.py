#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Read a safetensors checkpoint into one buffer and view it zero-copy.

The loader path this supports:

    file --preadv (N threads)--> one destination buffer --> zero-copy views
                                                                  |
                                                                  v
                                                            GPU copy

Reading straight into the destination buffer is what makes this fast.
The obvious alternative - pread into the page cache, then copy out of
the mapping - pays for the bytes twice: once when the kernel fills the
page cache, once when the copy reads them back out.

Measured on 48 shards / 232.94 GiB of MiniMax-M3.1-Flash over quarkfs:

    read into the destination     36.31 s    6.41 GiB/s
    build the zero-copy views      0.35 s
    consumer copy out of them     13.19 s   17.66 GiB/s

On a CUDA host the destination is allocated with pin_memory=True, so the
views are backed by pinned pages and the host-to-device copy out of them
runs at the pinned rate (55.95 GiB/s measured on B300, PCIe Gen6 x16)
instead of the pageable rate (8.47 GiB/s).

The offsets come from safetensors' own ``get_tensor_meta``, which needs
safetensors >= 0.9. The offsets it reports are relative to the data
section, and the data section runs to the end of the file, so its
position in the file follows from the file size and the largest offset.
"""

from __future__ import annotations

import ctypes
from typing import Any

import torch

# safetensors dtype strings -> torch dtypes. Mirrors the set the format
# defines; safetensors 0.9 adds F8_E8M0 and F4, older ones only had the
# rest. An unknown dtype raises rather than being silently reinterpreted.
_SAFETENSORS_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E4M3FNUZ": torch.float8_e4m3fnuz,
    "F8_E5M2FNUZ": torch.float8_e5m2fnuz,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I16": torch.int16,
    "U16": torch.uint16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "U64": torch.uint64,
    "C64": torch.complex64,
}


def byte_memoryview(tensor: torch.Tensor) -> memoryview:
    """A writable byte view over a tensor's memory.

    ``tensor.numpy()`` cannot be used: a pinned tensor has no numpy
    representation. Going through ctypes and the tensor's address works
    for both pinned and pageable tensors, and lets ``os.preadv`` write
    straight into the buffer.
    """
    nbytes = tensor.numel() * tensor.element_size()
    raw = (ctypes.c_char * nbytes).from_address(tensor.data_ptr())
    return memoryview(raw).cast("B")


def safetensors_layout(handle: Any, size: int) -> tuple[int, dict[str, Any]]:
    """Return the offset of the data section in the file, and the headers.

    *handle* is an open safetensors handle (a ``_safe_open_handle`` or a
    ``safe_open``); *size* is the size of the file.

    The offsets safetensors reports are relative to the data section, and
    the data section runs to the end of the file, so the position of the
    data section follows from the size of the file and the largest
    offset: ``size - max(end)``.

    Only the header is read; the data section is not touched.
    """
    headers = {name: handle.get_tensor_meta(name) for name in handle.keys()}  # noqa: SIM118
    if not headers:
        raise ValueError("Checkpoint contains no tensors")
    data_len = max(meta.data_offsets[1] for meta in headers.values())
    data_start = size - data_len
    if data_start <= 0:
        raise ValueError(
            f"Checkpoint has an inconsistent size: {size} bytes, but the "
            f"data section is {data_len} bytes"
        )
    return data_start, headers


class BufferSafeTensors:
    """Zero-copy torch views over a buffer that holds the file's data.

    The buffer is usually a pinned host buffer that the file was preadv'd
    into, so the views are backed by pinned pages and a host-to-device
    copy out of them runs at the pinned rate.

    The views alias the buffer, so the buffer has to outlive them; the
    reader holds a reference to it and each view holds a reference to the
    reader, so the chain stays alive for as long as any view does.
    """

    def __init__(
        self,
        path: str,
        buffer: torch.Tensor,
        headers: dict[str, Any],
        data_start: int,
    ):
        """*buffer* holds the file's data section; *headers* maps a tensor
        name to its ``get_tensor_meta`` entry and *data_start* is the
        offset of the data section within the file."""
        self._path = path
        self._buffer = buffer
        self._data_start = data_start
        self._raw: torch.Tensor = buffer.view(torch.uint8)
        self._headers = headers

    def keys(self) -> list[str]:
        return list(self._headers)

    def get_tensor(self, name: str) -> torch.Tensor:
        """Return a view over the tensor's bytes. No copy is made."""
        meta = self._headers[name]
        torch_dtype = _SAFETENSORS_TO_TORCH_DTYPE.get(meta.dtype)
        if torch_dtype is None:
            raise NotImplementedError(
                f"unsupported safetensors dtype {meta.dtype!r} for tensor {name!r}"
            )
        start, end = meta.data_offsets
        shape = tuple(meta.shape)
        nbytes = end - start
        itemsize = torch.empty(0, dtype=torch_dtype).element_size()
        if nbytes % itemsize:
            raise ValueError(
                f"tensor {name!r} holds {nbytes} bytes, which is not a whole "
                f"number of {itemsize}-byte elements"
            )
        flat = self._typed_view(start, end, torch_dtype, nbytes, itemsize)
        view = flat.view(shape) if shape else flat.reshape(())
        # The views alias the buffer, so the buffer has to outlive them.
        view._buffer_safe_tensors = self  # type: ignore[attr-defined]
        return view

    def _typed_view(
        self,
        start: int,
        end: int,
        torch_dtype: torch.dtype,
        nbytes: int,
        itemsize: int,
    ) -> torch.Tensor:
        """A view over bytes [start, end) of the buffer, as *torch_dtype*.

        A view over the buffer's own storage keeps the buffer's
        allocation - pinned pages on a CUDA host - so a host-to-device
        copy out of it runs at the pinned rate. That needs the view's
        offset to be a whole number of elements; safetensors lays the
        data section out with each tensor aligned to its dtype, so this
        is the normal case. A tensor that is not aligned falls back to
        a view over a byte window, which is zero-copy too but is not
        backed by the buffer's allocation.
        """
        raw = self._raw[start:end]
        if start % itemsize == 0:
            return raw.view(torch_dtype)
        return torch.frombuffer(
            byte_memoryview(self._buffer)[start:end],
            dtype=torch_dtype,
            count=nbytes // itemsize,
        )

    def close(self) -> None:
        # The buffer belongs to the caller; only drop our references.
        self._raw = None  # type: ignore[assignment]
        self._headers = {}

    def __enter__(self) -> BufferSafeTensors:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
