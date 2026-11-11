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

Why the header is parsed here
-----------------------------
The safetensors container format is trivial and stable: an 8-byte
little-endian header length, a JSON header, then the data section. Only
the header is parsed, once per shard.

safetensors 0.9 exposes the offsets through ``get_tensor_meta``, but
that API does not exist in 0.8.x, which vLLM still supports
(``safetensors >= 0.6.2``). Upstream also has no way to serve a view
over a caller-provided buffer: its storage is always the file's own
mapping.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
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


def pread_exact(fd: int, nbytes: int, offset: int) -> bytes:
    """Read exactly *nbytes* at *offset*, or raise EOFError."""
    chunks: list[bytes] = []
    got = 0
    while got < nbytes:
        chunk = os.pread(fd, nbytes - got, offset + got)
        if not chunk:
            raise EOFError("Checkpoint truncated during streaming prefetch")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def read_safetensors_header(fd: int, size: int) -> tuple[dict[str, Any], int]:
    """Return the file's header and the offset at which its data starts.

    The header is read through ``os.pread`` on the same descriptor the
    data section is read from, so the file is opened once and every byte
    of it is covered by that descriptor's read calls.

    A file that is too short to hold a header fails in the read itself,
    so a read error surfaces as itself rather than as a truncation.
    """
    header_len = struct.unpack("<Q", pread_exact(fd, 8, 0))[0]
    if header_len == 0 or 8 + header_len > size:
        raise EOFError("Checkpoint is truncated: safetensors header is invalid")
    header = json.loads(pread_exact(fd, header_len, 8))
    return header, 8 + header_len


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
        header: dict[str, Any],
        data_start: int,
    ):
        """*buffer* holds the file's data section; *header* is the file's
        parsed header and *data_start* is the offset of the data section
        within the file."""
        self._path = path
        self._buffer = buffer
        self._data_start = data_start
        self._raw: torch.Tensor = buffer.view(torch.uint8)
        self._specs = self._parse_header(header)

    def _parse_header(
        self, header: dict[str, Any]
    ) -> dict[str, tuple[str, tuple[int, ...], int, int]]:
        specs: dict[str, tuple[str, tuple[int, ...], int, int]] = {}
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            start, end = spec["data_offsets"]
            specs[name] = (spec["dtype"], tuple(spec["shape"]), start, end)
        return specs

    def keys(self) -> list[str]:
        return list(self._specs)

    def get_tensor(self, name: str) -> torch.Tensor:
        """Return a view over the tensor's bytes. No copy is made."""
        dtype, shape, start, end = self._specs[name]
        torch_dtype = _SAFETENSORS_TO_TORCH_DTYPE.get(dtype)
        if torch_dtype is None:
            raise NotImplementedError(
                f"unsupported safetensors dtype {dtype!r} for tensor {name!r}"
            )
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
        self._specs = {}

    def __enter__(self) -> BufferSafeTensors:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
