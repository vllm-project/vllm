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
runs at the pinned rate (55.95 GB/s measured on B300, PCIe Gen6 x16)
instead of the pageable rate (8.47 GB/s).

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

import json
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


def parse_safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    """Return the file's header and the offset at which its data starts."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
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
        flat = self._raw[start:end].view(torch_dtype)
        view = flat.view(shape) if shape else flat.reshape(())
        # The views alias the buffer, so the buffer has to outlive them.
        view._buffer_safe_tensors = self  # type: ignore[attr-defined]
        return view

    def close(self) -> None:
        # The buffer belongs to the caller; only drop our references.
        self._raw = None

    def __enter__(self) -> BufferSafeTensors:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
