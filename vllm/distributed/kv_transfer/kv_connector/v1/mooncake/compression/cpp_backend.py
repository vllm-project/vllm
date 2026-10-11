# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python side of the CUDA compression backend.

Builds one flat int64 metadata blob per ``(slot, kv_part)`` execution plan and
calls the C++ implementation once per chunk.  The C++ side performs all
per-layer gather/transform/quantize (or their inverses) work; Python keeps
only session handling, the ANS codec and buffer allocation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

_CSRC = Path(__file__).resolve().parent / "csrc"

KVS_META_HEADER_FIELDS = 16
KVS_PART_FIELDS = 24
KVS_SECTION_FIELDS = 8

_module: Any | None = None


def load_kvs_compress(device: torch.device | None = None) -> Any:
    """Build (once) and return the compiled extension module."""
    global _module
    if _module is None:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(device)
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
        _module = load(
            name="kvs_compress",
            sources=[
                str(_CSRC / "kvs_compress.cu"),
                str(_CSRC / "kvs_bind.cpp"),
            ],
            extra_include_paths=[str(_CSRC)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    return _module


def ping(value: int = 41) -> int:
    """Smoke test for build integration."""
    return int(load_kvs_compress().kvs_ping(value))


def enabled() -> bool:
    """Whether the connector should route compression through C++."""
    backend = os.environ.get("KVSERVE_COMPRESSION_BACKEND", "python")
    if backend not in ("python", "cpp"):
        raise ValueError(f"Unknown KVSERVE_COMPRESSION_BACKEND: {backend!r}")
    return backend == "cpp"


_AXIS_CODES = {"channel": 0, "token": 1, "tensor": 2}


def build_meta(pipeline: Any, slot: Any, kv_part: int) -> torch.Tensor:
    """Pack the execution plan into the flat int64 metadata blob."""
    layout = pipeline.layout
    parts = list(layout.parts(kv_part))
    total_sections = sum(len(part.sections) for part in parts)
    meta = torch.zeros(
        KVS_META_HEADER_FIELDS
        + KVS_PART_FIELDS * len(parts)
        + KVS_SECTION_FIELDS * total_sections,
        dtype=torch.int64,
    )
    meta[0] = len(parts)
    meta[1] = slot.arena_a.data_ptr()
    meta[2] = slot.arena_b.data_ptr()
    meta[3] = slot.aux.data_ptr()
    meta[4] = slot.raw_tail.data_ptr() if slot.raw_tail is not None else 0
    meta[5] = layout.u8_base
    meta[6] = layout.aux_arena_bytes

    section_index = 0
    for index, part in enumerate(parts):
        row = KVS_META_HEADER_FIELDS + KVS_PART_FIELDS * index
        cache = pipeline._cache_part(part)
        element_bytes = cache.element_size()
        meta[row + 0] = cache.data_ptr()
        meta[row + 1] = cache.stride(0) * element_bytes
        meta[row + 2] = cache.stride(1) * element_bytes
        meta[row + 3] = cache.stride(2) * element_bytes
        signs = pipeline.signs.get((part.layer_name, part.kv_part))
        meta[row + 4] = signs.data_ptr() if signs is not None else 0
        order = pipeline.head_orders.get(part.layer_name)
        identity = order is None or part.layer_name in pipeline.identity_orders
        meta[row + 5] = 0 if identity else order.data_ptr()
        meta[row + 6] = part.num_heads
        meta[row + 7] = part.head_size
        meta[row + 8] = part.block_size // pipeline.blocks_per_logical
        meta[row + 9] = part.raw_offset
        meta[row + 10] = layout.scratch_stride + part.raw_offset
        meta[row + 11] = part.u8_offset
        meta[row + 12] = part.raw_bytes
        meta[row + 13] = int("transformer" in pipeline.stages)
        meta[row + 14] = int("quantizer" in pipeline.stages)
        meta[row + 15] = len(part.sections)
        meta[row + 16] = section_index
        meta[row + 17] = part.block_size
        meta[row + 18] = pipeline.blocks_per_logical
        for section in part.sections:
            srow = (
                KVS_META_HEADER_FIELDS
                + KVS_PART_FIELDS * len(parts)
                + KVS_SECTION_FIELDS * section_index
            )
            meta[srow + 0] = section.head_start
            meta[srow + 1] = section.head_count
            meta[srow + 2] = section.num_levels
            meta[srow + 3] = _AXIS_CODES[section.axis]
            meta[srow + 4] = section.aux_min_offset
            meta[srow + 5] = section.aux_scale_offset
            section_index += 1
    return meta


def encode_chunk(
    meta: torch.Tensor,
    indices: torch.Tensor,
    num_blocks: int,
    valid_tokens: int,
    stream: torch.cuda.Stream,
) -> None:
    load_kvs_compress().kvs_encode_flat(
        meta.data_ptr(),
        indices.data_ptr(),
        int(num_blocks),
        int(valid_tokens),
        int(stream.cuda_stream),
    )


def decode_chunk(
    meta: torch.Tensor,
    indices: torch.Tensor,
    num_blocks: int,
    valid_tokens: int,
    u8_bytes: int,
    stream: torch.cuda.Stream,
) -> None:
    load_kvs_compress().kvs_decode_flat(
        meta.data_ptr(),
        indices.data_ptr(),
        int(num_blocks),
        int(valid_tokens),
        int(u8_bytes),
        int(stream.cuda_stream),
    )
