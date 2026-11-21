# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense attention dispatch for RDNA4 Octave."""

import torch

from .geometry import OctaveGeometry
from .hip import HipOctave
from .splits import recommend_splits


class TunedHipOctave(HipOctave):
    """Choose precompiled scalar/decode or matrix kernels for D=256/R=64."""

    def __init__(self, head_dim: int, rope_dim: int, arch: str):
        super().__init__(OctaveGeometry(head_dim, rope_dim), arch, geometry_only=False)

    def _use_geometry(self, query, cache, table, prefill):
        heads, kv_heads = query.shape[1], cache.shape[1]
        if kv_heads <= 0 or heads % kv_heads:
            return False
        group = heads // kv_heads
        if prefill:
            tokens, requests = query.shape[0], table.shape[0]
            if requests == 0:
                return True
            capacity = table.shape[1] * cache.shape[2]
            compact = cache.shape[-1] == self.layout("octave_k3v3_compact").slot_bytes
            if (
                (group in (1, 2) or (compact and group in (4, 8)))
                and tokens >= 256
                and tokens // requests >= 32
            ):
                return False
            # Reuse one packed KV tile across six long-prefill queries.
            if (
                group == 6
                and tokens >= 256
                and capacity >= (tokens + requests - 1) // requests + 1024
                and (requests == 1 or tokens // requests >= 128)
            ):
                return False
            # A single matrix family handles every GQA and prefill length.
            # Retain scalar latency for the small chunks below.
            if group == 6 and tokens >= 2 and tokens > requests:
                return True
            if tokens >= 256 and tokens // max(1, requests) >= 32:
                return True
            # Very small per-request MHA chunks underfill the query tile.
            if group == 1 and tokens < 32 and tokens // max(1, requests) < 8:
                return False
            # The existing packed FP32 paths win on short K3V3 contexts.
            # At 256/64 the three wire formats have distinct slot sizes.
            if (
                group <= 2
                and 512 <= capacity <= 2048
                and cache.shape[-1] != self.layout("octave_k3v4").slot_bytes
            ):
                return False
            if (
                group in (1, 2, 4, 8)
                and tokens >= 16
                and tokens > requests
                and tokens // max(1, requests) >= 4
                and (tokens < 256 or tokens // max(1, requests) < 32)
            ):
                return True
            return group not in (1, 2, 4, 6, 8)
        if group in (1, 2, 3, 6):
            return False
        capacity = table.shape[1] * cache.shape[2]
        tokens = query.shape[0]
        # Matrix setup needs enough reuse across heads or query rows. Keep
        # short-context and small-GQA latency on the original packed path.
        if group < 8:
            return tokens >= 32 and capacity >= 1024
        if group < 16:
            return tokens >= 256 or (tokens >= 8 and capacity >= 1024)
        return (
            capacity >= 65536
            or (capacity >= 512 and (tokens >= 4 or group >= 64))
            or tokens * group >= 256
        )

    def suggest_splits(
        self,
        query,
        cache,
        block_table,
        cache_dtype,
        *,
        prefill=False,
        context_capacity=None,
    ):
        if query.ndim != 3 or cache.ndim != 4 or block_table.ndim != 2:
            raise ValueError("Expected 3D query, 4D cache and 2D page table")
        return recommend_splits(
            tokens=query.shape[0],
            heads=query.shape[1],
            kv_heads=cache.shape[1],
            requests=block_table.shape[0],
            capacity=min(block_table.shape[1] * cache.shape[2], context_capacity)
            if context_capacity is not None and context_capacity > 0
            else block_table.shape[1] * cache.shape[2],
            page_size=cache.shape[2],
            cache_dtype=cache_dtype,
            prefill=prefill,
            head_dim=self.head_dim,
            specialized_fast=not self._use_geometry(query, cache, block_table, prefill),
        )

    def prefill_expanded(
        self,
        query,
        key,
        value,
        cache,
        table,
        starts,
        lengths,
        signs,
        cache_dtype,
        *,
        key_tmp,
        value_tmp,
        query_tmp,
        out,
        scale,
    ):
        """Expand the packed prefix into FP16 workspace for HIP WMMA prefill."""
        torch.ops._C.octave_rdna4_prefill_expanded(
            out,
            query,
            key,
            value,
            cache,
            table,
            starts,
            lengths,
            key_tmp,
            value_tmp,
            query_tmp,
            signs,
            scale,
            self.layout(cache_dtype).kernel_code,
        )
        return out
