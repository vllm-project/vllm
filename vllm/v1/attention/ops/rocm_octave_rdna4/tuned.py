# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense attention dispatch for RDNA4 Octave."""

from .geometry import OctaveGeometry
from .hip import HipOctave, build_library, get_geometry_engine
from .splits import recommend_splits


class TunedHipOctave(HipOctave):
    """Choose the packed 256/64 or general geometry kernel from host shapes."""

    def __init__(self, head_dim: int, rope_dim: int, arch: str):
        geometry = OctaveGeometry(head_dim, rope_dim)
        specialized = (head_dim, rope_dim) == (256, 64)
        super().__init__(
            build_library(geometry, arch, specialized=specialized),
            geometry,
            sparse=not specialized,
        )
        self._geometry_backend = (
            get_geometry_engine(head_dim, rope_dim, arch) if specialized else None
        )

    def _use_geometry(self, query, cache, table, prefill):
        if self._geometry_backend is None:
            return True
        heads, kv_heads = query.shape[1], cache.shape[1]
        if kv_heads <= 0 or heads % kv_heads:
            return False
        group = heads // kv_heads
        if prefill:
            tokens, requests = query.shape[0], table.shape[0]
            capacity = table.shape[1] * cache.shape[2]
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
    ):
        if query.ndim != 3 or cache.ndim != 4 or block_table.ndim != 2:
            raise ValueError("Expected 3D query, 4D cache and 2D page table")
        return recommend_splits(
            tokens=query.shape[0],
            heads=query.shape[1],
            kv_heads=cache.shape[1],
            requests=block_table.shape[0],
            capacity=block_table.shape[1] * cache.shape[2],
            page_size=cache.shape[2],
            cache_dtype=cache_dtype,
            prefill=prefill,
            head_dim=self.head_dim,
            specialized_fast=not self._use_geometry(query, cache, block_table, prefill),
        )

    def _attention(
        self,
        query,
        cache,
        table,
        mapping,
        lengths,
        signs,
        cache_dtype,
        scale,
        splits,
        out,
        scratch,
        *,
        key=None,
        value=None,
    ):
        engine = (
            self._geometry_backend
            if self._geometry_backend is not None
            and self._use_geometry(query, cache, table, key is not None)
            else self
        )
        attention = super()._attention if engine is self else engine._attention
        return attention(
            query,
            cache,
            table,
            mapping,
            lengths,
            signs,
            cache_dtype,
            scale,
            splits,
            out,
            scratch,
            key=key,
            value=value,
        )
