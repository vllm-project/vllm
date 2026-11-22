# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense attention dispatch for RDNA4 Octave."""

from math import gcd, isfinite, prod

import torch

from vllm.utils.math_utils import cdiv, round_up

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

    def streamed_workspace(self, query, cache, requests, capacity, budget):
        """Bound the prefix partials by the existing per-layer scratch budget."""
        tokens, heads, dim = query.shape
        kv = cache.shape[1]
        rows = tokens * heads
        # The two INT8 query digits share staging with the later raw K/V tiles.
        staging = max(2 * rows * dim, (tokens // 32 + requests) * 32 * kv * dim * 4)
        overhead = sum(round_up(n, 256) for n in (staging, 4 * rows, rows))
        per_split = rows * (dim + 2) * 4
        # The occupancy targets below are calibrated in units of 128 query vectors.
        work_units = cdiv(cdiv(tokens, requests) * (heads // kv), 128) * kv
        # Short chunks need more independent prefix tiles to fill the GPU.
        target = 512 if capacity >= 32768 else 256 if capacity >= 8192 else 64
        # Long single-request chunks already have enough query parallelism.
        max_splits = 4 if requests == 1 and tokens >= 512 and capacity >= 32768 else 16
        splits = min(max_splits, cdiv(target, work_units))
        if requests == 1 and 128 <= tokens <= 448:
            groups = cdiv(tokens * (heads // kv), 160) * kv
            if groups <= 32:
                # A short final block of groups still scans its prefix. Use
                # the smallest split count that fills a wave of 32 groups.
                full = 32 // gcd(groups, 32)
                if groups >= 28 or (
                    groups >= 20 and capacity <= round_up(tokens + 512, 32)
                ):
                    splits = 1
                elif full <= 4 or capacity >= 32768:
                    splits = min(16, full)
                else:
                    first = 32 // groups
                    splits = first if first * groups >= 28 else 64 // groups
        splits = min(
            splits,
            cdiv(capacity, 256),
            (budget - overhead) // round_up(per_split, 256),
        )
        if splits < 1:
            return ()
        return (
            ((splits, tokens, heads, dim + 2), torch.float32),
            ((staging,), torch.int8),
            ((tokens, heads), torch.float32),
            ((tokens, heads), torch.uint8),
        )

    def prefill_workspace(
        self, query, cache, table, context_capacity, scale, budget, uniform_prefix=None
    ):
        tokens, heads, dim = query.shape
        requests, pages = table.shape
        kv = cache.shape[1]
        if (
            heads != 6 * kv
            or tokens < 32
            or not requests
            or not pages
            or not context_capacity
            or context_capacity < 0
            or not isfinite(scale)
            or scale <= 0
        ):
            return "packed", ()

        capacity = round_up(min(context_capacity, pages * cache.shape[2]), 32)
        expanded = (requests, kv, capacity, dim)
        correction_bytes = 2 * prod(expanded) + 32 * 4 * requests * kv
        fits = (
            4 * prod(expanded)
            + round_up(correction_bytes, 256)
            + 2 * query.numel()
            + round_up(2 * tokens * heads, 256)
            <= budget
        )
        chunk = tokens // requests
        # Short batches underfill expanded attention. Long, thin requests can
        # also use less scratch by reading the quantized pages directly.
        prefix = uniform_prefix or max(0, capacity - chunk)
        long_prefix = requests > 1 and (
            prefix >= max(32768, 192 * chunk) or prefix >= max(98304, 128 * chunk)
        )
        short_single = (
            requests == 1
            and tokens <= 448
            and cdiv(tokens * (heads // kv), 160) * kv <= 32
        )
        if (
            cache.shape[2] % 16 == 0
            and tokens >= 16 * requests
            and (tokens <= 384 or short_single or not fits or long_prefix)
        ):
            shapes = self.streamed_workspace(query, cache, requests, capacity, budget)
            if shapes:
                return "streamed", shapes
        if tokens >= 256 and fits:
            return "expanded", (
                (expanded, torch.float16),
                (expanded, torch.float16),
                (tuple(query.shape), torch.float16),
                ((tokens, heads, 2), torch.uint8),
                ((correction_bytes,), torch.uint8),
            )
        return "packed", ()

    def prefill_streamed(
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
        buffers,
        out,
        scale,
    ):
        """Attend to the paged prefix in place and merge the causal raw chunk."""
        torch.ops._C.octave_rdna4_prefill_streamed(
            out,
            query,
            key,
            value,
            cache,
            table,
            starts,
            lengths,
            *buffers,
            signs,
            scale,
            self.layout(cache_dtype).kernel_code,
        )
        return out

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
        repair=None,
        correction=None,
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
            repair,
            correction,
        )
        return out
