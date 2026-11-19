# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor wrappers for the RDNA4 Octave ops registered in vLLM's _C extension."""

from functools import cache

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_octave import OctaveFormat

from .geometry import OctaveGeometry


class HipOctave:
    """Paged Octave on gfx1200/gfx1201, with FP32 attention output.

    Callers supply valid page indices and unique live write slots. Q and K are
    post-RoPE. The contiguous int32 sign words follow the Octave format.
    """

    def __init__(self, geometry: OctaveGeometry, arch: str, *, geometry_only: bool):
        if not torch.version.hip:
            raise RuntimeError("RDNA4 Octave requires PyTorch built for ROCm")
        if arch not in ("gfx1200", "gfx1201"):
            raise ValueError(f"Unsupported RDNA4 target {arch}")
        current_platform.import_kernels()
        if not hasattr(torch.ops._C, "octave_rdna4_decode"):
            raise RuntimeError(
                f"RDNA4 Octave ops are unavailable for {arch}. "
                "Rebuild and reinstall vLLM with ROCm support for gfx1200 or gfx1201."
            )
        self.geometry = geometry
        self.head_dim, self.rope_dim = geometry.head_dim, geometry.rope_dim
        self._geometry_only = geometry_only

    def layout(self, cache_dtype: str) -> OctaveFormat:
        return self.geometry.layout(cache_dtype)

    def store(self, key, value, cache, slots, signs, cache_dtype):
        torch.ops._C.octave_rdna4_cache_store(
            key,
            value,
            cache,
            slots,
            signs,
            self.layout(cache_dtype).kernel_code,
            self._geometry_only,
        )

    def window_store(
        self, key, value, slots, positions, wtab, wtags, wlocks, stab, stags, slocks
    ):
        """Keep exact K/V for selected recent and leading QSA positions.

        Direct-mapped tables use physical-slot tags. Collisions evict complete
        rows, so sparse attention can safely fall back to packed KV.
        """
        if not self._geometry_only:
            raise RuntimeError("Sparse Octave requires the geometry runtime")
        if wlocks is None or slocks is None:
            raise ValueError("Exact K/V stores require both entry-lock tables")
        tensors = (
            key,
            value,
            slots,
            positions,
            wtab,
            wtags,
            wlocks,
            stab,
            stags,
            slocks,
        )
        torch.ops._C.octave_rdna4_window_store(*tensors)
        stream = torch.cuda.current_stream(key.device)
        for tensor in tensors:
            tensor.record_stream(stream)

    def decode_sparse(
        self,
        query,
        cache,
        block_table,
        q_to_req,
        indices,
        signs,
        cache_dtype,
        *,
        positions=None,
        wtab=None,
        wtags=None,
        stab=None,
        stags=None,
        scale=None,
        splits=1,
        out=None,
        scratch=None,
    ):
        """FP32 sparse attention over selected logical positions and holes.

        Optional exact tables preserve the first four and last 64 selected
        positions. Missing or evicted tags use the packed cache. No prefix is
        expanded to a dense K/V tensor. All buffers may be supplied for capture.
        """
        if not self._geometry_only:
            raise RuntimeError("Sparse Octave requires the geometry runtime")
        out, scratch = self._buffers(query, splits, out, scratch, 1024)
        torch.ops._C.octave_rdna4_decode_sparse(
            out,
            query,
            cache,
            block_table,
            q_to_req,
            indices,
            scratch,
            signs,
            self.head_dim**-0.5 if scale is None else scale,
            splits,
            self.layout(cache_dtype).kernel_code,
            positions,
            wtab,
            wtags,
            stab,
            stags,
        )
        stream = torch.cuda.current_stream(query.device)
        for tensor in (
            query,
            cache,
            block_table,
            q_to_req,
            indices,
            signs,
            positions,
            wtab,
            wtags,
            stab,
            stags,
            out,
            scratch,
        ):
            if tensor is not None:
                tensor.record_stream(stream)
        return out

    def decode(
        self,
        query,
        cache,
        block_table,
        q_to_req,
        q_to_klen,
        signs,
        cache_dtype,
        *,
        scale=None,
        splits=1,
        out=None,
        scratch=None,
    ):
        out, scratch = self._buffers(query, splits, out, scratch, 256)
        torch.ops._C.octave_rdna4_decode(
            out,
            query,
            cache,
            block_table,
            q_to_req,
            q_to_klen,
            scratch,
            signs,
            self.head_dim**-0.5 if scale is None else scale,
            splits,
            self.layout(cache_dtype).kernel_code,
            self._use_geometry(query, cache, block_table, False),
        )
        return out

    def prefill(
        self,
        query,
        key,
        value,
        cache,
        block_table,
        query_start_loc,
        seq_lens,
        signs,
        cache_dtype,
        *,
        scale=None,
        splits=1,
        out=None,
        scratch=None,
    ):
        out, scratch = self._buffers(query, splits, out, scratch, 256)
        torch.ops._C.octave_rdna4_prefill(
            out,
            query,
            key,
            value,
            cache,
            block_table,
            query_start_loc,
            seq_lens,
            scratch,
            signs,
            self.head_dim**-0.5 if scale is None else scale,
            splits,
            self.layout(cache_dtype).kernel_code,
            self._use_geometry(query, cache, block_table, True),
        )
        return out

    def _use_geometry(self, query, cache, table, prefill):
        return self._geometry_only

    def _buffers(self, query, splits, out, scratch, max_splits):
        if not 1 <= splits <= max_splits:
            raise ValueError(f"Expected 1 to {max_splits} Octave splits")
        if out is None:
            out = torch.empty(query.shape, device=query.device, dtype=torch.float32)
        if splits > 1 and scratch is None:
            scratch = torch.empty(
                (*query.shape[:2], splits, self.head_dim + 2),
                device=query.device,
                dtype=torch.float32,
            )
        return out, scratch


@cache
def get_geometry_engine(head_dim: int, rope_dim: int, arch: str) -> HipOctave:
    return HipOctave(OctaveGeometry(head_dim, rope_dim), arch, geometry_only=True)
