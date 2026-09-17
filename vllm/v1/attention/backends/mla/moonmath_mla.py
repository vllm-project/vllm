# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A16W8 MLA decode on the optional `moonmath_amd` CDNA3 kernels.

Replaces ONLY the decode, single-token and speculative verify alike: a bf16
query against the fp8 KV cache, with the whole draft window resident in one
CTA, so the KV is read once for the window instead of once per position.
AITER's only bf16xfp8 gqa=16 object is qSeqLen=4 and pads a smaller head count
up to 16.

Prefill, metadata and cudagraph buffers are inherited from `AiterMLABackend`
unchanged; a batch outside the kernel's domain falls back to it. Under decode
context parallelism each rank attends its own KV shard, with the causal limit
taken from the global lengths, and returns its LSE for the cross-rank merge.
The package is optional and imported lazily, so this module is safe to import
without it; `_get_backend_priorities` only offers the backend when
`has_moonmath_amd()`.
"""

from __future__ import annotations

import math

import torch

from vllm.config import get_current_vllm_config
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    AiterMLABackend,
    AiterMLAImpl,
    AiterMLAMetadata,
    AiterMLAMetadataBuilder,
)

_LAT, _ROPE = 512, 64
# Kernel domain: H <= 128, and B * ceil(q_len * H / 96) <= 304 query row slices.
_MAX_HEADS, _ROWS_PER_SLICE, _MAX_ROW_SLICES = 128, 96, 304


def _kernel_serves(num_reqs: int, q_len: int, heads: int) -> bool:
    return (
        heads <= _MAX_HEADS
        and num_reqs * cdiv(q_len * heads, _ROWS_PER_SLICE) <= _MAX_ROW_SLICES
    )


class MoonmathMLAMetadataBuilder(AiterMLAMetadataBuilder):
    # Verify reads the flat per-token view; the kernel places each row's causal
    # window on the rank's shard from the global lengths.
    segmented_dcp_verify = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # The query is bf16 here (supports_quant_query_input=False), so pin
        # q_dtype rather than let the inherited env gate disagree with it.
        self._mla_q_dtype = self.decode_attn_out_dtype

    def _decode_reads_aiter_schedule(
        self, num_reqs: int, qo_len: int, uniform_qo_len: bool, causal: bool
    ) -> bool:
        return not (
            causal
            and uniform_qo_len
            and _kernel_serves(num_reqs, qo_len, self._decode_num_heads)
        )


class MoonmathMLAImpl(AiterMLAImpl):
    """Aiter MLA with the decode on the moonmath kernel."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # a16w8: bf16 query vs fp8 KV. Set after super(), which assigns True.
        self.supports_quant_query_input = False
        self._mm_kv_scale: float | None = None
        self._max_model_len = get_current_vllm_config().model_config.max_model_len

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        decode = attn_metadata.decode
        assert decode is not None
        # A bf16 (nope, pe) pair (supports_quant_query_input is False); DCP
        # concatenates the halves before gathering the query heads.
        q_lat, q_pe = q if isinstance(q, tuple) else q.split((_LAT, _ROPE), dim=-1)
        # Routing reads only shapes and the causal flag, so it is fixed per
        # captured graph.
        num_tokens, heads, _ = q_lat.shape
        num_reqs = decode.seq_lens.size(0)
        q_len = num_tokens // num_reqs
        dcp = self.dcp_world_size > 1
        if (
            not attn_metadata.causal
            or (dcp and decode.dcp_tot_seq_lens is None)
            or decode.paged_kv_indices is None
            or num_tokens != num_reqs * q_len
            or not _kernel_serves(num_reqs, q_len, heads)
        ):
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)

        assert kv_c_and_k_pe_cache.dtype == torch.float8_e4m3fnuz, (
            "moonmath MLA needs an fp8_e4m3fnuz KV cache (--kv-cache-dtype fp8), "
            f"got {kv_c_and_k_pe_cache.dtype}"
        )
        # Both halves arrive as transposed views; the kernel takes contiguous
        # [B*q_len, H, dim] rows, request b owning [b*q_len, (b+1)*q_len).
        q_lat = q_lat.contiguous()
        q_pe = q_pe.contiguous()
        out = torch.empty_like(q_lat)
        lse = (
            torch.empty(num_tokens, heads, dtype=torch.float32, device=q_lat.device)
            if dcp
            else None
        )

        if self._mm_kv_scale is None:
            # Static load-time constant, read once: .item() is illegal inside
            # graph capture (the first call is always the eager dummy run).
            k = getattr(layer, "_k_scale_float", None)
            self._mm_kv_scale = float(layer._k_scale if k is None else k)

        import moonmath_amd as ma

        # The kernel plans its KV split from kv_indices' length (the mean KV
        # length per request), so hand it the persistent buffer cut at the
        # configured ceiling: a shape-only slice, fixed at graph capture.
        ma.mla_decode_a16w8(
            q_lat,
            q_pe,
            kv_c_and_k_pe_cache.view(-1, 1, _LAT + _ROPE),
            out,
            decode.seq_lens,
            decode.paged_kv_indices[: num_reqs * self._max_model_len],
            decode.paged_kv_indptr,
            self.scale,
            self._mm_kv_scale,
            lse=lse,
            glen=decode.dcp_tot_seq_lens if dcp else None,
            cp_rank=self.dcp_rank,
            cp_world=self.dcp_world_size,
        )
        if lse is None:
            return out, None
        # The kernel's LSE is base 2; the DCP merge reads aiter's natural log.
        return out, lse.mul_(math.log(2))


class MoonmathMLABackend(AiterMLABackend):
    @staticmethod
    def get_name() -> str:
        return "MOONMATH_MLA"

    @staticmethod
    def get_impl_cls() -> type[MoonmathMLAImpl]:
        return MoonmathMLAImpl

    @staticmethod
    def get_builder_cls() -> type[MoonmathMLAMetadataBuilder]:
        return MoonmathMLAMetadataBuilder
