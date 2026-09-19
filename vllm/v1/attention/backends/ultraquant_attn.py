# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UltraQuant attention implementation.

UltraQuant is a 4-bit KV-cache format (FP4 E2M1 codes + UE8M0 power-of-two
per-group-32 scales) with Hadamard-rotated keys. It reuses the TurboQuant
attention scaffolding (forward orchestration, first-chunk prefill, metadata,
CUDA-graph pre-warm) and overrides only the format-specific seams:

  * store          -> ultraquant_store (rotate K, pack FP4 + UE8M0 scales)
  * decode         -> FlyDSL D=256 scaled-MFMA kernel, Triton unified fallback
  * continuation   -> small chunk: Triton unified; large chunk: dequant + FA

Selected by the TurboQuant backend's ``get_impl_cls`` when the cache dtype is
``ultraquant_4bit``; the base ``TurboQuantAttentionImpl`` is untouched.
"""

import math
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.v1.attention.backends.turboquant_attn import (
    _CK_MAX_HEAD_DIM,
    _CONTINUATION_DECODE_THRESHOLD,
    _HAS_FLASH_ATTN,
    TurboQuantAttentionImpl,
    TurboQuantMetadata,
)
from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    flydsl_ultraquant_decode_attention,
    ultraquant_flydsl_decode_eligible,
)
from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    is_flydsl_available as is_ultraquant_flydsl_available,
)
from vllm.v1.attention.ops.ultraquant.triton_unified_attention import (
    ultraquant_unified_attention,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


class UltraQuantAttentionImpl(TurboQuantAttentionImpl):
    """UltraQuant 4-bit KV-cache attention (FP4 E2M1 + UE8M0, rotated K).

    Overrides the quantizer-config, decode-backend, quant-table, store, decode,
    and continuation seams of :class:`TurboQuantAttentionImpl`; all shared
    prefill/forward/metadata machinery is inherited unchanged.
    """

    # FlyDSL D=256 decode loads query as bf16; fp16 is not supported.
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]

    def _init_quant_config(self, kv_cache_dtype: str, head_size: int) -> None:
        # UltraQuant is not described by a TurboQuantConfig; the format lives in
        # ops/ultraquant. No Lloyd-Max centroids or per-config byte counts.
        self.tq_config = None  # type: ignore[assignment]
        self._mse_bytes = 0
        self._val_data_bytes = 0
        self._n_centroids = 0

    def _init_decode_backend(self) -> None:
        # UltraQuant uses AoS store (never SoA) and its own FlyDSL decode probe.
        self._use_flydsl = False
        self._soa_store = False
        self._use_ultraquant_flydsl = is_ultraquant_flydsl_available()
        # Pre-warm capture buffers/workspace when the FlyDSL decode is active.
        self._prewarm_capture_buffers = self._use_ultraquant_flydsl

    def _init_quant_tables(self, layer: Any, D: int, device) -> None:
        # UltraQuant needs no centroids/midpoints; keep the attrs present (empty)
        # so shared code that references them stays valid.
        layer._tq_centroids = torch.empty(0, device=device, dtype=torch.float32)
        layer._tq_midpoints = torch.empty(0, device=device, dtype=torch.float32)

    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        """Rotate K by the Hadamard and pack K+V into FP4/UE8M0 slots."""
        from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store

        ultraquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            PiT=layer._tq_PiT,
        )

    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        _mid_o, output_buf, _lse = self._acquire_decode_buffers(query)

        gqa = self.num_kv_groups
        flydsl_eligible = ultraquant_flydsl_decode_eligible(
            head_size=self.head_size,
            num_kv_groups=gqa,
            has_sinks=self.sinks is not None,
            sliding_window=self.sliding_window,
            flydsl_loaded=self._use_ultraquant_flydsl,
        )
        if flydsl_eligible:
            return flydsl_ultraquant_decode_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                scale=self.scale,
                PiT=PiT,
                max_seq_len=attn_metadata.max_seq_len,
                output_buf=output_buf,
                buf_holder=layer,
                max_num_kv_splits=self.max_num_kv_splits,
                sinks=self.sinks,
            )

        logger.info_once(
            "UltraQuant FlyDSL is ineligible for head_size=%s, GQA=%s, "
            "sinks=%s, sliding_window=%s; using Triton fallback.",
            self.head_size,
            gqa,
            self.sinks is not None,
            self.sliding_window,
        )
        return ultraquant_unified_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            query_start_loc=attn_metadata.query_start_loc,
            scale=self.scale,
            PiT=PiT,
            output=output_buf,
            max_query_len=1,
            max_seq_len=attn_metadata.max_seq_len,
            sinks=self.sinks,
            sliding_window=self.sliding_window,
        )

    def _continuation_chunk_attention(
        self,
        *,
        i: int,
        q_seq: torch.Tensor,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        cached_len: int,
        seq_len: int,
        q_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None,
        layer: Any,
        arange_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Attend a continuation chunk against its cached prefix (UltraQuant).

        Small chunks stay on the Triton unified (decode-shaped) kernel; large
        chunks dequant the prefix once and run a dense prefill.
        """
        if q_len <= _CONTINUATION_DECODE_THRESHOLD:
            synth_seq_lens = arange_cache[cached_len + 1 : seq_len + 1]
            synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
            return ultraquant_unified_attention(
                query=q_seq,
                kv_cache=kv_cache,
                block_table=synth_bt,
                seq_lens=synth_seq_lens,
                query_start_loc=arange_cache[: q_len + 1],
                scale=self.scale,
                PiT=PiT,
                max_query_len=1,
                max_seq_len=seq_len,
                sinks=self.sinks,
                sliding_window=self.sliding_window,
            )
        return self._ultraquant_continuation_prefill(
            layer=layer,
            query=q_seq,
            key_chunk=k_seq,
            val_chunk=v_seq,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table[i : i + 1],
            cached_len=cached_len,
            seq_len=seq_len,
            PiT=PiT,
        )

    def _ultraquant_continuation_prefill(
        self,
        *,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D) unrotated
        key_chunk: torch.Tensor,  # (q_len, Hk, D) unrotated
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        PiT: torch.Tensor,  # (D, D) fp32 Hadamard
    ) -> torch.Tensor:
        """Large continuation chunk: dequant the cached prefix, then flash_attn.

        The cache stores K already Hadamard-rotated, so only the current chunk's
        K and the queries need rotating. Hadamard is symmetric orthonormal, so
        rotating both sides leaves the attention scores unchanged.
        """
        from vllm.v1.attention.ops.ultraquant.triton_dequant import (
            ultraquant_full_dequant_kv,
        )

        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        qdtype = query.dtype
        block_size = kv_cache.shape[1]
        alloc_len = math.ceil(cached_len / block_size) * block_size

        # Dequant target, shared across layers by the workspace manager so long
        # context does not pay one allocation per layer.
        buf_shape = (1, Hk, alloc_len, D)
        k_buf, v_buf = current_workspace_manager().get_simultaneous(
            (buf_shape, qdtype),
            (buf_shape, qdtype),
        )
        ultraquant_full_dequant_kv(
            kv_cache=kv_cache,
            block_table=block_table,
            k_out=k_buf,
            v_out=v_buf,
            alloc_len=alloc_len,
        )

        # Layer-cached prefix+chunk concatenation, grown once to worst case.
        cap = block_table.shape[1] * block_size
        k_full_buf = getattr(layer, "_uq_kfull_buf", None)
        if (
            k_full_buf is None
            or k_full_buf.shape[0] < cap
            or k_full_buf.dtype != qdtype
        ):
            k_full_buf = torch.empty(cap, Hk, D, dtype=qdtype, device=device)
            layer._uq_kfull_buf = k_full_buf
        v_full_buf = getattr(layer, "_uq_vfull_buf", None)
        if (
            v_full_buf is None
            or v_full_buf.shape[0] < cap
            or v_full_buf.dtype != qdtype
        ):
            v_full_buf = torch.empty(cap, Hk, D, dtype=qdtype, device=device)
            layer._uq_vfull_buf = v_full_buf
        k_full = k_full_buf[:seq_len]
        v_full = v_full_buf[:seq_len]

        k_full[:cached_len] = k_buf[0, :, :cached_len, :].transpose(0, 1)
        v_full[:cached_len] = v_buf[0, :, :cached_len, :].transpose(0, 1)
        k_full[cached_len:] = (key_chunk.to(torch.float32) @ PiT).to(qdtype)
        v_full[cached_len:] = val_chunk
        q_rot = (query.to(torch.float32) @ PiT).to(qdtype)

        # Unequal q/k lengths with causal=True gives lower-right alignment:
        # query i (absolute position cached_len + i) sees keys 0..cached_len+i.
        #
        # FA2 on ROCm has no sinks parameter, and CK cannot serve head dims
        # above 256, so those cases route through the Triton unified kernel.
        # Q/K are already in Hadamard-rotated space (q_rot, k_full) so the dot
        # products match the original-space attention scores.
        if self.sinks is not None or D > _CK_MAX_HEAD_DIM:
            from vllm.v1.attention.ops.triton_unified_attention import (
                unified_attention,
            )

            out = torch.empty_like(q_rot)
            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=device)
            block_table_single = torch.zeros((1, 1), dtype=torch.int32, device=device)
            window = (
                (self.sliding_window - 1, 0)
                if self.sliding_window and self.sliding_window > 0
                else (-1, -1)
            )
            unified_attention(
                q=q_rot,
                k=k_full.unsqueeze(0),
                v=v_full.unsqueeze(0),
                out=out,
                cu_seqlens_q=cu_q,
                max_seqlen_q=q_len,
                seqused_k=seqused_k,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=window,
                block_table=block_table_single,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                sinks=self.sinks,
            )
            return out.to(qdtype)
        if _HAS_FLASH_ATTN:
            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            return self._flash_attn_varlen(
                q=q_rot,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
                window_size=(
                    (self.sliding_window - 1, 0)
                    if self.sliding_window and self.sliding_window > 0
                    else None
                ),
            )
        q_t = q_rot.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v_full.transpose(0, 1).unsqueeze(0)
        if Hq > Hk:
            k_t = k_t.expand(1, Hq, -1, -1)
            v_t = v_t.expand(1, Hq, -1, -1)
        return (
            F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=self.scale
            )
            .squeeze(0)
            .transpose(0, 1)
            .to(qdtype)
        )
