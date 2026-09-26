# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UltraQuant 4-bit KV-cache attention backend.

UltraQuant is a 4-bit KV-cache format (FP4 E2M1 codes + UE8M0 power-of-two
per-group-32 scales) with Hadamard-rotated keys. It reuses the TurboQuant
attention scaffolding (forward orchestration, metadata, cache-layout addressing,
CUDA-graph pre-warm) by subclassing, and overrides only the format-specific
seams:

  * store         -> ultraquant_store (rotate K, pack FP4 + UE8M0 scales)
  * decode        -> FlyDSL D=256 scaled-MFMA kernel, Triton unified fallback
  * continuation  -> small chunk: Triton unified; large chunk: dequant + FA

``UltraQuantAttentionBackend`` is a standalone backend with its own identity
(``ULTRAQUANT``) and dtype (``ultraquant_4bit``); the TurboQuant backend/impl
carry no UltraQuant knowledge.
"""

import contextlib
import math
from dataclasses import replace
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.turboquant_attn import (
    _CONTINUATION_DECODE_THRESHOLD,
    _HAS_FLASH_ATTN,
    TurboQuantAttentionBackend,
    TurboQuantAttentionImpl,
    TurboQuantMetadata,
    _build_hadamard,
)
from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    flydsl_ultraquant_decode_attention,
    is_ultraquant_flydsl_available,
    ultraquant_flydsl_decode_eligible,
)
from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.triton_unified_attention import (
    ultraquant_unified_attention,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

# CK FlashAttention on ROCm has no compiled kernel above head_dim 256 and no
# sinks argument; those cases route through the Triton unified kernel instead.
_CK_MAX_HEAD_DIM = 256


class UltraQuantAttentionBackend(TurboQuantAttentionBackend):
    """Standalone backend for the ``ultraquant_4bit`` KV-cache dtype.

    Subclasses :class:`TurboQuantAttentionBackend` for the shared metadata and
    cache-layout plumbing, and overrides identity, spec, impl, block size, and
    capabilities so no UltraQuant knowledge lives in the TurboQuant backend.
    """

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["ultraquant_4bit"]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """UltraQuant packs K+V into one 4-bit FP4/UE8M0 slot per head."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_ultraquant:
            return spec
        return replace(spec, state_content_bytes=slot_size(spec.head_size))

    @staticmethod
    def get_name() -> str:
        return "ULTRAQUANT"

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # FlyDSL D=256 decode geometry requires a 64-token KV block; pin it so
        # the KV-cache manager pages stay dense and splittable into kernel blocks.
        return 64

    @staticmethod
    def get_impl_cls() -> type["UltraQuantAttentionImpl"]:
        return UltraQuantAttentionImpl

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "ultraquant_4bit"

    @classmethod
    def supports_sink(cls) -> bool:
        # The FlyDSL fast path is marked ineligible when sinks are present; the
        # Triton unified fallback implements sinks, so the backend supports them.
        return True


class UltraQuantAttentionImpl(TurboQuantAttentionImpl):
    """UltraQuant 4-bit KV-cache attention (FP4 E2M1 + UE8M0, rotated K).

    Overrides the init, on-device setup, store, decode, and prefill/continuation
    seams of :class:`TurboQuantAttentionImpl`; the shared ``forward`` /
    ``do_kv_cache_update`` orchestration and metadata handling are inherited.
    """

    # FlyDSL D=256 decode loads query as bf16; fp16 is not supported.
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.fa_version = get_flash_attn_version(head_size=head_size)

        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        self.sliding_window = sliding_window
        self.sinks = kwargs.get("sinks")
        self._max_model_len = vllm_config.model_config.max_model_len

        self._use_ultraquant_flydsl = is_ultraquant_flydsl_available()
        # FlyDSL decode eligibility is static per layer config; compute once.
        self._flydsl_decode_eligible = ultraquant_flydsl_decode_eligible(
            head_size=self.head_size,
            num_kv_groups=self.num_kv_groups,
            has_sinks=self.sinks is not None,
            sliding_window=self.sliding_window,
            flydsl_loaded=self._use_ultraquant_flydsl,
        )

    def _ensure_on_device(self, layer, device):
        """Build the shared Hadamard rotation and pre-warm FlyDSL buffers.

        UltraQuant needs no Lloyd-Max centroids, but the inherited forward reads
        ``layer._tq_centroids``, so an empty tensor is kept on the layer.
        """
        if self._use_ultraquant_flydsl:
            # CUDA-graph capture safety for the FlyDSL decode path on ROCm:
            # pre-allocate index buffers and grow the workspace to its max size
            # before capture so replay never lands on stale addresses.
            _max_len = self._max_model_len
            _already_ok = (
                hasattr(self, "_arange_cache")
                and self._arange_cache.device.type == str(device).split(":")[0]
                and self._arange_cache.shape[0] >= _max_len + 2
            )
            if not _already_ok:
                self._arange_cache = torch.arange(
                    0, _max_len + 2, device=device, dtype=torch.int32
                )
            if not hasattr(self, "_cu_2") or self._cu_2.device != torch.device(device):
                self._cu_2 = torch.zeros(2, device=device, dtype=torch.int32)
            if (
                is_workspace_manager_initialized()
                and not current_workspace_manager().is_locked()
            ):
                B_max = self._max_capture_batch_size()
                D = self.head_size
                Hq = self.num_heads
                S = self.max_num_kv_splits
                _pre_warm_bytes = (
                    B_max * Hq * (S * (D + 1) + D) * 4  # fp32 mid_o + fp32 lse
                    + B_max * Hq * D * 2  # query-dtype output (bf16 = 2 B)
                    + 512  # alignment padding
                )
                with contextlib.suppress(AssertionError):
                    current_workspace_manager().get_simultaneous(
                        ((_pre_warm_bytes,), torch.uint8)
                    )

        if not hasattr(layer, "_tq_cached"):
            D = self.head_size
            # Pure Hadamard: orthonormal + symmetric (H = H^T), enabling
            # in-kernel butterfly fusion and trivial inverse for continuation.
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = H
            layer._tq_Pi = H
            layer._tq_Pi_half = H.to(torch.float16)
            # Inherited TurboQuantAttentionImpl.forward reads _tq_centroids
            # unconditionally; UltraQuant has no centroid table, so an empty
            # tensor satisfies the read.
            layer._tq_centroids = torch.empty(0, device=device, dtype=torch.float32)
            layer._tq_cached = True

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
        # Acquire shared decode scratch from the WorkspaceManager (UltraQuant
        # only needs the output buffer; the mid_o/lse slots keep the triple
        # allocation identical to the base decode for buffer reuse).
        B = query.shape[0]
        D = self.head_size
        S = self.max_num_kv_splits
        Hq = self.num_heads
        output_buf = None
        if is_workspace_manager_initialized():
            _mid_o, output_buf, _lse = current_workspace_manager().get_simultaneous(
                ((B, Hq, S, D + 1), torch.float32),
                ((B, Hq, D), query.dtype),
                ((B, Hq), torch.float32),
            )

        if self._flydsl_decode_eligible:
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
            self.num_kv_groups,
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

    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape

        # Fast path: flash_attn for first-chunk prefills (all K/V in batch).
        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )

        # Continuation or no flash_attn: per-request attention.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        _max_seq = attn_metadata.max_seq_len
        _ac: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if _ac is None or _ac.shape[0] <= _max_seq:
            _ac = torch.arange(
                0, _max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._arange_cache = _ac
        _arange_cache: torch.Tensor = _ac

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if _HAS_FLASH_ATTN:
                    self._cu_2[1:2] = q_len
                    cu = self._cu_2
                    out = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: prefix already stored (Hadamard-rotated
                # FP4) by do_kv_cache_update. Small chunks reuse the Triton
                # unified decode-shaped kernel; large chunks dequant the prefix
                # once and run a dense flash-attention prefill.
                cached_len = seq_len - q_len
                if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                    synth_seq_lens = _arange_cache[cached_len + 1 : seq_len + 1]
                    synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                    out = ultraquant_unified_attention(
                        query=q_seq,
                        kv_cache=kv_cache,
                        block_table=synth_bt,
                        seq_lens=synth_seq_lens,
                        query_start_loc=_arange_cache[: q_len + 1],
                        scale=self.scale,
                        PiT=PiT,
                        max_query_len=1,
                        max_seq_len=seq_len,
                        sinks=self.sinks,
                        sliding_window=self.sliding_window,
                    )
                else:
                    out = self._ultraquant_continuation_prefill(
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
                output[q_start:q_end] = out.to(query.dtype)

        return output

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
