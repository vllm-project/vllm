# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms.rocm import on_gfx950, on_gfx1250
from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend,
    RocmAttentionImpl,
    RocmAttentionMetadata,
    RocmAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    KVQuantMode,
    get_kv_quant_mode,
    is_quantized_kv_cache,
)

logger = init_logger(__name__)


@dataclass
class _FmhaExtend:
    """Row split of a batch with long extend-prefill requests (built on the host)."""

    long_tok: torch.Tensor  # [n] int64: query rows of the long requests
    long_cu_q: torch.Tensor  # [L+1] int32
    long_cu_k: torch.Tensor  # [L+1] int32
    long_max_q: int
    long_max_k: int
    kv_rows: torch.Tensor  # [sum k] int64: (block * H) * block_size + offset
    short_tok: torch.Tensor | None  # [m] int64, None if every request is long
    short_cu_q: torch.Tensor | None
    short_seq_lens: torch.Tensor | None
    short_block_table: torch.Tensor | None
    short_max_q: int
    short_max_k: int


@dataclass
class RocmAiterUnifiedAttentionMetadata(RocmAttentionMetadata):
    fmha_extend: _FmhaExtend | None = None


def _build_fmha_extend(
    cm: CommonAttentionMetadata,
    block_size: int,
    num_kv_heads: int,
    max_decode_q: int = 1,
) -> _FmhaExtend | None:
    min_q = envs.VLLM_ROCM_UNIFIED_ATTN_FMHA_EXTEND_MIN_Q
    if min_q <= 0 or not on_gfx950():
        return None
    # Only prefill rows have an exact seq_lens_cpu_upper_bound; (async spec)
    # decode rows of up to max_decode_q tokens never take the FMHA path.
    min_q = max(min_q, max_decode_q + 1)
    if cm.max_query_len < min_q:
        return None
    seq_ub = cm.seq_lens_cpu_upper_bound  # exact on prefill rows
    if seq_ub is None or cm.causal is not True:
        return None
    qsl = cm.query_start_loc_cpu.tolist()
    n = cm.num_reqs
    q_lens = [qsl[i + 1] - qsl[i] for i in range(n)]
    long_reqs = [i for i in range(n) if q_lens[i] >= min_q]
    if not long_reqs:
        return None
    short_reqs = [i for i in range(n) if 0 < q_lens[i] < min_q]
    dev = cm.query_start_loc.device
    seq = seq_ub.tolist()

    def rows_of(reqs):
        return torch.cat([torch.arange(qsl[i], qsl[i + 1]) for i in reqs]).to(
            dev, non_blocking=True
        )

    def cumsum(vals):
        out = [0]
        for v in vals:
            out.append(out[-1] + v)
        return torch.tensor(out, dtype=torch.int32).to(dev, non_blocking=True)

    kv_rows = []
    for i in long_reqs:
        t = torch.arange(seq[i], device=dev)
        blocks = cm.block_table_tensor[i].index_select(0, t // block_size).long()
        kv_rows.append(blocks * (num_kv_heads * block_size) + t % block_size)
    ext = _FmhaExtend(
        long_tok=rows_of(long_reqs),
        long_cu_q=cumsum([q_lens[i] for i in long_reqs]),
        long_cu_k=cumsum([seq[i] for i in long_reqs]),
        long_max_q=max(q_lens[i] for i in long_reqs),
        long_max_k=max(seq[i] for i in long_reqs),
        kv_rows=torch.cat(kv_rows),
        short_tok=None,
        short_cu_q=None,
        short_seq_lens=None,
        short_block_table=None,
        short_max_q=0,
        short_max_k=0,
    )
    if short_reqs:
        idx = torch.tensor(short_reqs, dtype=torch.long).to(dev, non_blocking=True)
        ext.short_tok = rows_of(short_reqs)
        ext.short_cu_q = cumsum([q_lens[i] for i in short_reqs])
        ext.short_seq_lens = cm.seq_lens.index_select(0, idx)
        ext.short_block_table = cm.block_table_tensor.index_select(0, idx)
        ext.short_max_q = max(q_lens[i] for i in short_reqs)
        ext.short_max_k = max(seq[i] for i in short_reqs)
    return ext


def _kv_connector_enabled() -> bool:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return False
    kv_transfer_config = vllm_config.kv_transfer_config
    return kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance


class RocmAiterUnifiedAttentionMetadataBuilder(RocmAttentionMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmAttentionMetadata:
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        spec = self.vllm_config.speculative_config
        ext = _build_fmha_extend(
            common_attn_metadata,
            self.block_size,
            self.kv_cache_spec.num_kv_heads,
            1 + (spec.num_speculative_tokens if spec is not None else 0),
        )
        if ext is None:
            return md
        return RocmAiterUnifiedAttentionMetadata(**vars(md), fmha_extend=ext)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> RocmAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        if isinstance(attn_metadata, RocmAiterUnifiedAttentionMetadata):
            # The FMHA row split is built on the host each step; graphs only
            # capture unified attention.
            attn_metadata.fmha_extend = None
        # Avoid capturing the real maximum sequence length while preserving
        # query_start_loc, which unified attention consumes during replay.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata


class RocmAiterUnifiedAttentionBackend(RocmAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 64

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Not implemented
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return True

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_UNIFIED_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RocmAiterUnifiedAttentionImpl"]:
        return RocmAiterUnifiedAttentionImpl

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Keep K and V packed in the content dim, unlike the native HIP
        kernels the base class targets."""
        # block_size == 1 is the per-token page-size probe (see
        # Platform.get_page_size_bytes); real blocks must be gatherable in
        # 16-token units by the ROCm kernel.
        if spec.block_size != 1 and spec.block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return spec

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # K and V come out of the content dim as transposed views rather than
        # copies, so the head dim may sit on either side of the block dim, but
        # the layer must stay outermost.
        if _kv_connector_enabled():
            # Connectors like MoRI assume contiguous blocks
            return (KVCacheLayout.LBHNC,)
        return (KVCacheLayout.LBHNC, KVCacheLayout.LHBNC)

    @staticmethod
    def get_builder_cls() -> type["RocmAiterUnifiedAttentionMetadataBuilder"]:
        return RocmAiterUnifiedAttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """RocmAiterUnifiedAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )


class RocmAiterUnifiedAttentionImpl(RocmAttentionImpl):
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        logger.info_once(
            "Using aiter unified attention for RocmAiterUnifiedAttentionImpl"
        )
        from aiter.ops.triton.unified_attention import unified_attention

        from vllm.platforms.rocm import on_gfx11

        self.unified_attention = unified_attention
        # RDNA3 has no low precision MMA, preserve query precision
        self.supports_quant_query_input = not on_gfx11()

        # Layout flag: must mirror get_kv_cache_shape, which is attn_type
        # agnostic because the allocation is shared across layers.
        self._shuffle_kv_layout = (
            rocm_aiter_ops.is_shuffle_kv_cache_enabled() and on_gfx1250()
        )
        # Kernel flag: the shuffled reader/writer is causal decoder only.
        self._use_shuffle_kv = (
            self._shuffle_kv_layout and self.attn_type != AttentionType.ENCODER_DECODER
        )

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._shuffle_kv_layout:
            # (2, B, N, H, hs) -> ((B, N, H, hs), (B, N, H, hs))
            return kv_cache.unbind(0)
        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
        return kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

    def _shuffle_kv_views(
        self, key_cache: torch.Tensor, value_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nb, bs, nkv, hd = key_cache.shape
        x = 16 // key_cache.element_size()
        kc = key_cache.reshape(nb, nkv, hd // x, bs, x)
        vc = value_cache.reshape(nb, nkv, bs // x, hd, x)
        return kc, vc

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            layer: The attention layer, providing the q/k/v quantization scales.
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
            output: Tensor that the attention result is written into.
            output_scale: Scale for fused output quantization.
            output_block_scale: Block scale for fused output quantization;
                not supported by this backend.

        Returns:
            shape = [num_tokens, num_heads * head_size]

        """
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for RocmAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        key_cache, value_cache = self._split_kv_cache(kv_cache)

        softmax_scale = self.scale
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        if self._use_shuffle_kv:
            key_cache, value_cache = self._shuffle_kv_views(key_cache, value_cache)

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        ext = getattr(attn_metadata, "fmha_extend", None)
        if ext is not None and self._fmha_extend_ok(query, kv_cache, output_scale):
            self._forward_fmha_extend(layer, query, kv_cache, output, ext)
            if ext.short_tok is None:
                return output
            # The remaining rows (decode / short extend) as a compact batch.
            short_out = output.new_empty((ext.short_tok.numel(), *output.shape[1:]))
            self.unified_attention(
                q=query.index_select(0, ext.short_tok),
                k=key_cache,
                v=value_cache,
                out=short_out,
                cu_seqlens_q=ext.short_cu_q,
                max_seqlen_q=ext.short_max_q,
                seqused_k=ext.short_seq_lens,
                max_seqlen_k=ext.short_max_k,
                softmax_scale=softmax_scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=ext.short_block_table,
                softcap=self.logits_soft_cap,
                q_descale=layer._q_scale if query.dtype == self.fp8_dtype else None,
                k_descale=layer._k_scale,
                v_descale=layer._v_scale,
                sinks=self.sinks,
                output_scale=None,
                shuffled_kv_cache=self._use_shuffle_kv,
            )
            output.index_copy_(0, ext.short_tok, short_out)
            return output

        if attn_metadata.causal:
            self.unified_attention(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                q_descale=layer._q_scale if query.dtype == self.fp8_dtype else None,
                k_descale=layer._k_scale,
                v_descale=layer._v_scale,
                sinks=self.sinks,
                output_scale=output_scale,
                shuffled_kv_cache=self._use_shuffle_kv,
            )
        else:
            # The aiter kernel is causal-only. Non-causal cross-attention
            # (ENCODER_DECODER, e.g. Whisper) falls back to the vLLM Triton
            # unified kernel, which shares this layout and honors the flag.
            from vllm.v1.attention.ops.triton_unified_attention import (
                unified_attention as triton_unified_attention,
            )

            descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])
            triton_unified_attention(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                q_descale=layer._q_scale if query.dtype == self.fp8_dtype else None,
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
                sinks=self.sinks,
                output_scale=output_scale,
            )

        return output

    def _fmha_extend_ok(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output_scale: torch.Tensor | None,
    ) -> bool:
        return (
            output_scale is None
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and self.kv_cache_dtype in ("fp8", "fp8_e4m3")
            and not self._shuffle_kv_layout
            and self.alibi_slopes is None
            and self.sinks is None
            and not self.logits_soft_cap
            and tuple(self.sliding_window) == (-1, -1)
            and self.head_size in (64, 128)
            and kv_cache.is_contiguous()
        )

    def _forward_fmha_extend(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        ext: _FmhaExtend,
    ) -> None:
        """Long extend-prefill rows: gather their fp8 K/V (context + new tokens,
        already in the cache) to contiguous tensors and run AITER's FMHA v3 fp8
        varlen kernel, bottom-right causal. The query is the layer's statically
        quantized fp8 query, the same input unified attention would have read."""
        from aiter.ops.mha import _flash_attn_varlen_forward

        hs = self.head_size
        kv = kv_cache.view(self.fp8_dtype)  # [blocks, H, block_size, 2 * hs]
        nkv, block_size = kv.shape[1], kv.shape[2]
        heads = torch.arange(nkv, device=kv.device) * block_size
        rows = (ext.kv_rows[:, None] + heads[None, :]).view(-1)
        sel = kv.view(-1, 2 * hs).index_select(0, rows).view(-1, nkv, 2 * hs)
        k = sel[..., :hs].contiguous()
        v = sel[..., hs:].contiguous()
        q = query.index_select(0, ext.long_tok)
        if q.dtype != self.fp8_dtype:
            q = q.to(self.fp8_dtype)  # unscaled, as a q_scale of 1.0 would
            q_descale = torch.ones(1, dtype=torch.float32, device=q.device)
        else:
            q_descale = layer._q_scale.view(1)
        out = _flash_attn_varlen_forward(
            q,
            k,
            v,
            ext.long_cu_q,
            ext.long_cu_k,
            None,
            None,
            ext.long_max_q,
            ext.long_max_k,
            0,
            0.0,
            self.scale,
            causal=True,
            logits_soft_cap=0.0,
            window_size_left=-1,
            window_size_right=-1,
            sink_size=0,
            bias=None,
            alibi_slopes=None,
            q_descale=q_descale,
            k_descale=layer._k_scale.view(1),
            v_descale=layer._v_scale.view(1),
            return_lse=False,
            return_softmax=False,
        )[0]
        output.index_copy_(0, ext.long_tok, out.to(output.dtype))

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        key_cache, value_cache = self._split_kv_cache(kv_cache)

        if self._use_shuffle_kv:
            from vllm.v1.attention.backends.rocm_aiter_fa import (
                reshape_and_cache_shuffle_triton,
            )

            if get_kv_quant_mode(self.kv_cache_dtype) == KVQuantMode.FP8_PER_TENSOR:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            _dummy = torch.empty(1, dtype=torch.float32, device=key.device)
            reshape_and_cache_shuffle_triton(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                _dummy,
                _dummy,
            )
            return

        # Reshape the input keys and values and store them in the cache.
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        return rocm_aiter_ops.is_enabled()

    def fused_qk_norm_rope_kvcache_supported(self):
        return rocm_aiter_ops.is_enabled()

    def fused_qk_norm_mrope_kvcache_supported(self) -> bool:
        return IS_AITER_FOUND

    def set_interleaved_v_cache(self):
        # No-op override of RocmAttentionImpl: this backend
        # decodes with the AITER triton unified-attention kernel
        pass

    def do_qk_norm_rope_kvcache_update(
        self,
        layer: AttentionLayer,
        qkv: torch.Tensor,
        q_out: torch.Tensor,
        k_out: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        rms_norm_eps: float,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        rocm_aiter_ops.do_qk_norm_rope_kvcache_update(
            qkv=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            num_heads_q=self.num_heads,
            num_heads_k=self.num_kv_heads,
            head_dim=self.head_size,
            is_neox=is_neox,
            rms_norm_eps=rms_norm_eps,
            q_out=q_out,
            k_out=k_out,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=layer_slot_mapping,
            k_scale=layer._k_scale_cpu,
            v_scale=layer._v_scale_cpu,
            kv_cache_dtype=self.kv_cache_dtype,
            use_shuffle_layout=self._use_shuffle_kv,
            return_kv=False,
        )

    def do_qk_norm_mrope_kvcache_update(
        self,
        layer: AttentionLayer,
        qkv: torch.Tensor,
        q_out: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        rms_norm_eps: float,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        mrope_section: tuple[int, int, int],
        is_interleaved: bool,
        rotary_dim: int,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        rocm_aiter_ops.do_qk_norm_mrope_kvcache_update(
            qkv=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            num_heads_q=self.num_heads,
            num_heads_k=self.num_kv_heads,
            head_dim=self.head_size,
            is_neox=is_neox,
            mrope_section=list(mrope_section),
            is_interleaved=is_interleaved,
            rms_norm_eps=rms_norm_eps,
            q_out=q_out,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=layer_slot_mapping,
            k_scale=layer._k_scale_cpu,
            v_scale=layer._v_scale_cpu,
            kv_cache_dtype=self.kv_cache_dtype,
            rotary_dim=rotary_dim,
        )

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        key_cache, value_cache = self._split_kv_cache(kv_cache)

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        if self._use_shuffle_kv:
            key_cache, value_cache = self._shuffle_kv_views(key_cache, value_cache)
        flash_layout = not self._use_shuffle_kv

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
