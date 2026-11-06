# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill."""

from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
    has_native_kv_cache_layout,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    is_quantized_kv_cache,
)

logger = init_logger(__name__)


@dataclass
class RocmAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None

    # DFlash drafting sets this to False via CommonAttentionMetadata.
    causal: bool = True

    # Per-layer dense [num_blocks, num_kv_heads, block_size] scale tables that
    # the AITER asm decode reads for an fp8 cache; None when it is unused.
    k_scale: dict[str, torch.Tensor] | None = None
    v_scale: dict[str, torch.Tensor] | None = None


class RocmAttentionMetadataBuilder(AttentionMetadataBuilder[RocmAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    kv_cache_spec: AttentionSpec

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        self.use_aiter_asm_decode = (
            bool(rocm_aiter_ops.is_mha_enabled())
            and vllm_config.cache_config.cache_dtype in ("fp8", "fp8_e4m3")
            and self.block_size == 16
            and kv_cache_spec.head_size == 128
        )
        self.k_scale: dict[str, torch.Tensor] | None = None
        self.v_scale: dict[str, torch.Tensor] | None = None

    def _build_aiter_kv_scale_tables(self) -> None:
        # Map each layer to the shared table for its per-tensor scale, shaped by
        # this group's spec, once its KV cache is bound; until then keep HIP.
        if self.k_scale is not None or not self.use_aiter_asm_decode:
            return
        from vllm.v1.attention.backends.rocm_aiter_fa import (
            AiterFlashAttentionMetadataBuilder,
        )

        tables = AiterFlashAttentionMetadataBuilder.asm_kv_scale_tables
        attn_layers = self.vllm_config.compilation_config.static_forward_context
        k_scale_tables: dict[str, torch.Tensor] = {}
        v_scale_tables: dict[str, torch.Tensor] = {}
        for layer_name in self.layer_names:
            layer = attn_layers[layer_name]
            kv_cache = getattr(layer, "kv_cache", None)
            if not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0:
                return
            num_blocks = kv_cache.shape[0]
            k_scale_tables[layer_name] = tables.get(
                self.device,
                num_blocks,
                self.kv_cache_spec.num_kv_heads,
                self.block_size,
                layer._k_scale_float,
            )
            v_scale_tables[layer_name] = tables.get(
                self.device,
                num_blocks,
                self.kv_cache_spec.num_kv_heads,
                self.block_size,
                layer._v_scale_float,
            )
        self.k_scale, self.v_scale = k_scale_tables, v_scale_tables

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> RocmAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)

        # Zero device query start locations to avoid invalid memory access in
        # the prefix prefill kernel during graph capture (#25985).
        common_attn_metadata.query_start_loc.zero_()

        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        self._build_aiter_kv_scale_tables()

        attn_metadata = RocmAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=common_attn_metadata.causal,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
        )
        return attn_metadata


class RocmAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
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
        # ROCM paged attention native C++ kernel only supports block sizes 16 and 32
        # due to shared memory (LDS) constraints on AMD GPUs.
        # The (head_size, block_size) pairs that have a native kernel are enumerated
        # by BLOCK_SIZES and VARIANTS in csrc/rocm/generate_paged_attn.py; the
        # runtime gate is use_rocm_custom_paged_attention() in vllm/platforms/rocm.py.
        # However, vLLM allows support for any multiple of 16 via the Triton path.
        # As addressed in PR: https://github.com/vllm-project/vllm/pull/31380,
        # non-standard models (like qwen3-next with block_size 544, or qwen3_5
        # with 784 and 1056) are dynamically routed to our optimized Triton kernel
        # in `do_kv_cache_update`.
        return [MultipleOf(16)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # Only 16 (and 32, where a native kernel exists) runs on the native
        # C++ paged-attention kernel; any other multiple of 16 falls back to
        # the slower Triton path. Keep the native kernel for default-sized
        # runs; explicit --block-size values are unaffected.
        return 16

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Head sizes with a native ROCm kernel are only 64 and 128 (see
        # csrc/rocm/attention_paged_attn_h{64,128}_*.cu); the remaining entries are
        # served by the Triton fallback selected in use_rocm_custom_paged_attention().
        return [32, 64, 80, 96, 128, 160, 192, 224, 256]

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Not implemented
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        # ROCM custom attention kernel does not support sinks.
        # Callink this backend with sinks will cause it to fall back to the Triton
        # kernel, which is less efficient than the proper triton backends.
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        # ROCM_ATTN uses (2, num_blocks, ...) KV cache layout which is
        # incompatible with KV connectors that require blocks-first layout.
        return False

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "ROCM_ATTN"

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["RocmAttentionImpl"]:
        return RocmAttentionImpl

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """ENCODER_DECODER is not supported because
        chunked_prefill_paged_decode's prefill kernel (context_attention_fwd)
        assumes self-attention semantics: it treats passed K/V as new tokens
        to mix with cached K/V. For cross-attention layers the encoder K/V
        are already fully cached, so mixing them again produces incorrect
        results when max_query_len > 1 (e.g. beam search).
        """
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """K and V as two head groups so the native HIP kernels address each side
        as one contiguous region (x-packed interior applied in split_kv_cache)."""
        if spec.state_content_bytes is not None:
            return spec
        assert spec.head_size == spec.head_size_v, (
            "Separate K/V head groups require symmetric K/V head sizes."
        )
        return replace(
            spec,
            num_head_slots=2,
            state_content_bytes=spec.num_kv_heads
            * spec.head_size
            * get_dtype_size(spec.dtype),
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The native HIP kernels hardcode group-contiguous block addressing, so
        # they need the K and V groups to span all blocks (H outermost within the layer)
        # The Triton fallbacks are stride-aware, so block-interior LBHNC is also
        # supported, just without the native kernels.
        return (KVCacheLayout.LHBNC, KVCacheLayout.LBHNC)

    @staticmethod
    def get_builder_cls() -> type["RocmAttentionMetadataBuilder"]:
        return RocmAttentionMetadataBuilder


class RocmAttentionImpl(AttentionImpl):
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
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )

        self._use_interleaved_v_cache = False

        # fp8 cache with 16-token blocks matches the AITER asm decode layout;
        # its ultra-precision kernel only exists for GQA ratios up to 8.
        self._use_aiter_asm_decode = (
            bool(rocm_aiter_ops.is_mha_enabled())
            and attn_type == AttentionType.DECODER
            and kv_cache_dtype in ("fp8", "fp8_e4m3")
            and head_size == 128
            and 1 <= self.num_queries_per_kv <= 8
            and alibi_slopes is None
            and sinks is None
            and self.sliding_window == (-1, -1)
        )
        # Same cut as AITER's own asm/HIP choice; below it the HIP kernel here
        # is as fast and more accurate on an fp8 cache.
        self._aiter_asm_min_heads = (
            2 * num_compute_units() if self._use_aiter_asm_decode else 0
        )
        if self._use_aiter_asm_decode:
            logger.info_once(
                "ROCM_ATTN: using AITER asm paged attention for pure-decode "
                "batches above %d query heads with an fp8 KV cache.",
                self._aiter_asm_min_heads,
            )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer

        """
        # For encoder attention, process FP8 quantization if needed
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
            sinks=self.sinks,
        )
        return output

    def _forward_aiter_asm_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
    ) -> None:
        # V [blocks, kv_heads, head_size, 16] is AITER's shuffled V with one
        # 16-token group, so unsqueeze gives its [blocks, kv_heads, 1, hs, 16].
        assert attn_metadata.k_scale is not None
        assert attn_metadata.v_scale is not None
        num_seqs = attn_metadata.num_actual_tokens
        block_table = attn_metadata.block_table[:num_seqs]
        rocm_aiter_ops.pa_fwd_asm(
            Q=query[:num_seqs],
            K=key_cache,
            V=value_cache.unsqueeze(2),
            block_tables=block_table,
            context_lens=attn_metadata.seq_lens[:num_seqs],
            block_tables_stride0=block_table.stride(0),
            K_QScale=attn_metadata.k_scale[layer.layer_name],
            V_QScale=attn_metadata.v_scale[layer.layer_name],
            out_=output[:num_seqs],
            high_precision=2,
        )

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
            kv_cache: logical [num_blocks, 2, block_size, num_kv_heads *
                head_size] under LHBNC (physically K/V-group-first)
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

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # The bound view is logical [B, 2, N, H*hs]; split_kv_cache expects
        # the K/V groups first.
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size
        )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            # q_scale only applies to an fp8 query; this path keeps the query
            # in full precision, so a non-1.0 q_scale is not applicable here.
            if query.dtype == self.fp8_dtype and layer._q_scale_float != 1.0:
                raise NotImplementedError(
                    "A non 1.0 q_scale with an fp8 query is not currently "
                    "supported by RocmAttentionImpl."
                )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        if (
            self._use_aiter_asm_decode
            and max_seqlen_q == 1
            and num_actual_tokens * self.num_heads > self._aiter_asm_min_heads
            and attn_metadata.k_scale is not None
            and attn_metadata.causal
            and output_scale is None
            and query.dtype == torch.bfloat16
            and max_seqlen_k <= 128 * 1024
            and key_cache.shape[3] == 16
            and has_native_kv_cache_layout(key_cache, value_cache)
        ):
            self._forward_aiter_asm_decode(
                layer, query, key_cache, value_cache, output, attn_metadata
            )
            return output

        # Compute attention and update output up to `num_actual_tokens`.
        chunked_prefill_paged_decode(
            query=query[:num_actual_tokens],
            key=key[:num_actual_tokens] if key is not None else None,
            value=value[:num_actual_tokens] if value is not None else None,
            output=output[:num_actual_tokens],
            kv_cache_dtype=self.kv_cache_dtype,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            query_start_loc=cu_seqlens_q,
            seq_lens=seqused_k,
            max_seq_len=max_seqlen_k,
            max_query_len=max_seqlen_q,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
            alibi_slopes=self.alibi_slopes,
            # self.sliding_window[0] is the FlashAttention-style left span (W - 1).
            # chunked_prefill_paged_decode expects the full window length W.
            sliding_window=1 + self.sliding_window[0],
            sm_scale=self.scale,
            output_scale=output_scale,
            sinks=self.sinks,
            use_interleaved_v_cache=self._use_interleaved_v_cache,
            causal=attn_metadata.causal,
        )

        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size
        )

        # Reshape the input keys and values and store them in the cache.
        # Get the actual block_size from value_cache
        # value_cache shape: [num_blocks, num_heads, head_size, block_size]
        block_size = value_cache.shape[3]
        has_native_layout = has_native_kv_cache_layout(key_cache, value_cache)

        # Match the fused writer's interleaved V-cache when fusion is active.
        interleaved_v = self._use_interleaved_v_cache
        interleaved_v_pack_factor = (
            (16 // value_cache.element_size()) if interleaved_v else 0
        )

        if block_size in (16, 32) and has_native_layout:
            # Normal 16, 32 with contiguous blocks: use vLLM native HIP C++ logic.
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
                interleaved_v,
            )
        else:
            # Non-standard blocks and hybrid attention/Mamba layouts need the
            # stride-aware Triton writer. The native reshape_and_cache kernel
            # assumes contiguous block storage and writes to the wrong hybrid
            # cache blocks.
            triton_reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
                interleaved_v_pack_factor,
            )

    def fused_qk_norm_rope_kvcache_supported(self):
        return rocm_aiter_ops.is_enabled()

    def set_interleaved_v_cache(self):
        # Decode/prefill kernels must read the interleaved V-cache the AITER
        # fused write emits when use_shuffle_layout=True.
        self._use_interleaved_v_cache = True

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
        # kv_cache is logical [num_blocks, 2, block_size, num_kv_heads * head_size]
        # (see forward); the AITER shuffle writer takes one (B, N, H, hs) per side.
        key_cache, value_cache = kv_cache.unflatten(
            -1, (self.num_kv_heads, self.head_size)
        ).unbind(1)
        use_shuffle_layout = self._use_interleaved_v_cache
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
            use_shuffle_layout=use_shuffle_layout,
            return_kv=True,
        )

    def fused_rope_kvcache_supported(self):
        return rocm_aiter_ops.is_enabled()

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
            return
        if self._use_interleaved_v_cache:
            # The Triton RoPE + cache writer only stores the regular V layout.
            # With layer_name hoisted, one RoPE + KV-cache pattern serves every
            # layer, so a layer switched to the interleaved V cache is handled
            # here at run time rather than left out of the fusion.
            ops.rotary_embedding(
                positions,
                query,
                key,
                layer.head_size,  # type: ignore[attr-defined]
                cos_sin_cache,
                is_neox,
            )
            self.do_kv_cache_update(layer, key, value, kv_cache, layer_slot_mapping)
            return
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1),
            layer.num_kv_heads,  # type: ignore[attr-defined]
            layer.head_size,  # type: ignore[attr-defined]
        )
        flash_layout = False

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

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
