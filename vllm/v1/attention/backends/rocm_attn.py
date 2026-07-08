# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import is_quantized_kv_cache
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
from vllm.v1.attention.backends.utils import (
    compute_mm_prefix_range_tensor,
    get_kv_cache_layout,
)
from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode, get_kv_quant_mode

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

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None
    rswa_prefix_lens: torch.Tensor | None = None
    rswa_window: int | None = None

    # Decode scheduling buffers used by the shared Triton unified-attention
    # kernel. ROCM_ATTN owns a different cache shape from TRITON_ATTN, but the
    # runtime scheduling contract is intentionally the same.
    seq_threshold_3D: int = 0
    num_par_softmax_segments: int = 0
    softmax_segm_output: torch.Tensor | None = None
    softmax_segm_max: torch.Tensor | None = None
    softmax_segm_expsum: torch.Tensor | None = None
    has_decode: bool = False

    # DFlash drafting sets this to False via CommonAttentionMetadata.
    causal: bool = True


class RocmAttentionMetadataBuilder(AttentionMetadataBuilder[RocmAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    @staticmethod
    def _decode_split_kv_segments(num_kv_heads: int) -> int:
        """KV segments for the 3D split-KV (flash-decoding) decode.

        The decode kernel needs enough launch-grid thread blocks to saturate the
        GPU. The 2D single-pass grid is ``num_seqs * num_kv_heads`` blocks; when a
        batch is smaller than that, the 3D kernel splits each (sequence, kv-head)
        into this many KV segments to make up the difference. The target is ~128
        blocks per sequence -- the MI300 saturation point (rocprof, Llama-70B TP8,
        1 KV head, 9k context: decode attention runs 43.9 / 27.2 / 19.6 us at
        16 / 32 / 128 segments). Deriving the count from ``num_kv_heads`` keeps a
        single occupancy target across models with no per-model tuning. This is
        the same 128-block target and 2D/3D threshold TRITON_ATTN uses; TRITON_ATTN
        fixes its segment count instead, which under-fills low-KV-head shards.
        """
        launch_grid_target = 128
        return max(1, launch_grid_target // num_kv_heads)

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

        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )
        # Segments for the 3D split-KV decode, and -- the same target-derived
        # value -- the batch size above which the 2D single-pass grid is already
        # large enough to skip the split. See _decode_split_kv_segments.
        self.num_par_softmax_segments = self._decode_split_kv_segments(
            self.num_heads_kv
        )
        self.seq_threshold_3D = self.num_par_softmax_segments
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."
            # The pre-allocated reduction buffers are indexed by captured batch
            # size, so snap the 2D/3D threshold to the nearest captured size.
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )
        headdim_padded = next_power_of_2(self.headdim)
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )

        self.rswa_window = model_config.rswa_window
        self.persistent_rswa_prefix_lens: torch.Tensor | None = None
        if self.rswa_window is not None:
            self.persistent_rswa_prefix_lens = torch.empty(
                vllm_config.scheduler_config.max_num_seqs,
                dtype=torch.int32,
                device=device,
            )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> RocmAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)

        # Here we set the query start locs to 0. This is to
        # cover up an invalid memory access in the prefix_prefil kernel
        # that we run into during graph capture (#25985)
        common_attn_metadata.query_start_loc.zero_()
        common_attn_metadata.query_start_loc_cpu.zero_()

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
        num_reqs = common_attn_metadata.num_reqs
        query_lens_cpu = (
            common_attn_metadata.query_start_loc_cpu[1 : num_reqs + 1]
            - common_attn_metadata.query_start_loc_cpu[:num_reqs]
        )
        has_decode = bool((query_lens_cpu == 1).any())

        use_cascade = common_prefix_len > 0
        prefix_scheduler_metadata = None

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None

        attn_metadata = RocmAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            causal=common_attn_metadata.causal,
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
            has_decode=has_decode,
        )

        mm_ranges = common_attn_metadata.mm_req_doc_ranges
        if mm_ranges is not None:
            attn_metadata.mm_prefix_range = mm_ranges
            attn_metadata.mm_prefix_range_tensor = compute_mm_prefix_range_tensor(
                mm_ranges,
                len(seq_lens),
                seq_lens.device,
            )

        rswa_prefix_lens = common_attn_metadata.rswa_prefix_lens
        if self.rswa_window is not None and rswa_prefix_lens is not None:
            assert self.persistent_rswa_prefix_lens is not None
            rswa_prefix_lens = rswa_prefix_lens.to(
                device=self.device, dtype=torch.int32, non_blocking=True
            )
            persistent_prefix_lens = self.persistent_rswa_prefix_lens[: len(seq_lens)]
            persistent_prefix_lens.copy_(rswa_prefix_lens[: len(seq_lens)])
            attn_metadata.rswa_prefix_lens = persistent_prefix_lens
            attn_metadata.rswa_window = self.rswa_window

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
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # ROCM paged attention native C++ kernel only supports block sizes 16 and 32
        # due to shared memory (LDS) constraints on AMD GPUs.
        # See csrc/rocm/attention.cu CALL_CUSTOM_LAUNCHER_BLK macro.
        # However, vLLM allows support for any multiple of 16 via the Triton path.
        # As addressed in PR: https://github.com/vllm-project/vllm/pull/31380,
        # non-standard models (like qwen3-next with block_size 544, or qwen3_5
        # with 784 and 1056) are dynamically routed to our optimized Triton kernel
        # in `do_kv_cache_update`.
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 80, 96, 128, 160, 192, 224, 256, 512]

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return True

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "ROCM_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RocmAttentionImpl"]:
        return RocmAttentionImpl

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """ROCM_ATTN currently supports self-attention style cache ownership."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # Logical: (num_layers, num_blocks, num_kv_heads, block_size, 2D)
            # Physical: (num_blocks, num_layers, block_size, num_kv_heads, 2D)
            return (1, 0, 3, 2, 4)
        if cache_layout == "NHD":
            # Logical: (num_blocks, num_kv_heads, block_size, 2D)
            # Physical: (num_blocks, block_size, num_kv_heads, 2D)
            return (0, 2, 1, 3)
        if cache_layout == "HND" and include_num_layers_dimension:
            # Logical: (num_layers, num_blocks, num_kv_heads, block_size, 2D)
            # Physical: (num_blocks, num_kv_heads, num_layers, block_size, 2D)
            return (1, 2, 0, 3, 4)
        if cache_layout == "HND":
            return (0, 1, 2, 3)
        raise ValueError(f"Unknown cache layout: {cache_layout}")

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

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
        self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)
        self._cached_kv_cache: torch.Tensor | None = None
        self._cached_key_cache: torch.Tensor | None = None
        self._cached_value_cache: torch.Tensor | None = None
        self._cached_prefix_key_cache: torch.Tensor | None = None
        self._cached_prefix_value_cache: torch.Tensor | None = None

    def _split_kv_cache_for_update(
        self,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return NHD K/V views from content-packed ROCM_ATTN cache.

        Logical ROCM_ATTN cache is
        ``[num_blocks, num_kv_heads, block_size, 2 * head_size]``. Writers that
        use the flash-style cache contract take separate
        ``[num_blocks, block_size, num_kv_heads, head_size]`` K/V tensors, so
        this split preserves the shared storage and exposes the expected view.
        """
        if kv_cache is not self._cached_kv_cache:
            self._cached_kv_cache = kv_cache
            key_cache = kv_cache[..., : self.head_size]
            value_cache = kv_cache[..., self.head_size :]
            self._cached_key_cache = key_cache.transpose(1, 2)
            self._cached_value_cache = value_cache.transpose(1, 2)
            x = 16 // kv_cache.element_size()
            self._cached_prefix_key_cache = torch.as_strided(
                key_cache,
                (
                    key_cache.shape[0],
                    key_cache.shape[1],
                    self.head_size // x,
                    key_cache.shape[2],
                    x,
                ),
                (
                    key_cache.stride(0),
                    key_cache.stride(1),
                    x * key_cache.stride(3),
                    key_cache.stride(2),
                    key_cache.stride(3),
                ),
            )
            self._cached_prefix_value_cache = torch.as_strided(
                value_cache,
                (
                    value_cache.shape[0],
                    value_cache.shape[1],
                    self.head_size,
                    value_cache.shape[2],
                ),
                (
                    value_cache.stride(0),
                    value_cache.stride(1),
                    value_cache.stride(3),
                    value_cache.stride(2),
                ),
            )

        assert self._cached_key_cache is not None
        assert self._cached_value_cache is not None
        return self._cached_key_cache, self._cached_value_cache

    def _split_kv_cache_for_prefix_prefill(
        self,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return prefix-prefill K/V views backed by content-packed storage."""
        self._split_kv_cache_for_update(kv_cache)
        assert self._cached_prefix_key_cache is not None
        assert self._cached_prefix_value_cache is not None
        return self._cached_prefix_key_cache, self._cached_prefix_value_cache

    def _split_kv_cache_for_unified_attention(
        self,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unified-attention K/V views backed by content-packed storage."""
        return self._split_kv_cache_for_update(kv_cache)

    def _can_use_prefix_prefill_fast_path(
        self,
        attn_metadata: RocmAttentionMetadata,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
    ) -> bool:
        # Only pure-prefill batches take context_attention_fwd. Mixed
        # prefill+decode batches stay on unified_attention: although
        # context_attention_fwd is faster on the prefill rows in isolation,
        # routing the whole mixed batch through it measured ~1.5% slower
        # end-to-end on 8k/1k TP8 (the decode rows lose more than the prefill
        # rows gain), so the has_decode gate is kept.
        return (
            attn_metadata.max_query_len > 1
            and not attn_metadata.has_decode
            and key is not None
            and value is not None
            and attn_metadata.mm_prefix_range_tensor is None
            and attn_metadata.rswa_prefix_lens is None
            and not self.logits_soft_cap
        )

    def _can_use_native_decode_fast_path(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output_scale: torch.Tensor | None,
    ) -> bool:
        """Return whether content-packed decode should use ROCm native pages.

        The native ROCm paged-attention kernel only wins for short-context
        decode. For long context the shared Triton unified-attention path is
        faster because its split-softmax (3D) reduction parallelizes the long
        KV scan, while the native kernel serializes it per query-head. This was
        measured with rocprof on a clean MI300 (bf16, GQA>4, 8 seqs):

            ctx   native (paged_attention_rocm)   unified (3D split)
            1k    ~25 us                           ~38 us   -> native
            8k    ~81 us  (Qwen) / ~32 us (Llama)  ~41/23us -> unified

        So gate native to short context only. Keep it narrow otherwise so
        feature-bearing paths continue through the stride-aware Triton kernel.
        """
        if (
            attn_metadata.max_query_len != 1
            or self.sinks is not None
            or self.logits_soft_cap
            or output_scale is not None
            or self.kv_cache_dtype != "auto"
            or self.num_queries_per_kv <= 4
            or self.sliding_window != (-1, -1)
            or attn_metadata.rswa_prefix_lens is not None
            or attn_metadata.causal is not True
        ):
            return False

        # NOTE: mm_prefix_range only enables bidirectional attention *within* an
        # image span during prefill. This gate requires max_query_len == 1, so
        # every query is a single decode token positioned after all prompt
        # (image) tokens; the bidirectional range is a no-op for those queries
        # and native decode matches unified attention exactly. Multimodal decode
        # (e.g. Qwen2-VL) can therefore still take the native short-context path
        # instead of being excluded just because the request carries an image.

        block_size = kv_cache.shape[2]
        if block_size not in (16, 32):
            return False

        num_seqs = attn_metadata.seq_lens.shape[0]
        if num_seqs > attn_metadata.seq_threshold_3D:
            return False

        # Native only wins for short context; long context stays on unified,
        # which matches the 2D/3D switch inside unified_attention itself. 2048 is
        # the rocprof crossover on MI300 where unified's KV-parallel 3D reduction
        # starts beating the single-pass native paged-attention kernel.
        native_decode_max_seq_len = 2048
        return attn_metadata.max_seq_len <= native_decode_max_seq_len

    def _native_decode(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
    ) -> torch.Tensor:
        prefix_key_cache, prefix_value_cache = self._split_kv_cache_for_prefix_prefill(
            kv_cache
        )
        num_seqs = attn_metadata.block_table.shape[0]
        block_size = kv_cache.shape[2]
        max_num_partitions = (attn_metadata.max_seq_len + 255) // 256
        tmp_output = torch.empty(
            (num_seqs, self.num_heads, max_num_partitions, self.head_size),
            dtype=query.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            (num_seqs, self.num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

        ops.paged_attention_rocm(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            prefix_key_cache,
            prefix_value_cache,
            self.num_kv_heads,
            scale=self.scale,
            block_tables=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            query_start_loc=attn_metadata.query_start_loc,
            block_size=block_size,
            max_seq_len=attn_metadata.max_seq_len,
            alibi_slopes=self.alibi_slopes,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
        )
        return output

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
        )
        return output

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
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, num_kv_heads, block_size, 2 * head_size]
            attn_metadata: Metadata for attention.
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

        assert attn_metadata.use_cascade is False

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

        key_cache, value_cache = self._split_kv_cache_for_unified_attention(kv_cache)

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

        if self._can_use_prefix_prefill_fast_path(attn_metadata, key, value):
            prefix_key_cache, prefix_value_cache = (
                self._split_kv_cache_for_prefix_prefill(kv_cache)
            )
            context_attention_fwd(
                q=query[:num_actual_tokens],
                k=key[:num_actual_tokens],
                v=value[:num_actual_tokens],
                o=output[:num_actual_tokens],
                kv_cache_dtype=self.kv_cache_dtype,
                k_cache=prefix_key_cache,
                v_cache=prefix_value_cache,
                b_loc=attn_metadata.block_table,
                b_start_loc=attn_metadata.query_start_loc,
                b_seq_len=attn_metadata.seq_lens,
                max_seq_len=attn_metadata.max_seq_len,
                max_input_len=attn_metadata.max_query_len,
                k_scale=layer._k_scale,
                v_scale=layer._v_scale,
                alibi_slopes=self.alibi_slopes,
                sliding_window=self.sliding_window[0],
                sm_scale=self.scale,
                fp8_out_scale=output_scale,
                sinks=self.sinks,
                causal=attn_metadata.causal,
            )
            return output

        if self._can_use_native_decode_fast_path(
            kv_cache, attn_metadata, output_scale
        ):
            self._native_decode(
                layer,
                query[:num_actual_tokens],
                kv_cache,
                output[:num_actual_tokens],
                attn_metadata,
            )
            return output

        q_descale = (
            layer._q_scale
            if (
                self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
                and query.dtype == self.fp8_dtype
            )
            else None
        )

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            q_descale=q_descale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=attn_metadata.mm_prefix_range_tensor,
            rswa_prefix_lens=attn_metadata.rswa_prefix_lens,
            rswa_window=attn_metadata.rswa_window,
            kv_quant_mode=self._kv_quant_mode,
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
        key_cache, value_cache = self._split_kv_cache_for_update(kv_cache)

        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

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
        key_cache, value_cache = self._split_kv_cache_for_update(kv_cache)
        flash_layout = True

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
