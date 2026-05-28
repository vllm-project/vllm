# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

from dataclasses import dataclass
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import async_tensor_h2d, is_quantized_kv_cache
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
    get_kv_cache_layout,
    get_num_attention_heads_from_layers,
)
from vllm.v1.attention.ops.triton_per_token_head_attention import (
    triton_per_token_head_attention,
    triton_per_token_head_prefill,
)
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
    kv_cache_uses_per_token_head_scales,
)

logger = init_logger(__name__)

_CONTINUATION_DECODE_THRESHOLD = 128


# constants
MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments


@dataclass
class TritonAttentionMetadata:
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

    seq_threshold_3D: int
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor

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

    all_pure_first_prefill: bool = False

    num_decodes: int = 0
    num_decode_tokens: int = 0
    prefill_is_first_chunk: bool = False

    seq_lens_cpu: torch.Tensor | None = None
    query_start_loc_cpu: torch.Tensor | None = None

    # Per-token-head kernel inputs: precomputed on CPU in build() and copied
    # into pre-allocated GPU buffers so pointers are stable across CUDA graph
    # capture/replay. Slices into the builder-owned buffers.
    q_to_req: torch.Tensor | None = None
    q_to_klen: torch.Tensor | None = None

    @staticmethod
    def compute_mm_prefix_range_tensor(
        mm_prefix_range: dict[int, list[tuple[int, int]]] | None,
        num_seqs: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Convert mm_prefix_range dict to padded tensor for Triton kernel.

        Returns shape: (num_seqs, max_ranges, 2) with 0-padding for empty ranges.
        Empty ranges have start==end==0, which kernel skips via is_valid check.
        """
        if mm_prefix_range is None:
            return None

        # Collect ranges, using [(0,0)] for empty sequences to ensure uniform dims
        range_lists = [
            mm_prefix_range.get(i, [(0, 0)]) or [(0, 0)] for i in range(num_seqs)
        ]

        # Return None if all ranges are trivial (only (0,0) placeholders)
        if all(r == [(0, 0)] for r in range_lists):
            return None

        # Build on CPU first then move to GPU in a single H2D transfer
        max_ranges = max(len(r) for r in range_lists)
        # Pad all sequences to the same number of ranges
        padded = []
        for r in range_lists:
            padded_r = list(r) + [(0, 0)] * (max_ranges - len(r))
            padded.append(padded_r)
        # Build on pinned CPU memory so the H2D transfer is non-blocking.
        padded = async_tensor_h2d(padded, dtype=torch.int32, device=device)
        return padded.view(num_seqs, max_ranges, 2)


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self._is_per_token_head = kv_cache_spec.kv_quant_mode.is_per_token_head
        if self._is_per_token_head:
            self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
            # Persistent GPU buffers for the kernel's per-query maps. Sized
            # to max_num_batched_tokens — the scheduler's upper bound on
            # tokens per forward. Pointers stay stable across CUDA graph
            # capture/replay; build() writes contents via .copy_().
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self._q_to_req_buf = torch.empty(
                max_tokens, dtype=torch.int32, device=device
            )
            self._q_to_klen_buf = torch.empty(
                max_tokens, dtype=torch.int32, device=device
            )

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        # Compatible with models with non-uniform per-layer head counts.
        self.num_heads_q = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        # Check if CUDA Graphs are enabled for decode
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
        # A lower bound for num_q_blocks is the number of sequences.
        # To ensure the minimum launch grid size is achieved, the number of sequences
        # must be at least equal to the threshold below.
        # If this threshold is not reached (i.e., the batch size is not large enough),
        # the 3D kernel will be selected instead.
        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        # Modify the threshold if needed.
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."

            # Select the CUDA Graph capture size closest to self.seq_threshold_3D
            # as threshold. This ensures that each captured graph covers the
            # correct execution path.
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
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

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        attn_metadata.all_pure_first_prefill = False
        attn_metadata.prefill_is_first_chunk = False
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0

        # Per-request check.  Only runs when the scheduler already
        # materialized the CPU copy of seq_lens — otherwise we skip the
        # fast-path gate to avoid triggering a D2H sync here.
        seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        qsl_cpu = None
        num_decodes = 0
        num_decode_tokens = 0
        prefill_is_first_chunk = False
        if seq_lens_cpu is not None:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1:] - qsl_cpu[:-1]
            all_pure_first_prefill = bool(
                torch.equal(query_lens_cpu, seq_lens_cpu.to(query_lens_cpu.dtype))
            )
            if self._is_per_token_head:
                decode_mask = query_lens_cpu <= 1
                if bool(decode_mask.all()):
                    num_decodes = int(query_lens_cpu.shape[0])
                elif not bool(decode_mask[0]):
                    num_decodes = 0
                else:
                    num_decodes = int(decode_mask.to(torch.int32).sum().item())
                num_decode_tokens = int(qsl_cpu[num_decodes].item())
                if num_decodes < query_lens_cpu.shape[0]:
                    ql_pref = query_lens_cpu[num_decodes:]
                    sl_pref = seq_lens_cpu[num_decodes:].to(ql_pref.dtype)
                    prefill_is_first_chunk = bool(torch.equal(ql_pref, sl_pref))

                # Compute per-query maps on CPU and stage into persistent
                # GPU buffers. Pointers are stable; only contents change.
                q_lens_i32 = query_lens_cpu.to(torch.int32)
                num_reqs_total = q_lens_i32.shape[0]
                total_q = int(qsl_cpu[-1].item())
                if total_q > 0:
                    if num_reqs_total == total_q:
                        # Pure decode fast path: q_to_req = arange,
                        # q_to_klen = seq_lens.
                        q_to_req_cpu = torch.arange(num_reqs_total, dtype=torch.int32)
                        q_to_klen_cpu = seq_lens_cpu.to(torch.int32)
                    else:
                        qsl_i32 = qsl_cpu[:-1].to(torch.int32)
                        seq_lens_i32 = seq_lens_cpu.to(torch.int32)
                        q_to_req_cpu = torch.repeat_interleave(
                            torch.arange(num_reqs_total, dtype=torch.int32),
                            q_lens_i32,
                        )
                        cached_len_per_req = seq_lens_i32 - q_lens_i32
                        pos_in_req = (
                            torch.arange(total_q, dtype=torch.int32)
                            - qsl_i32[q_to_req_cpu.long()]
                        )
                        q_to_klen_cpu = (
                            cached_len_per_req[q_to_req_cpu.long()] + pos_in_req + 1
                        )
                    self._q_to_req_buf[:total_q].copy_(q_to_req_cpu, non_blocking=True)
                    self._q_to_klen_buf[:total_q].copy_(
                        q_to_klen_cpu, non_blocking=True
                    )
        else:
            all_pure_first_prefill = False

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
            prefix_scheduler_metadata = None

        attn_metadata = TritonAttentionMetadata(
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
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
            all_pure_first_prefill=all_pure_first_prefill,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            prefill_is_first_chunk=prefill_is_first_chunk,
            seq_lens_cpu=seq_lens_cpu,
            query_start_loc_cpu=qsl_cpu,
            q_to_req=(
                self._q_to_req_buf[:num_actual_tokens]
                if self._is_per_token_head and num_actual_tokens > 0
                else None
            ),
            q_to_klen=(
                self._q_to_klen_buf[:num_actual_tokens]
                if self._is_per_token_head and num_actual_tokens > 0
                else None
            ),
        )
        return attn_metadata


class TritonAttentionBackend(AttentionBackend):
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
        "int2_per_token_head",
        "int4_per_token_head",
        "int8_per_token_head",
        "fp8_per_token_head",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

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
        if kv_cache_uses_per_token_head_scales(cache_dtype_str):
            from vllm.utils.torch_utils import (
                STR_DTYPE_TO_TORCH_DTYPE,
                get_dtype_size,
            )

            cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype_str]
            scale_pad = get_dtype_size(torch.float32) // get_dtype_size(cache_dtype)
            data_head_size = get_kv_quant_mode(cache_dtype_str).packed_head_size(
                head_size
            )
            return (num_blocks, 2, block_size, num_kv_heads, data_head_size + scale_pad)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (1, 4, 0, 2, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout: {cache_layout}")
        return stride_order

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """TritonAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonAttentionImpl(AttentionImpl):
    # Per-token-head scale caches (float32 strided views over KV cache bytes).
    _k_scale_cache: torch.Tensor | None = None
    _v_scale_cache: torch.Tensor | None = None

    def _ensure_scale_caches(self, kv_cache: torch.Tensor) -> None:
        """Extract per-head scale views from the padded head dimension.

        The KV cache shape is ``(num_blocks, 2, block_size, nkv, hs+pad)``
        where ``pad = sizeof(float32) / sizeof(cache_dtype)``.  The last
        ``pad`` elements of each head hold one float32 scale.  We create
        strided float32 views over those bytes.

        Scale shape: ``(num_blocks, block_size, num_kv_heads)``
        """
        if self._k_scale_cache is not None:
            return
        from vllm.utils.torch_utils import get_dtype_size

        num_blocks, _, block_size, nkv, padded_hs = kv_cache.shape
        dtype_sz = kv_cache.element_size()
        scale_pad = get_dtype_size(torch.float32) // dtype_sz
        hs = padded_hs - scale_pad

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
            raw
        )

        kv_half_bytes = block_size * nkv * padded_hs * dtype_sz
        full_block_f32 = 2 * kv_half_bytes // 4
        slot_f32 = nkv * padded_hs * dtype_sz // 4
        head_f32 = padded_hs * dtype_sz // 4
        scale_off_f32 = hs * dtype_sz // 4

        self._k_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=scale_off_f32,
        )
        self._k_scale_cache.fill_(1.0)

        v_base_f32 = kv_half_bytes // 4
        self._v_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=v_base_f32 + scale_off_f32,
        )
        self._v_scale_cache.fill_(1.0)

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
        use_alibi_sqrt: bool = False,
        chunk_lookback: int = -1,
    ) -> None:
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

        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )
        self.use_alibi_sqrt = use_alibi_sqrt
        self.chunk_lookback = chunk_lookback
        self.supports_quant_query_input = current_platform.is_cuda()

        self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)
        self._is_per_token_head_quant = self._kv_quant_mode.is_per_token_head

        if self._is_per_token_head_quant:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
            self.max_num_kv_splits = (
                vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
            )

        # Enable tensor descriptors for Q/K/V load/store on platforms that
        # benefit from HW 2D block reads (Intel Xe2/Xe3).  The dead branch
        # is eliminated at Triton compile time, so other platforms see
        # zero cost when TD is off.
        #
        # ``VLLM_TRITON_ATTN_USE_TD`` is tri-state:
        #   - unset (None): auto-select (TD on for XPU, off elsewhere),
        #   - ``1``: force TD on regardless of platform,
        #   - ``0``: force TD off regardless of platform (useful for A/B).
        td_override = envs.VLLM_TRITON_ATTN_USE_TD
        if td_override is None:
            self.use_td = current_platform.is_xpu()
        else:
            self.use_td = td_override

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
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

        # Dedicated prefill kernel (with packed variant for INT4 / INT2).
        # Decode is gated separately below: only INT8 / FP8 use the
        # dedicated split-KV decode kernel — INT4 / INT2 fall through to
        # ``unified_attention`` / ``_attn_packed``, which gets tensor cores
        # via ``tl.dot`` whereas the split-KV decode kernel uses vector
        # mul-reduce.
        if (
            self._is_per_token_head_quant
            and self.alibi_slopes is None
            and not self.use_alibi_sqrt
            and self.sinks is None
            and not self.logits_soft_cap
            and attn_metadata.mm_prefix_range_tensor is None
            and output_scale is None
            and self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            num_dec = attn_metadata.num_decodes
            num_dec_tok = attn_metadata.num_decode_tokens
            pref_first_chunk = attn_metadata.prefill_is_first_chunk
            all_first_chunk = attn_metadata.all_pure_first_prefill

            if num_dec == 0 and all_first_chunk:
                context_attention_fwd(
                    q=query[:num_actual_tokens],
                    k=key[:num_actual_tokens],
                    v=value[:num_actual_tokens],
                    o=output[:num_actual_tokens],
                    b_start_loc=attn_metadata.query_start_loc,
                    b_seq_len=attn_metadata.seq_lens,
                    max_input_len=attn_metadata.max_query_len,
                    is_causal=True,
                    softmax_scale=self.scale,
                    sliding_window_q=self.sliding_window[0],
                    sliding_window_k=self.sliding_window[1],
                )
                return output

            if num_dec > 0 and num_dec_tok < num_actual_tokens and pref_first_chunk:
                self._ensure_scale_caches(kv_cache)
                key_cache, value_cache = kv_cache.unbind(1)
                if (
                    self._kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD
                    and key_cache.dtype == torch.uint8
                ):
                    key_cache = key_cache.view(self.fp8_dtype)
                    value_cache = value_cache.view(self.fp8_dtype)

                unified_attention(
                    q=query[:num_dec_tok],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_dec_tok],
                    cu_seqlens_q=attn_metadata.query_start_loc[: num_dec + 1],
                    max_seqlen_q=1,
                    seqused_k=attn_metadata.seq_lens[:num_dec],
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    alibi_slopes=None,
                    use_alibi_sqrt=False,
                    window_size=self.sliding_window,
                    block_table=attn_metadata.block_table[:num_dec],
                    softcap=0,
                    q_descale=None,
                    k_descale=None,
                    v_descale=None,
                    seq_threshold_3D=attn_metadata.seq_threshold_3D,
                    num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
                    softmax_segm_output=attn_metadata.softmax_segm_output,
                    softmax_segm_max=attn_metadata.softmax_segm_max,
                    softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
                    sinks=None,
                    output_scale=None,
                    mm_prefix_range=None,
                    kv_quant_mode=self._kv_quant_mode,
                    k_scale_cache=self._k_scale_cache,
                    v_scale_cache=self._v_scale_cache,
                )

                pref_qsl = attn_metadata.query_start_loc[num_dec:] - num_dec_tok
                pref_max_q = attn_metadata.max_query_len
                context_attention_fwd(
                    q=query[num_dec_tok:num_actual_tokens],
                    k=key[num_dec_tok:num_actual_tokens],
                    v=value[num_dec_tok:num_actual_tokens],
                    o=output[num_dec_tok:num_actual_tokens],
                    b_start_loc=pref_qsl,
                    b_seq_len=attn_metadata.seq_lens[num_dec:],
                    max_input_len=pref_max_q,
                    is_causal=True,
                    softmax_scale=self.scale,
                    sliding_window_q=self.sliding_window[0],
                    sliding_window_k=self.sliding_window[1],
                )
                return output

            # FP3: pure prefill with at least one continuation chunk.
            # Reads paged cache with inline per-token-head dequant via a
            # flash-attention-shaped kernel — avoids falling through to the
            # decode-shaped unified_attention which wastes K loads across
            # query tiles.
            if (
                num_dec == 0
                and num_actual_tokens > 0
                and self.sliding_window == (-1, -1)
            ):
                self._ensure_scale_caches(kv_cache)
                key_cache, value_cache = kv_cache.unbind(1)
                k_scale_cache = self._k_scale_cache
                v_scale_cache = self._v_scale_cache
                if (
                    self._kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD
                    and key_cache.dtype == torch.uint8
                ):
                    key_cache = key_cache.view(self.fp8_dtype)
                    value_cache = value_cache.view(self.fp8_dtype)
                # int8 cache on ROCm → route Q·Kᵀ through native int8
                # WMMA/MFMA (~2× bf16 throughput). fp8 cache keeps the
                # bf16 path (no fp8 MMA on RDNA3; MI300X fp8 path TBD).
                use_qk_int8_wmma = (
                    key_cache.dtype == torch.int8 and current_platform.is_rocm()
                )
                num_reqs_pref = attn_metadata.query_start_loc.shape[0] - 1
                triton_per_token_head_prefill(
                    query=query[:num_actual_tokens],
                    output=output[:num_actual_tokens],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    k_scale_cache=k_scale_cache,
                    v_scale_cache=v_scale_cache,
                    block_table=attn_metadata.block_table,
                    query_start_loc=attn_metadata.query_start_loc,
                    seq_lens=attn_metadata.seq_lens,
                    softmax_scale=self.scale,
                    num_reqs=num_reqs_pref,
                    max_query_len=attn_metadata.max_query_len,
                    use_qk_int8_wmma=use_qk_int8_wmma,
                    kv_quant_mode=self._kv_quant_mode,
                )
                return output

            # FP4: mixed decode + prefill where the prefill portion includes
            # at least one continuation chunk. Decode portion goes through
            # unified_attention (decode-tuned); prefill portion uses the
            # flash-attention prefill kernel.
            if (
                num_dec > 0
                and num_dec_tok < num_actual_tokens
                and self.sliding_window == (-1, -1)
            ):
                self._ensure_scale_caches(kv_cache)
                key_cache, value_cache = kv_cache.unbind(1)
                k_scale_cache = self._k_scale_cache
                v_scale_cache = self._v_scale_cache
                if (
                    self._kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD
                    and key_cache.dtype == torch.uint8
                ):
                    key_cache = key_cache.view(self.fp8_dtype)
                    value_cache = value_cache.view(self.fp8_dtype)

                unified_attention(
                    q=query[:num_dec_tok],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_dec_tok],
                    cu_seqlens_q=attn_metadata.query_start_loc[: num_dec + 1],
                    max_seqlen_q=1,
                    seqused_k=attn_metadata.seq_lens[:num_dec],
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    alibi_slopes=None,
                    use_alibi_sqrt=False,
                    window_size=self.sliding_window,
                    block_table=attn_metadata.block_table[:num_dec],
                    softcap=0,
                    q_descale=None,
                    k_descale=None,
                    v_descale=None,
                    seq_threshold_3D=attn_metadata.seq_threshold_3D,
                    num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
                    softmax_segm_output=attn_metadata.softmax_segm_output,
                    softmax_segm_max=attn_metadata.softmax_segm_max,
                    softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
                    sinks=None,
                    output_scale=None,
                    mm_prefix_range=None,
                    kv_quant_mode=self._kv_quant_mode,
                    k_scale_cache=k_scale_cache,
                    v_scale_cache=v_scale_cache,
                )

                pref_qsl = attn_metadata.query_start_loc[num_dec:] - num_dec_tok
                num_reqs_pref = attn_metadata.query_start_loc.shape[0] - 1 - num_dec
                use_qk_int8_wmma = (
                    key_cache.dtype == torch.int8 and current_platform.is_rocm()
                )
                triton_per_token_head_prefill(
                    query=query[num_dec_tok:num_actual_tokens],
                    output=output[num_dec_tok:num_actual_tokens],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    k_scale_cache=k_scale_cache,
                    v_scale_cache=v_scale_cache,
                    block_table=attn_metadata.block_table[num_dec:],
                    query_start_loc=pref_qsl,
                    seq_lens=attn_metadata.seq_lens[num_dec:],
                    softmax_scale=self.scale,
                    num_reqs=num_reqs_pref,
                    max_query_len=attn_metadata.max_query_len,
                    use_qk_int8_wmma=use_qk_int8_wmma,
                    kv_quant_mode=self._kv_quant_mode,
                )
                return output

        # Per-token-head: dedicated split-KV kernel for decode and small
        # continuation prefill (q_len ≤ threshold). Same trick as TQ: each
        # query gets its own causal K length via q_to_klen. For large
        # continuation (q_len > threshold), fall through to unified_attention.
        if self._is_per_token_head_quant:
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
            if (
                self._kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD
                and key_cache.dtype == torch.uint8
            ):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            q_descale = None
            k_descale = None
            v_descale = None

            # Dedicated split-KV kernel for decode + small continuation
            # prefill. Gated on max_query_len ≤ threshold; larger shapes
            # fall through to unified_attention (better BLOCK_Q sharing).
            if (
                attn_metadata.max_query_len <= _CONTINUATION_DECODE_THRESHOLD
                and attn_metadata.q_to_req is not None
                and attn_metadata.q_to_klen is not None
                and self.alibi_slopes is None
                and not self.use_alibi_sqrt
                and self.sinks is None
                and not self.logits_soft_cap
                and self.sliding_window == (-1, -1)
                and attn_metadata.mm_prefix_range_tensor is None
                and output_scale is None
            ):
                mid_o_buf = getattr(layer, "_pth_mid_o_buf", None)
                output_buf = getattr(layer, "_pth_output_buf", None)
                lse_buf = getattr(layer, "_pth_lse_buf", None)
                use_qk_int8_wmma = (
                    key_cache.dtype == torch.int8 and current_platform.is_rocm()
                )
                triton_per_token_head_attention(
                    query=query[:num_actual_tokens],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    k_scale_cache=k_scale_cache,
                    v_scale_cache=v_scale_cache,
                    block_table=attn_metadata.block_table,
                    q_to_req=attn_metadata.q_to_req,
                    q_to_klen=attn_metadata.q_to_klen,
                    scale=self.scale,
                    max_num_kv_splits=self.max_num_kv_splits,
                    output=output[:num_actual_tokens],
                    mid_o_buf=mid_o_buf,
                    output_buf=output_buf,
                    lse_buf=lse_buf,
                    buf_holder=layer,
                    use_qk_int8_wmma=use_qk_int8_wmma,
                    kv_quant_mode=self._kv_quant_mode,
                )
                return output
        # FP8 per-tensor / auto path (original flow).
        else:
            key_cache, value_cache = kv_cache.unbind(1)
            if (
                is_quantized_kv_cache(self.kv_cache_dtype)
                and key_cache.dtype != self.fp8_dtype
            ):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                key_cache.shape[2],
            )
            q_descale = (
                layer._q_scale
                if (
                    self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
                    and query.dtype == self.fp8_dtype
                )
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)
            k_scale_cache = None
            v_scale_cache = None

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=self.chunk_lookback,
            use_td=self.use_td,
        )

        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
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
        # Quantized KV cache is not supported for encoder attention.
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantized KV cache is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # Encoder attention is bidirectional
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
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
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        # Reshape the input keys and values and store them in the cache.
        if self._is_per_token_head_quant:
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
            if (
                self._kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD
                and key_cache.dtype == torch.uint8
            ):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            triton_reshape_and_cache_flash_per_token_head_quant(
                key,
                value,
                key_cache,
                value_cache,
                k_scale_cache,
                v_scale_cache,
                slot_mapping,
                kv_quant_mode=self._kv_quant_mode,
            )
            return
        # For decoder and cross-attention, use KV cache as before.
        key_cache, value_cache = kv_cache.unbind(1)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        triton_reshape_and_cache_flash(
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
        if self._is_per_token_head_quant:
            return False
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
        key_cache, value_cache = kv_cache.unbind(1)
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
