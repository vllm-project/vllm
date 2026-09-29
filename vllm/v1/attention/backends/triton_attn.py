# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

import contextlib
import os
from dataclasses import dataclass, replace
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
from vllm.utils.torch_utils import get_dtype_size, is_quantized_kv_cache
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

    causal: bool | torch.Tensor

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


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    # Step-dependent fields reference persistent input buffers directly.
    supports_draft_decode_metadata_update = True

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
        num_reqs = common_attn_metadata.num_reqs
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
            # The scheduler did not materialize the CPU copy of seq_lens; the
            # spec-decode verify step takes this branch. The per-query maps
            # would then keep whatever the allocator left in the persistent
            # buffers, and the metadata below hands those to the kernel either
            # way -- whose own guards turn the garbage into kv_len 0, so
            # attention silently returns zeros (corrupt text, drafts rejected).
            # Derive the maps from the GPU tensors instead: same values, no D2H
            # sync. searchsorted also covers cudagraph padding, whose queries
            # fall past the last request and get kv_len 0.
            if self._is_per_token_head and num_actual_tokens > 0:
                dev = query_start_loc.device
                idx = torch.arange(num_actual_tokens, dtype=torch.int32, device=dev)
                qsl = query_start_loc.to(torch.int32)
                q_lens = qsl[1:] - qsl[:-1]
                req_ids = torch.searchsorted(qsl[1:], idx, right=True).to(torch.int32)
                req_c = req_ids.clamp(max=num_reqs - 1).long()
                klen = (
                    seq_lens.to(torch.int32)[req_c]
                    - q_lens[req_c]
                    + (idx - qsl[:-1][req_c])
                    + 1
                )
                self._q_to_req_buf[:num_actual_tokens].copy_(req_ids)
                self._q_to_klen_buf[:num_actual_tokens].copy_(
                    torch.where(req_ids < num_reqs, klen, torch.zeros_like(klen))
                )

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
            causal=common_attn_metadata.causal,
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

        mm_ranges = common_attn_metadata.mm_req_doc_ranges
        if mm_ranges is not None:
            attn_metadata.mm_prefix_range = mm_ranges
            attn_metadata.mm_prefix_range_tensor = compute_mm_prefix_range_tensor(
                mm_ranges, num_reqs, seq_lens.device
            )

        rswa_prefix_lens = common_attn_metadata.rswa_prefix_lens
        if self.rswa_window is not None and rswa_prefix_lens is not None:
            assert self.persistent_rswa_prefix_lens is not None
            rswa_prefix_lens = rswa_prefix_lens.to(
                device=self.device, dtype=torch.int32, non_blocking=True
            )
            persistent_prefix_lens = self.persistent_rswa_prefix_lens[:num_reqs]
            persistent_prefix_lens.copy_(rswa_prefix_lens[:num_reqs])
            attn_metadata.rswa_prefix_lens = persistent_prefix_lens
            attn_metadata.rswa_window = self.rswa_window

        return attn_metadata

    def update_draft_decode_metadata(self, _metadata: TritonAttentionMetadata) -> None:
        pass


class TritonAttentionBackend(AttentionBackend):
    @classmethod
    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        """Per-token-head modes pack inline fp32 scales after each head's
        data, so the content is (data + one scale) per K/V side."""
        mode = spec.kv_quant_mode
        if spec.state_content_bytes is not None or not mode.is_per_token_head:
            return spec
        hs_k, hs_v = spec.head_size, spec.head_size_v
        if mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            hs_k, hs_v = hs_k // 2, hs_v // 2
        scale_bytes = get_dtype_size(torch.float32)
        content = (hs_k + hs_v) * get_dtype_size(spec.dtype) + 2 * scale_bytes
        return replace(spec, state_content_bytes=content)

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

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

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
    _rht_signs: torch.Tensor | None = None

    def _get_rht_signs(self, device: torch.device) -> torch.Tensor:
        """Cached RHT D₁ signs [head_size] for fused INT4 kernel."""
        if self._rht_signs is None:
            from vllm.v1.attention.ops.triton_quant_kv._hadamard import (
                _get_rht_signs,
            )

            self._rht_signs = _get_rht_signs(self.head_size, 0, device)
        return self._rht_signs

    def _ensure_scale_caches(self, kv_cache: torch.Tensor) -> None:
        """Extract per-head scale views from the padded content dimension.

        The KV cache is packed as logical shape
        ``(num_blocks, nkv, block_size, 2 * (hs + pad))`` where
        ``pad = sizeof(float32) / sizeof(cache_dtype)``.  The content dim holds
        ``[K(hs) | K_scale(pad) | V(hs) | V_scale(pad)]`` per (head, slot); the
        last ``pad`` elements of each half hold one float32 scale.  We create
        strided float32 views over those bytes.  ``kv_cache`` must be the
        packed logical tensor (call before any transpose), but may have HND or
        NHD physical strides.

        Scale shape: ``(num_blocks, block_size, num_kv_heads)``
        """
        if self._k_scale_cache is not None:
            return
        from vllm.utils.torch_utils import get_dtype_size

        num_blocks, nkv, block_size, content = kv_cache.shape
        dtype_sz = kv_cache.element_size()
        scale_pad = get_dtype_size(torch.float32) // dtype_sz  # e.g. 4
        padded_hs = content // 2
        hs = padded_hs - scale_pad

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
            raw
        )

        def to_f32_units(elements: int) -> int:
            nbytes = elements * dtype_sz
            assert nbytes % 4 == 0
            return nbytes // 4

        # Actual strides (in float32 units) from the tensor. The logical cache
        # may be physically NHD, so do not assume C-contiguous HND layout.
        strides = kv_cache.stride()
        block_f32 = to_f32_units(strides[0])
        head_f32 = to_f32_units(strides[1])
        slot_f32 = to_f32_units(strides[2])
        # Scale sits at byte offset hs within each (K, then V) content half.
        base_off_f32 = to_f32_units(kv_cache.storage_offset())
        k_scale_off_f32 = base_off_f32 + to_f32_units(hs)
        v_scale_off_f32 = base_off_f32 + to_f32_units(padded_hs + hs)

        # K scales (first content half)
        self._k_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(block_f32, slot_f32, head_f32),
            storage_offset=k_scale_off_f32,
        )
        self._k_scale_cache.fill_(1.0)

        # V scales (second content half)
        self._v_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(block_f32, slot_f32, head_f32),
            storage_offset=v_scale_off_f32,
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
        if current_platform.is_cuda():
            cap = current_platform.get_device_capability()
            cap_str = cap.as_version_str() if cap is not None else "unknown"
            dev = current_platform.get_device_name()
            if self.kv_cache_dtype.startswith("fp8") and not (
                current_platform.has_device_capability(89)
            ):
                suggested = (
                    "float16" if (cap is None or cap.to_int() < 80) else "bfloat16"
                )
                raise ValueError(
                    f"FP8 KV cache is not supported by the Triton attention backend "
                    f"on {dev} (compute capability {cap_str}); native FP8 (fp8e4nv) "
                    f"requires SM89+. Re-run with --kv-cache-dtype {suggested}."
                )
            if self.kv_cache_dtype == "bfloat16" and not (
                current_platform.has_device_capability(80)
            ):
                raise ValueError(
                    f"bfloat16 KV cache is not supported on {dev} (compute capability "
                    f"{cap_str}); bfloat16 requires SM80+. Re-run with "
                    f"--kv-cache-dtype float16."
                )
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
            self._max_cudagraph_capture_size = (
                vllm_config.compilation_config.max_cudagraph_capture_size or 4
            )

        # Pre-compute RDNA3 INT4 fast-path eligibility flags once.
        _is_rdna3_int4 = (
            self._kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD
            and head_size in (128, 256)
            and current_platform.is_rocm()
            and hasattr(torch.ops, "_C")
        )
        # HIP decode for HS=128 and HS=256 (symmetric format: scalar shuffle
        # without per-token zp correction is now competitive at long ctx).
        _decode_hs_ok = head_size in (128, 256)
        _no_special_features = (
            alibi_slopes is None
            and not use_alibi_sqrt
            and sinks is None
            and not self.logits_soft_cap
            and sliding_window is None
            and kv_sharing_target_layer_name is None
        )
        self._rdna3_int4_decode_ready = (
            _is_rdna3_int4
            and _decode_hs_ok
            and _no_special_features
            and hasattr(torch.ops._C, "pth_decode_int4_rdna3")
        )
        # Prefill HIP kernel for HS=128 and HS=256.  HS=256 uses
        # pragma-unroll-1 to force register recycling (16 WMMA frags).
        self._rdna3_int4_prefill_ready = (
            _is_rdna3_int4
            and _no_special_features
            and hasattr(torch.ops._C, "paged_prefill_attn_rdna3_int4")
        )
        # HIP scalar v3 decode for INT8 per-token-head (HS=256 only).
        _is_rdna3_int8 = (
            self._kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD
            and head_size == 256
            and current_platform.is_rocm()
            and hasattr(torch.ops, "_C")
        )
        self._rdna3_int8_decode_ready = (
            _is_rdna3_int8
            and _no_special_features
            and hasattr(torch.ops._C, "pth_decode_int8_rdna3")
        )
        # Pre-compute INT4 softmax scale (avoids division every forward call).
        self._int4_scale = self.scale / head_size if _is_rdna3_int4 else 0.0

        # Enable tensor descriptors for Q/K/V load/store on platforms that
        # benefit from HW 2D block reads (Intel XPU).  The dead branch
        # is eliminated at Triton compile time, so other platforms see
        # zero cost when TD is off.
        #
        # ``VLLM_TRITON_USE_TD`` is tri-state:
        #   - unset (None): auto-select (TD on for XPU, off elsewhere),
        #   - ``1``: force TD on regardless of platform,
        #   - ``0``: force TD off regardless of platform (useful for A/B).
        td_override = envs.VLLM_TRITON_USE_TD
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

        # ---- RDNA3 INT8 fast-path: short-circuit for decode ----
        # Single-wave scalar v3 kernel, zero __syncthreads.
        if (
            self._rdna3_int8_decode_ready
            and self._k_scale_cache is not None
            and attn_metadata.max_query_len <= _CONTINUATION_DECODE_THRESHOLD
            and attn_metadata.q_to_req is not None
        ):
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            # FIX (RDNA3 int8 mixed-batch page fault): the persistent buffer is
            # allocated ONCE at the cudagraph capture size and NEVER reassigned.
            # The decode CUDA graph bakes in this pointer at capture; the old code
            # reallocated + reassigned _pth_mid_o_buf whenever query.size(0) grew
            # past it (continuation-decode batches: max_query_len<=128 so a
            # prefix-cache resume puts decodes + per-req chunks up to 128 tokens
            # here, easily >8). That freed the capture-time buffer, and a later
            # decode-graph replay then wrote split-KV partials into the freed
            # (reused, read-only) pages -> the marching TCP-client WRITE page fault
            # in dmesg (RW:1, PERMISSION_FAULTS, 66KB = mid_o head-stride). Larger
            # (eager-only; captured sizes are <= capture size) batches use a
            # transient buffer instead of clobbering the persistent one.
            mid_o_buf = getattr(layer, "_pth_mid_o_buf", None)
            if mid_o_buf is None:
                mid_o_buf = torch.zeros(
                    self._max_cudagraph_capture_size,
                    self.num_heads,
                    self.max_num_kv_splits,
                    self.head_size + 2,
                    dtype=torch.float32,
                    device=query.device,
                )
                layer._pth_mid_o_buf = mid_o_buf
            num_kv_splits = self.max_num_kv_splits
            if mid_o_buf.shape[0] < query.size(0):
                mid_o_buf, num_kv_splits = self._transient_mid_o(
                    query.size(0), query.device
                )
            # FIX (RDNA3 int8 decode use-after-free): under async scheduling the
            # caching allocator can free + REUSE a transient input's memory
            # before this async kernel reads it. Symptoms (both confirmed via
            # rocgdb / soak): a freed page -> GPU page fault ("Memory access
            # fault, page not present"); a reused block -> the kernel reads
            # another tensor's data as metadata -> garbage kv_len -> infinite KV
            # loop -> GPU hang. record_stream tells the allocator these tensors
            # are in use on the current stream, so their blocks are not freed/
            # reused until the kernel completes. No copy, no extra memory.
            # Confirmed by control: same loop with data_ptr() (no record_stream)
            # hangs in ~20s; with record_stream it soaks 22min / 2000+ reqs.
            _s = torch.cuda.current_stream()
            for _t in (
                query,
                output,
                attn_metadata.block_table,
                attn_metadata.q_to_req,
                attn_metadata.q_to_klen,
                self._k_scale_cache,
                self._v_scale_cache,
                mid_o_buf,
            ):
                with contextlib.suppress(Exception):
                    _t.record_stream(_s)
            torch.ops._C.pth_decode_int8_rdna3(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._v_scale_cache,
                attn_metadata.block_table,
                attn_metadata.q_to_req,
                attn_metadata.q_to_klen,
                mid_o_buf,
                self.scale,
                num_kv_splits,
            )
            return output

        # ---- RDNA3 INT4 fast-path: short-circuit for decode ----
        # Bypasses ~20 Python ops of gate evaluation per layer.
        # Must check ALL conditions that the normal path checks, but
        # with minimal Python overhead (cached attributes, no slicing).
        if (
            self._rdna3_int4_decode_ready
            and self._rht_signs is not None
            and self._k_scale_cache is not None
            and attn_metadata.max_query_len <= _CONTINUATION_DECODE_THRESHOLD
            and attn_metadata.q_to_req is not None
        ):
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            # Capture-stable buffer (see the int8 fast-path fix above): allocate
            # once at the cudagraph capture size and NEVER reassign -- the decode
            # graph bakes in this pointer; a larger eager batch uses a transient.
            mid_o_buf = getattr(layer, "_pth_mid_o_buf", None)
            if mid_o_buf is None:
                mid_o_buf = torch.zeros(
                    self._max_cudagraph_capture_size,
                    self.num_heads,
                    self.max_num_kv_splits,
                    self.head_size + 2,
                    dtype=torch.float32,
                    device=query.device,
                )
                layer._pth_mid_o_buf = mid_o_buf
            num_kv_splits = self.max_num_kv_splits
            if mid_o_buf.shape[0] < query.size(0):
                mid_o_buf, num_kv_splits = self._transient_mid_o(
                    query.size(0), query.device
                )
            q_slice = query[:num_actual_tokens]
            o_slice = output[:num_actual_tokens]
            # HS=256 v2 kernel: Q pre-rotated externally, output post-rotated.
            # q_rot is likewise capture-stable: allocate once at capture size,
            # never reassign (transient for larger eager batches).
            if self.head_size > 128:
                q_rot = getattr(layer, "_pth_q_rot_buf", None)
                if q_rot is None:
                    q_rot = torch.empty(
                        self._max_cudagraph_capture_size,
                        self.num_heads,
                        self.head_size,
                        dtype=query.dtype,
                        device=query.device,
                    )
                    layer._pth_q_rot_buf = q_rot
                if q_rot.shape[0] < query.size(0):
                    q_rot = torch.empty(
                        query.size(0),
                        self.num_heads,
                        self.head_size,
                        dtype=query.dtype,
                        device=query.device,
                    )
                q_rot[:num_actual_tokens].copy_(q_slice)
                q_slice = q_rot[:num_actual_tokens]
                torch.ops._C.rht_rotate_inplace_rdna3(
                    q_slice, self._rht_signs, False, 1.0
                )
            torch.ops._C.pth_decode_int4_rdna3(
                o_slice,
                q_slice,
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._v_scale_cache,
                self._rht_signs,
                attn_metadata.block_table,
                attn_metadata.q_to_req,
                attn_metadata.q_to_klen,
                mid_o_buf,
                self._int4_scale,
                num_kv_splits,
            )
            if self.head_size > 128:
                torch.ops._C.rht_rotate_inplace_rdna3(
                    o_slice, self._rht_signs, True, 1.0 / self.head_size
                )
            return output

        # ---- RDNA3 INT4 fast-path: short-circuit for continuation prefill ----
        # Pure continuation (all from cache, no first-chunk) bypasses ~20
        # Python ops of gate evaluation per layer.
        if (
            self._rdna3_int4_prefill_ready
            and attn_metadata.num_decodes == 0
            and not attn_metadata.all_pure_first_prefill
        ):
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            rht_signs = self._get_rht_signs(query.device)
            torch.ops._C.paged_prefill_attn_rdna3_int4(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._v_scale_cache,
                rht_signs,
                attn_metadata.block_table,
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                attn_metadata.max_query_len,
                self._int4_scale,
                True,
            )
            return output

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

        # Dedicated prefill kernel (with packed variant for INT4).
        # Decode is gated separately below: only INT8 / FP8 use the
        # dedicated split-KV decode kernel — INT4 falls through to
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
                key_cache, value_cache = self._pth_key_value_caches(kv_cache)

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
                key_cache, value_cache = self._pth_key_value_caches(kv_cache)
                k_scale_cache = self._k_scale_cache
                v_scale_cache = self._v_scale_cache
                # int8 cache on ROCm → route Q·Kᵀ through native int8
                # WMMA/MFMA (~2× bf16 throughput). fp8 cache keeps the
                # bf16 path (no fp8 MMA on RDNA3; MI300X fp8 path TBD).
                use_qk_int8_wmma = (
                    key_cache.dtype == torch.int8 and current_platform.is_rocm()
                )

                # RDNA3 HIP kernel for INT8 per-token-head prefill.
                # Gate: int8 cache + RDNA3 + has the compiled op.
                # Diagnostic kill-switch: VLLM_RDNA3_INT8_PREFILL_HIP=0 forces
                # the Triton prefill path below instead of this HIP kernel,
                # keeping the int8 cache. Use it to A/B-isolate whether a hang
                # or fault originates in the HIP kernel vs shared metadata.
                _head_size = query.shape[2]
                _int8_prefill_hip = (
                    os.environ.get("VLLM_RDNA3_INT8_PREFILL_HIP", "1") != "0"
                )
                if (
                    _int8_prefill_hip
                    and use_qk_int8_wmma
                    and _head_size in (64, 128, 256)
                    and hasattr(torch.ops, "_C")
                    and hasattr(torch.ops._C, "paged_prefill_attn_rdna3_int8")
                ):
                    torch.ops._C.paged_prefill_attn_rdna3_int8(
                        output[:num_actual_tokens],
                        query[:num_actual_tokens],
                        key[:num_actual_tokens],
                        value[:num_actual_tokens],
                        key_cache,
                        value_cache,
                        k_scale_cache,
                        v_scale_cache,
                        attn_metadata.block_table,
                        attn_metadata.query_start_loc,
                        attn_metadata.seq_lens,
                        attn_metadata.max_query_len,
                        self.scale,
                        True,
                    )
                    return output

                # RDNA3 HIP kernel for INT4 per-token-head prefill.
                # RDNA3 HIP INT4 kernel: cache-only, fused RHT, bf16 WMMA.
                # Zero Python-side RHT overhead — identical compute to INT8.
                if self._rdna3_int4_prefill_ready:
                    torch.ops._C.paged_prefill_attn_rdna3_int4(
                        output[:num_actual_tokens],
                        query[:num_actual_tokens],
                        key_cache,
                        value_cache,
                        k_scale_cache,
                        v_scale_cache,
                        self._get_rht_signs(query.device),
                        attn_metadata.block_table,
                        attn_metadata.query_start_loc,
                        attn_metadata.seq_lens,
                        attn_metadata.max_query_len,
                        self._int4_scale,
                        True,
                    )
                    return output

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
                key_cache, value_cache = self._pth_key_value_caches(kv_cache)
                k_scale_cache = self._k_scale_cache
                v_scale_cache = self._v_scale_cache

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
                # INT4 prefill: use HIP kernel directly.
                if self._rdna3_int4_prefill_ready:
                    torch.ops._C.paged_prefill_attn_rdna3_int4(
                        output[num_dec_tok:num_actual_tokens],
                        query[num_dec_tok:num_actual_tokens],
                        key_cache,
                        value_cache,
                        k_scale_cache,
                        v_scale_cache,
                        self._get_rht_signs(query.device),
                        attn_metadata.block_table[num_dec:],
                        pref_qsl,
                        attn_metadata.seq_lens[num_dec:],
                        attn_metadata.max_query_len,
                        self._int4_scale,
                        True,
                    )
                else:
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
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
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
                # RDNA3 HIP decode kernel for INT4 — eliminates Triton
                # dispatch overhead (~40 µs/call) with fused RHT.
                if self._rdna3_int4_decode_ready:
                    # Capture-stable buffers (see the int8 fast-path fix): allocate
                    # once at the cudagraph capture size, NEVER reassign the
                    # persistent one (the decode graph bakes in its pointer); a
                    # larger eager batch uses a transient buffer.
                    mid_o_buf = getattr(layer, "_pth_mid_o_buf", None)
                    if mid_o_buf is None:
                        mid_o_buf = torch.zeros(
                            self._max_cudagraph_capture_size,
                            self.num_heads,
                            self.max_num_kv_splits,
                            self.head_size + 2,
                            dtype=torch.float32,
                            device=query.device,
                        )
                        layer._pth_mid_o_buf = mid_o_buf
                    num_kv_splits = self.max_num_kv_splits
                    if mid_o_buf.shape[0] < query.size(0):
                        mid_o_buf, num_kv_splits = self._transient_mid_o(
                            query.size(0), query.device
                        )
                    rht_signs = self._get_rht_signs(query.device)
                    q_slice = query[:num_actual_tokens]
                    o_slice = output[:num_actual_tokens]
                    if self.head_size > 128:
                        q_rot = getattr(layer, "_pth_q_rot_buf", None)
                        if q_rot is None:
                            q_rot = torch.empty(
                                self._max_cudagraph_capture_size,
                                self.num_heads,
                                self.head_size,
                                dtype=query.dtype,
                                device=query.device,
                            )
                            layer._pth_q_rot_buf = q_rot
                        if q_rot.shape[0] < query.size(0):
                            q_rot = torch.empty(
                                query.size(0),
                                self.num_heads,
                                self.head_size,
                                dtype=query.dtype,
                                device=query.device,
                            )
                        q_rot[:num_actual_tokens].copy_(q_slice)
                        q_slice = q_rot[:num_actual_tokens]
                        torch.ops._C.rht_rotate_inplace_rdna3(
                            q_slice, rht_signs, False, 1.0
                        )
                    torch.ops._C.pth_decode_int4_rdna3(
                        o_slice,
                        q_slice,
                        key_cache,
                        value_cache,
                        k_scale_cache,
                        v_scale_cache,
                        rht_signs,
                        attn_metadata.block_table,
                        attn_metadata.q_to_req,
                        attn_metadata.q_to_klen,
                        mid_o_buf,
                        self._int4_scale,
                        num_kv_splits,
                    )
                    if self.head_size > 128:
                        torch.ops._C.rht_rotate_inplace_rdna3(
                            o_slice, rht_signs, True, 1.0 / self.head_size
                        )
                    return output

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
                    block_kv=32
                    if self._kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD
                    else 16,
                    output=output[:num_actual_tokens],
                    mid_o_buf=mid_o_buf,
                    output_buf=output_buf,
                    lse_buf=lse_buf,
                    buf_holder=layer,
                    use_qk_int8_wmma=use_qk_int8_wmma,
                    kv_quant_mode=self._kv_quant_mode,
                )
                return output

            # RDNA3 INT8: mixed batch with a large continuation chunk
            # (max_query_len > threshold) otherwise falls through to
            # unified_attention, which is ~11x slower than fp16 on gfx1100.
            # Split it into the fast HIP kernels — decode tokens ->
            # pth_decode_int8_rdna3 (split-KV), prefill chunk ->
            # paged_prefill_attn_rdna3_int8 (flash, shared K loads). Validated
            # bit-identical to unified_attention (~7.5x faster on the call).
            # VLLM_RDNA3_INT8_MIXED_HIP=0 disables it (back to unified).
            num_dec = attn_metadata.num_decodes
            num_dec_tok = attn_metadata.num_decode_tokens
            if (
                os.environ.get("VLLM_RDNA3_INT8_MIXED_HIP", "1") != "0"
                and key_cache.dtype == torch.int8
                and current_platform.is_rocm()
                and self.head_size in (64, 128, 256)
                and num_dec_tok < num_actual_tokens
                and attn_metadata.q_to_req is not None
                and attn_metadata.q_to_klen is not None
                and key is not None
                and value is not None
                and hasattr(torch.ops._C, "pth_decode_int8_rdna3")
                and hasattr(torch.ops._C, "paged_prefill_attn_rdna3_int8")
            ):
                if num_dec_tok > 0:
                    # FIX (RDNA3 int8 mixed-batch page fault): use a DEDICATED
                    # buffer here, never layer._pth_mid_o_buf. That attr is the
                    # pure-decode buffer whose pointer the decode CUDA graph bakes
                    # in at capture. A mixed batch has query.size(0) = decode
                    # tokens + a large prefill chunk, so the old shared code
                    # reallocated + REASSIGNED _pth_mid_o_buf to a big tensor,
                    # freeing the capture-time buffer. On a later decode-graph
                    # replay the kernel then wrote split-KV partials into that
                    # freed (reused, read-only) region -> the marching TCP-client
                    # WRITE page fault seen in dmesg (RW:1, PERMISSION_FAULTS,
                    # 66KB = mid_o head-stride). A separate buffer keeps the
                    # decode graph's pointer stable. Size by num_dec_tok (the
                    # actual decode rows the kernel writes), not query.size(0).
                    mid_o_buf = getattr(layer, "_pth_mid_o_buf_mixed", None)
                    if mid_o_buf is None or mid_o_buf.shape[0] < num_dec_tok:
                        mid_o_buf = torch.zeros(
                            max(num_dec_tok, self._max_cudagraph_capture_size),
                            self.num_heads,
                            self.max_num_kv_splits,
                            self.head_size + 2,
                            dtype=torch.float32,
                            device=query.device,
                        )
                        layer._pth_mid_o_buf_mixed = mid_o_buf
                    torch.ops._C.pth_decode_int8_rdna3(
                        output[:num_dec_tok],
                        query[:num_dec_tok],
                        key_cache,
                        value_cache,
                        k_scale_cache,
                        v_scale_cache,
                        attn_metadata.block_table[:num_dec],
                        attn_metadata.q_to_req[:num_dec_tok],
                        attn_metadata.q_to_klen[:num_dec_tok],
                        mid_o_buf,
                        self.scale,
                        self.max_num_kv_splits,
                    )
                # Prefill continuation chunk: re-base cu_seqlens to 0.
                pref_qsl = attn_metadata.query_start_loc[num_dec:] - num_dec_tok
                torch.ops._C.paged_prefill_attn_rdna3_int8(
                    output[num_dec_tok:num_actual_tokens],
                    query[num_dec_tok:num_actual_tokens],
                    key[num_dec_tok:num_actual_tokens],
                    value[num_dec_tok:num_actual_tokens],
                    key_cache,
                    value_cache,
                    k_scale_cache,
                    v_scale_cache,
                    attn_metadata.block_table[num_dec:],
                    pref_qsl,
                    attn_metadata.seq_lens[num_dec:],
                    attn_metadata.max_query_len,
                    self.scale,
                    True,
                )
                return output
        # FP8 per-tensor / INT8 per-tensor / auto path (original flow).
        else:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size, dim=-1
            )
            if (
                is_quantized_kv_cache(self.kv_cache_dtype)
                and key_cache.dtype != self.fp8_dtype
            ):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            if self.kv_cache_dtype.startswith("fp8"):
                assert layer._q_scale_float == 1.0, (
                    "A non 1.0 q_scale is not currently supported."
                )
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
            causal=attn_metadata.causal,
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
            rswa_prefix_lens=attn_metadata.rswa_prefix_lens,
            rswa_window=attn_metadata.rswa_window,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=self.chunk_lookback,
            use_td=self.use_td,
            mm_prefix_clamp_sliding_window=getattr(
                layer, "mm_prefix_clamp_sliding_window", False
            ),
        )

        return output

    def _pth_key_value_caches(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token-head K/V cache views (ensures scale caches; FP8 retyped)."""
        self._ensure_scale_caches(kv_cache)
        padded_hs = kv_cache.shape[-1] // 2
        key_cache, value_cache = kv_cache.transpose(1, 2).split(padded_hs, dim=-1)
        if self._kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        return key_cache, value_cache

    def _transient_mid_o(
        self, num_q: int, device: torch.device
    ) -> tuple[torch.Tensor, int]:
        """Bounded stand-in for the capture-stable ``_pth_mid_o_buf``.

        The persistent buffer is sized for the cudagraph capture size, so an
        eager batch above it needs its own. Sizing that one with
        ``max_num_kv_splits`` makes it grow without bound with the batch: a
        continuation-decode step admits ``_CONTINUATION_DECODE_THRESHOLD``
        tokens per request, and a prefix-cache resume of ``max_num_seqs``
        requests reaches hundreds of query tokens. At head_size=256 with 256
        splits that is 2 MiB per token, and the allocation OOMs the worker
        mid-forward -- which strands its peers in the next collective, so the
        engine dies on an all-gather timeout far from the real fault.

        Split-K only buys parallelism when there are few query tokens; a batch
        this size already saturates the GPU. So trade splits for batch and keep
        the transient buffer no larger than the persistent one.

        Returns:
            The buffer and the split count it was sized for. The kernels take
            the count as an argument and read the layout from the tensor's
            strides, so the two must be passed together.
        """
        budget = self._max_cudagraph_capture_size * self.max_num_kv_splits
        num_kv_splits = max(1, min(self.max_num_kv_splits, budget // num_q))
        mid_o_buf = torch.zeros(
            num_q,
            self.num_heads,
            num_kv_splits,
            self.head_size + 2,
            dtype=torch.float32,
            device=device,
        )
        return mid_o_buf, num_kv_splits

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
            sinks=self.sinks,
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
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
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
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        if self.kv_cache_dtype.startswith("fp8"):
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
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
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
