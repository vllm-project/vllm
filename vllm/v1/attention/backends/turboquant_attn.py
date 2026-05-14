# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for vLLM.

Prefill: Standard scaled dot-product attention on uncompressed K/V,
         then quantize K and store K+V into combined cache slot.
Decode:  Compute TQ attention scores from compressed cache,
         unpack FP16 values, softmax + weighted sum.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)
  where slot_size = key_packed_size + value_fp16_size

Per-head per-position slot layout:
  [key_packed (kps bytes) | value_fp16 (D*2 bytes)]
  For turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612
"""

import functools
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
_CONTINUATION_DECODE_THRESHOLD = 128


def _get_turboquant_decode_workspace_shapes(
    *,
    batch_size: int,
    num_heads: int,
    head_size: int,
    max_num_kv_splits: int,
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    return (
        ((batch_size, num_heads, max_num_kv_splits, head_size + 1), torch.float32),
        ((batch_size, num_heads, head_size), torch.float32),
        ((batch_size, num_heads), torch.float32),
    )


def _get_turboquant_dequant_workspace_cache_len(
    *,
    vllm_config: Any,
    sliding_window: int | None,
) -> int:
    max_model_len = vllm_config.model_config.max_model_len
    dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
    parallel_size = dcp_world_size * pcp_world_size
    if parallel_size > 1:
        max_model_len = (max_model_len + parallel_size - 1) // parallel_size
    if sliding_window is not None:
        return min(max_model_len, sliding_window)
    return max_model_len


def _get_turboquant_dequant_workspace_shapes(
    *,
    cache_len: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    alloc_len = math.ceil(cache_len / block_size) * block_size
    buf_shape = (1, num_kv_heads, alloc_len, head_size)
    return (
        (buf_shape, torch.float16),
        (buf_shape, torch.float16),
    )


def reserve_turboquant_decode_workspace(
    *,
    vllm_config: Any,
    num_heads: int,
    head_size: int,
) -> bool:
    """Pre-grow WorkspaceManager for TurboQuant decode scratch buffers."""
    if not is_workspace_manager_initialized():
        return False

    batch_size = max(
        vllm_config.scheduler_config.max_num_seqs,
        _CONTINUATION_DECODE_THRESHOLD,
    )
    max_num_kv_splits = vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
    current_workspace_manager().get_simultaneous(
        *_get_turboquant_decode_workspace_shapes(
            batch_size=batch_size,
            num_heads=num_heads,
            head_size=head_size,
            max_num_kv_splits=max_num_kv_splits,
        )
    )
    return True


def reserve_turboquant_dequant_workspace(
    *,
    vllm_config: Any,
    num_kv_heads: int,
    head_size: int,
    sliding_window: int | None,
) -> bool:
    """Pre-grow WorkspaceManager for cached K/V dequant scratch buffers."""
    if not is_workspace_manager_initialized():
        return False

    cache_len = _get_turboquant_dequant_workspace_cache_len(
        vllm_config=vllm_config,
        sliding_window=sliding_window,
    )
    current_workspace_manager().get_simultaneous(
        *_get_turboquant_dequant_workspace_shapes(
            cache_len=cache_len,
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
    )
    return True


def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction), cached per (d, device).

    Precomputed D×D matrix enables matmul-based WHT — single cuBLAS GEMM
    instead of log2(D) butterfly kernel launches. 64KB for D=128.
    """
    # Normalize device string so "cuda" and "cuda:0" hit the same cache entry.
    return _build_hadamard_cached(d, str(torch.device(device_str)))


@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple[int, ...]:
        """Combined K+V cache shape — no leading 2 dimension.

        Standard attention backends use (2, num_blocks, block_size, num_kv_heads,
        head_dim) with a leading 2 to separate K and V. TurboQuant packs K+V
        into a single interleaved slot per head per position, so the cache is:

            (num_blocks, block_size, num_kv_heads, slot_size_aligned)

        Each slot = [key_packed | value_packed | padding].
        This is safe because TQ has its own get_kv_cache_shape override and
        never shares cache tensors with other backends. Layers that fall back
        to native dtype via kv_cache_dtype_skip_layers get their own
        standard-shaped cache allocation.

        head_size is the model's real head_dim. slot_size_aligned is computed
        from the TQ config to ensure correct cache allocation for all head dims.
        """
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if kv_cache_dtype == "turboquant_k8v4" and head_size > 256:
            return "turboquant_k8v4 requires FlashAttention-compatible head_size <= 256"
        return None

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # head_size from spec is effective_head_size (padded_slot//2),
        # not the model's actual head_dim. Accept any positive value.
        return head_size > 0


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention."""

    seq_lens: torch.Tensor  # (num_reqs,) — total context length per request
    slot_mapping: torch.Tensor  # (num_tokens,) — cache slot for each token
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor  # (num_reqs + 1,) — cu_seqlens for queries
    num_actual_tokens: int = 0  # actual tokens (excluding padding)
    max_query_len: int = 0  # longest query in batch
    max_seq_len: int = 0  # longest context in batch
    is_prefill: bool = False
    num_decodes: int = 0  # number of decode requests (first in batch)
    num_decode_tokens: int = 0  # tokens from decode requests
    # CPU-resident copies used by the prefill path for per-request iteration
    # without per-step D2H syncs.
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TurboQuantMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # Set seq_lens to 1 so CUDA graph capture is fast
        # (real seq_lens are filled at replay time).
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """Build TurboQuantMetadata from common attention metadata."""
        cam = common_attn_metadata

        # With reorder_batch_threshold=1, the model runner guarantees
        # decodes come first in the batch. split_decodes_and_prefills
        # finds the boundary (operates on CPU tensors — no GPU sync).
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )

        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
        )


class TurboQuantAttentionImpl(AttentionImpl["TurboQuantMetadata"]):
    """TurboQuant attention implementation.

    Vectorized PyTorch: batch quantize/store, vectorized bit-unpack
    decode with einsum scores and value gather.
    """

    supports_quant_query_input: bool = False

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
        self.sliding_window = sliding_window
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)

        # Pre-compute kernel constants from config (avoid repeated arithmetic)
        cfg = self.tq_config
        self._mse_bytes = (
            math.ceil(head_size * cfg.key_mse_bits / 8)
            if not cfg.key_fp8
            else head_size
        )
        self._val_data_bytes = math.ceil(head_size * cfg.effective_value_quant_bits / 8)
        self._n_centroids = cfg.n_centroids if not cfg.key_fp8 else 1

        # Detect flash-attn version (FA2/3/4) for prefill paths.
        self.fa_version = get_flash_attn_version(head_size=head_size)
        # vllm_flash_attn rejects head dimensions above 256 at runtime.
        # Gemma4 global-attention layers use global_head_dim=512, so TQ must
        # fall back to SDPA for those prefill/continuation paths.
        self._can_use_flash_attn = _HAS_FLASH_ATTN and head_size <= 256

        # Fixed NUM_KV_SPLITS (grid dims must be constant for cudagraph,
        # and benchmarks show no regression vs dynamic in eager mode).
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        reserve_turboquant_decode_workspace(
            vllm_config=vllm_config,
            num_heads=self.num_heads,
            head_size=self.head_size,
        )
        reserve_turboquant_dequant_workspace(
            vllm_config=vllm_config,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            sliding_window=self.sliding_window,
        )

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        # fa_utils.get_flash_attn_version() returns None on backends that
        # should not pass an explicit fa_version kwarg.
        if self.fa_version is None:
            return flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            fa_version=self.fa_version,
        )

    def _needs_sliding_window_mask(self, seq_len: int) -> bool:
        return self.sliding_window is not None and seq_len > self.sliding_window

    def _can_use_flash_prefill(
        self,
        seq_len: int,
        mm_prefix_ranges: torch.Tensor | None,
    ) -> bool:
        return (
            self._can_use_flash_attn
            and not self._needs_sliding_window_mask(seq_len)
            and mm_prefix_ranges is None
        )

    def _get_arange_cache(
        self,
        max_value: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        arange_cache: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if (
            arange_cache is None
            or arange_cache.shape[0] <= max_value
            or arange_cache.device != device
            or arange_cache.dtype != dtype
        ):
            arange_cache = torch.arange(
                0,
                max_value + 1,
                device=device,
                dtype=dtype,
            )
            self._arange_cache = arange_cache
        return arange_cache

    def _get_mm_prefix_ranges(
        self,
        attn_metadata: TurboQuantMetadata,
        req_idx: int,
    ) -> torch.Tensor | None:
        mm_prefix_range_tensor = getattr(attn_metadata, "mm_prefix_range_tensor", None)
        if mm_prefix_range_tensor is None:
            return None
        return mm_prefix_range_tensor[req_idx]

    def _sdpa_with_causal_and_sliding_mask(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_start_pos: int,
        mm_prefix_ranges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run SDPA with causal/sliding mask and optional mm-prefix ranges."""
        q_len = query.shape[0]
        kv_len = key.shape[0]
        device = query.device
        q_t = query.transpose(0, 1).unsqueeze(0)
        k_t = key.transpose(0, 1).unsqueeze(0)
        v_t = value.transpose(0, 1).unsqueeze(0)

        q_pos = torch.arange(q_len, device=device).unsqueeze(1) + query_start_pos
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
        mask = k_pos <= q_pos
        if self.sliding_window is not None:
            mask = mask & ((q_pos - k_pos) < self.sliding_window)
        if mm_prefix_ranges is not None:
            starts = mm_prefix_ranges[:, 0]
            ends = mm_prefix_ranges[:, 1]
            valid = starts < ends
            q_in_range = (
                (q_pos[..., None] >= starts) & (q_pos[..., None] <= ends) & valid
            )
            k_in_range = (
                (k_pos[..., None] >= starts) & (k_pos[..., None] <= ends) & valid
            )
            mask = mask | (q_in_range & k_in_range).any(dim=-1)

        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=mask,
            scale=self.scale,
            enable_gqa=(key.shape[1] < query.shape[1]),
        )
        return out[0].transpose(0, 1)

    def _sdpa_causal_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q_t = query.transpose(0, 1).unsqueeze(0)
        k_t = key.transpose(0, 1).unsqueeze(0)
        v_t = value.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            is_causal=True,
            scale=self.scale,
            enable_gqa=(key.shape[1] < query.shape[1]),
        )
        return out[0].transpose(0, 1)

    def _decode_prefill_from_cache(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        *,
        query_start_pos: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        mm_prefix_ranges: torch.Tensor | None = None,
        seq_lens_dtype: torch.dtype = torch.int32,
        layer: Any = None,
    ) -> torch.Tensor:
        q_len, Hq, D = query.shape
        out = torch.empty_like(query)
        arange_cache = self._get_arange_cache(
            query_start_pos + q_len,
            device=query.device,
            dtype=seq_lens_dtype,
        )

        for chunk_start in range(0, q_len, _CONTINUATION_DECODE_THRESHOLD):
            chunk_end = min(chunk_start + _CONTINUATION_DECODE_THRESHOLD, q_len)
            chunk = query[chunk_start:chunk_end]
            chunk_len = chunk_end - chunk_start
            synth_seq_lens = arange_cache[
                query_start_pos + chunk_start + 1 : query_start_pos + chunk_end + 1
            ]
            synth_block_table = block_table.expand(chunk_len, -1)
            mm_prefix_range = None
            if mm_prefix_ranges is not None:
                mm_prefix_range = (
                    mm_prefix_ranges.unsqueeze(0).expand(chunk_len, -1, -1).contiguous()
                )

            mid_o_buf = output_buf = lse_buf = None
            if is_workspace_manager_initialized():
                mid_o_buf, output_buf, lse_buf = (
                    current_workspace_manager().get_simultaneous(
                        (
                            (chunk_len, Hq, self.max_num_kv_splits, D + 1),
                            torch.float32,
                        ),
                        ((chunk_len, Hq, D), query.dtype),
                        ((chunk_len, Hq), torch.float32),
                    )
                )

            out[chunk_start:chunk_end] = triton_turboquant_decode_attention(
                query=chunk,
                kv_cache=kv_cache,
                block_table=synth_block_table,
                seq_lens=synth_seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_fp8=self.tq_config.key_fp8,
                norm_correction=self.tq_config.norm_correction,
                PiT=PiT,
                mid_o_buf=mid_o_buf,
                output_buf=output_buf,
                lse_buf=lse_buf,
                buf_holder=layer,
                max_num_kv_splits=self.max_num_kv_splits,
                sliding_window=self.sliding_window,
                mm_prefix_range=mm_prefix_range,
            )

        return out

    def _ensure_on_device(self, layer, device):
        """One-time derivation of TQ buffers (rotation matrix, midpoints).

        The Hadamard rotation is shared across all layers: random sign
        flips do not improve Lloyd-Max quantization quality because the
        quantizer is symmetric around zero (sign-flipping a coordinate
        maps it to the mirror centroid with identical distortion).
        """
        if not hasattr(layer, "_tq_cached"):
            D = self.head_size

            # Pure Hadamard: orthonormal + symmetric (H = H^T), enabling
            # in-kernel butterfly fusion and trivial inverse for continuation.
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = H
            layer._tq_Pi = H
            # fp16 copy for rotation in continuation prefill path
            layer._tq_Pi_half = H.to(torch.float16)

            # Centroids for Lloyd-Max quantization.
            layer._tq_centroids = get_centroids(D, self.tq_config.centroid_bits).to(
                device=device, dtype=torch.float32
            )

            c_sorted, _ = layer._tq_centroids.sort()
            layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            layer._tq_cached = True

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store compressed K/V into the combined TQ cache.

        Called as a separate custom op (unified_kv_cache_update) BEFORE
        the attention forward, matching FlashAttention's split pattern.
        slot_mapping is already sliced to num_actual_tokens by the caller.
        """
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._store_kv(k, v, kv_cache, slot_mapping, layer)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "TurboQuantMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        # Slice to actual tokens
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)

        # Get TQ buffers, ensure on device (one-time migration).
        # Use Any-typed alias for dynamic _tq_* attrs set by _ensure_on_device.
        tq_layer: Any = layer
        device = q.device
        self._ensure_on_device(tq_layer, device)
        Pi = tq_layer._tq_Pi
        PiT = tq_layer._tq_PiT
        centroids = tq_layer._tq_centroids

        # Compute attention (KV cache was already updated by do_kv_cache_update)
        # With reorder_batch_threshold=1, decodes come first in the batch.
        # num_decodes/num_decode_tokens from metadata give the split point.
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        if not attn_metadata.is_prefill:
            # Pure decode batch — fast path
            attn_out = self._decode_attention(
                q, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
        elif num_decodes == 0:
            # Pure prefill batch
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(
                q,
                k,
                v,
                kv_cache,
                attn_metadata,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )
        else:
            # Mixed batch: decodes first (guaranteed by reorder_batch).
            attn_out = torch.empty(
                N, self.num_heads, self.head_size, device=device, dtype=q.dtype
            )

            # --- Decode portion (first num_decodes requests) ---
            # Use full-batch max_seq_len as safe upper bound (no GPU sync).
            decode_meta = TurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
                mm_prefix_range_tensor=mm_prefix_range_tensor[:num_decodes]
                if mm_prefix_range_tensor is not None
                else None,
            )
            attn_out[:num_decode_tokens] = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, Pi, centroids, PiT, layer
            )

            # --- Prefill portion (remaining requests) ---
            # CRITICAL: use prefill-specific max_seq_len so flash_attn's
            # fast path (max_query_len == max_seq_len) triggers for
            # first-chunk prefills. Using full-batch max_seq_len breaks
            # this because decode requests inflate max_seq_len.
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            # Use the CPU-resident `seq_lens` upper-bound from the metadata
            # (populated in the builder) to compute the prefill sub-batch
            # max without a GPU→CPU sync.
            if attn_metadata.seq_lens_cpu is not None:
                prefill_max_seq = int(attn_metadata.seq_lens_cpu[num_decodes:].max())
            else:
                prefill_max_seq = attn_metadata.max_seq_len
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_qsl_cpu = None
            if attn_metadata.query_start_loc_cpu is not None:
                prefill_qsl_cpu = (
                    attn_metadata.query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
            prefill_meta = TurboQuantMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
                query_start_loc_cpu=prefill_qsl_cpu,
                seq_lens_cpu=attn_metadata.seq_lens_cpu[num_decodes:]
                if attn_metadata.seq_lens_cpu is not None
                else None,
                mm_prefix_range_tensor=mm_prefix_range_tensor[num_decodes:]
                if mm_prefix_range_tensor is not None
                else None,
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )

        # Write into output buffer: attn_out is (N, Hq, D)
        # output may be 2D (N, Hq*D) or 3D (N, Hq, D)
        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Store K/V into combined cache (vectorized)                         #
    # ------------------------------------------------------------------ #
    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        """Quantize + store via fused Triton kernel."""
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            layer._tq_midpoints,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
        )

    # ------------------------------------------------------------------ #
    #  Prefill: SDPA on raw Q/K/V with causal mask                        #
    # ------------------------------------------------------------------ #
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
        mm_prefix_range_tensor = getattr(attn_metadata, "mm_prefix_range_tensor", None)

        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # Both are Python ints — no GPU sync.
        if (
            self._can_use_flash_attn
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
            and not self._needs_sliding_window_mask(attn_metadata.max_seq_len)
            and mm_prefix_range_tensor is None
        ):
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
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Prefer the CPU-resident copies from the metadata if populated —
        # otherwise `.tolist()` on GPU tensors forces a synchronizing copy.
        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        # Pre-allocate cu_seqlens for single-request flash_attn calls
        # to avoid per-request host→device tensor creation.
        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        arange_cache = self._get_arange_cache(
            attn_metadata.max_seq_len,
            device=query.device,
            dtype=attn_metadata.seq_lens.dtype,
        )

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
            mm_prefix_ranges = self._get_mm_prefix_ranges(attn_metadata, i)

            if self.kv_sharing_target_layer_name is not None:
                # KV-sharing layers reuse the target layer's cache. Their raw
                # K/V tensors are intentionally not normalized/rotated by the
                # model, so prefill must read from the shared TQ cache instead.
                out = self._cache_prefill_attention(
                    layer,
                    q_seq,
                    kv_cache,
                    attn_metadata.block_table[i : i + 1],
                    seq_len,
                    seq_len - q_len,
                    Pi,
                    centroids,
                    mm_prefix_ranges=mm_prefix_ranges,
                )
            elif q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if self._needs_sliding_window_mask(seq_len) or (
                    mm_prefix_ranges is not None
                ):
                    out = self._sdpa_with_causal_and_sliding_mask(
                        q_seq,
                        k_seq,
                        v_seq,
                        query_start_pos=0,
                        mm_prefix_ranges=mm_prefix_ranges,
                    )
                elif self._can_use_flash_attn:
                    # Assign to slice to avoid gpu/cpu sync.
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
                    out = self._sdpa_causal_prefill(q_seq, k_seq, v_seq)
            else:
                # Continuation chunk: tokens already stored to TQ cache
                # by do_kv_cache_update. Use decode kernel directly to
                # avoid O(cached_len) full-dequant per continuation.
                # For large continuations, use flash-attn when that path is
                # available; otherwise run decode kernel chunks.
                cached_len = seq_len - q_len
                if q_len <= _CONTINUATION_DECODE_THRESHOLD or not (
                    self._can_use_flash_prefill(seq_len, mm_prefix_ranges)
                ):
                    out = self._decode_prefill_from_cache(
                        q_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        query_start_pos=cached_len,
                        Pi=Pi,
                        centroids=centroids,
                        PiT=PiT,
                        mm_prefix_ranges=mm_prefix_ranges,
                        seq_lens_dtype=arange_cache.dtype,
                        layer=layer,
                    )
                else:
                    # Large continuation: dequant cached K/V and use
                    # flash_attn for better throughput.
                    out = self._continuation_prefill(
                        layer,
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                        mm_prefix_ranges=mm_prefix_ranges,
                    )

            output[q_start:q_end] = out.to(query.dtype)

        return output

    def _dequant_cached_kv(
        self,
        layer: Any,
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cache_len: int,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize the first cache_len tokens from a single request cache."""
        assert cache_len > 0

        Hk = self.num_kv_heads
        D = self.head_size
        device = kv_cache.device
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)

        alloc_len = math.ceil(cache_len / block_size) * block_size
        buf_shape = (1, Hk, alloc_len, D)
        # Use WorkspaceManager for dequant buffers. It is shared across layers
        # and avoids per-layer allocations that would break CUDA graph capture.
        k_buf, v_buf = current_workspace_manager().get_simultaneous(
            (buf_shape, torch.float16),
            (buf_shape, torch.float16),
        )
        # Skip zeroing: the kernel writes all positions up to alloc_len, and
        # callers only read the first cache_len positions.
        k_cached = k_buf[:, :, :alloc_len, :]
        v_cached = v_buf[:, :, :alloc_len, :]

        grid = (alloc_len, Hk)
        _tq_full_dequant_kv[grid](
            kv_cache,
            block_table,
            layer._tq_centroids,
            k_cached,
            v_cached,
            k_cached.stride(0),
            k_cached.stride(1),
            k_cached.stride(2),
            v_cached.stride(0),
            v_cached.stride(1),
            v_cached.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_HEADS=Hk,
            MSE_BYTES=self._mse_bytes,
            KPS=self.tq_config.key_packed_size,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=self._val_data_bytes,
            MSE_BITS=self.tq_config.key_mse_bits,
            KEY_FP8=1 if self.tq_config.key_fp8 else 0,
            BLOCK_D=BLOCK_D,
            NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
            FP8_E4B15=_use_fp8_e4b15(device.index or 0),
            num_warps=4,
        )

        if not self.tq_config.key_fp8:
            Pi_half = layer._tq_Pi_half
            k_flat = k_cached[0, :, :cache_len, :].reshape(-1, D)
            k_flat = k_flat @ Pi_half
            k_cached_trim = k_flat.reshape(Hk, cache_len, D).transpose(0, 1)
        else:
            k_cached_trim = k_cached[0, :, :cache_len, :].transpose(0, 1)

        v_cached_trim = v_cached[0, :, :cache_len, :].transpose(0, 1)
        return k_cached_trim.to(output_dtype), v_cached_trim.to(output_dtype)

    def _prefill_attention_with_kv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_start_pos: int,
        mm_prefix_ranges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_len = query.shape[0]
        seq_len = key.shape[0]
        device = query.device

        if (
            self._can_use_flash_attn
            and not self._needs_sliding_window_mask(seq_len)
            and mm_prefix_ranges is None
        ):
            # Reuse pre-allocated cu_seqlens (avoid host→device transfer).
            if not hasattr(self, "_cu_2_q"):
                self._cu_2_q = torch.zeros(2, device=device, dtype=torch.int32)
                self._cu_2_k = torch.zeros(2, device=device, dtype=torch.int32)
            self._cu_2_q[1:2] = q_len
            self._cu_2_k[1:2] = seq_len
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=self._cu_2_q,
                cu_seqlens_k=self._cu_2_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
            )
        return self._sdpa_with_causal_and_sliding_mask(
            query,
            key,
            value,
            query_start_pos=query_start_pos,
            mm_prefix_ranges=mm_prefix_ranges,
        )

    def _cache_prefill_attention(
        self,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        seq_len: int,
        query_start_pos: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        mm_prefix_ranges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._can_use_flash_prefill(seq_len, mm_prefix_ranges):
            return self._decode_prefill_from_cache(
                query,
                kv_cache,
                block_table,
                query_start_pos=query_start_pos,
                Pi=Pi,
                centroids=centroids,
                PiT=getattr(layer, "_tq_PiT", None),
                mm_prefix_ranges=mm_prefix_ranges,
                layer=layer,
            )

        k_full, v_full = self._dequant_cached_kv(
            layer,
            kv_cache,
            block_table,
            seq_len,
            query.dtype,
        )
        return self._prefill_attention_with_kv(
            query,
            k_full,
            v_full,
            query_start_pos=query_start_pos,
            mm_prefix_ranges=mm_prefix_ranges,
        )

    def _continuation_prefill(
        self,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        mm_prefix_ranges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Handle continuation chunk by dequanting cached K/V from TQ cache.

        Dequants previously cached K/V, concatenates with the current
        chunk's raw K/V, then runs flash_attn with causal masking.
        """
        q_len, _, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        k_cached_trim, v_cached_trim = self._dequant_cached_kv(
            layer,
            kv_cache,
            block_table,
            cached_len,
            query.dtype,
        )

        # Concatenate cached + current chunk K/V (match query dtype)
        # Pre-allocate full K/V buffer, copy into slices (no cat alloc)
        qdtype = query.dtype
        k_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        v_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        k_full[:cached_len] = k_cached_trim.to(qdtype)
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached_trim.to(qdtype)
        v_full[cached_len:] = val_chunk

        # Attention: q_len queries attending to seq_len K/V with causal mask
        return self._prefill_attention_with_kv(
            query,
            k_full,
            v_full,
            query_start_pos=cached_len,
            mm_prefix_ranges=mm_prefix_ranges,
        )

    # ------------------------------------------------------------------ #
    #  Decode: Triton TQ decode attention                                 #
    # ------------------------------------------------------------------ #
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
        # Acquire shared decode scratch buffers from WorkspaceManager.
        # Layers execute sequentially so one set of buffers is sufficient.
        # Falls back to kernel-internal allocation if workspace unavailable.
        B = query.shape[0]
        D = self.head_size
        S = self.max_num_kv_splits
        Hq = self.num_heads
        mid_o_buf = output_buf = lse_buf = None
        if is_workspace_manager_initialized():
            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.
            mid_o_buf, output_buf, lse_buf = (
                current_workspace_manager().get_simultaneous(
                    ((B, Hq, S, D + 1), torch.float32),
                    ((B, Hq, D), query.dtype),
                    ((B, Hq), torch.float32),
                )
            )

        result = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=layer,
            max_num_kv_splits=self.max_num_kv_splits,
            sliding_window=self.sliding_window,
            mm_prefix_range=attn_metadata.mm_prefix_range_tensor,
        )
        return result
