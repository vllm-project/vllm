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
from torch.nn.attention.bias import causal_lower_right

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
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
_STREAMING_PREFILL_QUERY_TILE = 512
_STREAMING_PREFILL_KV_TILE = 1024
_RAW_PREFILL_STREAMING_THRESHOLD = 1024
_RAW_PREFILL_QUERY_TILE = 512
_RAW_PREFILL_KV_TILE = 512


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
        return kv_cache_dtype.startswith("turboquant_")

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
            mm_prefix_range_tensor=getattr(cam, "mm_prefix_range_tensor", None),
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
        self._can_use_flash_attn = _HAS_FLASH_ATTN and head_size <= 256

        # Fixed NUM_KV_SPLITS (grid dims must be constant for cudagraph,
        # and benchmarks show no regression vs dynamic in eager mode).
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
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

    def _can_use_dequant_prefill(
        self,
        seq_len: int,
        mm_prefix_ranges: torch.Tensor | None,
    ) -> bool:
        return not self._needs_sliding_window_mask(seq_len) and mm_prefix_ranges is None

    def _should_use_streaming_prefill(
        self,
        q_len: int,
        seq_len: int,
        mm_prefix_ranges: torch.Tensor | None,
    ) -> bool:
        """Use streaming only for large non-flash continuation prefills."""
        if q_len <= _CONTINUATION_DECODE_THRESHOLD:
            return False
        if seq_len <= q_len:
            return False
        if not is_workspace_manager_initialized():
            return False
        return self._can_use_dequant_prefill(
            seq_len, mm_prefix_ranges
        ) and not self._can_use_flash_prefill(seq_len, mm_prefix_ranges)

    def _should_use_streaming_raw_prefill(
        self,
        q_len: int,
        seq_len: int,
        mm_prefix_ranges: torch.Tensor | None,
    ) -> bool:
        """Use tiled PyTorch attention for large raw-KV non-flash prefills."""
        if q_len <= _RAW_PREFILL_STREAMING_THRESHOLD:
            return False
        if self._can_use_flash_prefill(seq_len, mm_prefix_ranges):
            return False
        return (
            self._needs_sliding_window_mask(seq_len)
            or mm_prefix_ranges is not None
            or not self._can_use_flash_attn
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

        q_pos = torch.arange(q_len, device=device) + query_start_pos
        k_pos = torch.arange(kv_len, device=device)
        mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        if self.sliding_window is not None:
            mask = mask & (
                (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)) < self.sliding_window
            )
        if mm_prefix_ranges is not None:
            starts = mm_prefix_ranges[:, 0]
            ends = mm_prefix_ranges[:, 1]
            valid = starts < ends
            q_in_range = (q_pos[:, None] >= starts) & (q_pos[:, None] <= ends) & valid
            k_in_range = (k_pos[:, None] >= starts) & (k_pos[:, None] <= ends) & valid
            mask = mask | (q_in_range.unsqueeze(1) & k_in_range.unsqueeze(0)).any(
                dim=-1
            )

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

    def _sdpa_lower_right_causal_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Run causal SDPA for continuation prefill without a dense mask."""
        q_t = query.transpose(0, 1).unsqueeze(0)
        k_t = key.transpose(0, 1).unsqueeze(0)
        v_t = value.transpose(0, 1).unsqueeze(0)
        bias = causal_lower_right(query.shape[0], key.shape[0])
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=bias,
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
        seq_lens_dtype: torch.dtype = torch.int32,
        layer: Any = None,
    ) -> torch.Tensor:
        q_len, _, _ = query.shape
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
            decode_meta = TurboQuantMetadata(
                seq_lens=synth_seq_lens,
                slot_mapping=torch.empty(0, device=query.device, dtype=torch.long),
                block_table=synth_block_table,
                query_start_loc=torch.arange(
                    chunk_len + 1,
                    device=query.device,
                    dtype=torch.int32,
                ),
                num_actual_tokens=chunk_len,
                max_query_len=1,
                max_seq_len=int(query_start_pos + chunk_end),
                is_prefill=False,
            )
            out[chunk_start:chunk_end] = self._decode_attention(
                chunk, kv_cache, decode_meta, Pi, centroids, PiT, layer
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
            mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor
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
    #  Prefill: raw SDPA/flash and cached TQ continuation paths           #
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

        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1
        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Prefer CPU-resident copies from metadata if available. Falling back
        # to .tolist() preserves the old behavior but may synchronize.
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
            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]
            mm_prefix_ranges = self._get_mm_prefix_ranges(attn_metadata, i)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if self._should_use_streaming_raw_prefill(
                    q_len, seq_len, mm_prefix_ranges
                ):
                    out = self._streaming_prefill_from_raw_kv(
                        q_seq,
                        k_seq,
                        v_seq,
                        query_start_pos=0,
                        mm_prefix_ranges=mm_prefix_ranges,
                    )
                elif self._needs_sliding_window_mask(seq_len) or (
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
                    self._cu_2[1:2] = q_len
                    out = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=self._cu_2,
                        cu_seqlens_k=self._cu_2,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                else:
                    out = self._sdpa_causal_prefill(q_seq, k_seq, v_seq)
            else:
                # Continuation chunk: tokens have already been stored to the
                # TQ cache by do_kv_cache_update. Use decode replay for small
                # unmasked continuations, otherwise use full/streaming prefill.
                cached_len = seq_len - q_len
                needs_masked_continuation = self._needs_sliding_window_mask(
                    seq_len
                ) or (mm_prefix_ranges is not None)
                if q_len <= _CONTINUATION_DECODE_THRESHOLD and (
                    not needs_masked_continuation
                ):
                    out = self._decode_prefill_from_cache(
                        q_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        query_start_pos=cached_len,
                        Pi=Pi,
                        centroids=centroids,
                        PiT=PiT,
                        seq_lens_dtype=arange_cache.dtype,
                        layer=layer,
                    )
                else:
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

    def _dequant_cached_kv_range(
        self,
        layer: Any,
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        start: int,
        length: int,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize a contiguous absolute position range from one cache."""
        assert start >= 0
        assert length > 0

        Hk = self.num_kv_heads
        D = self.head_size
        device = kv_cache.device
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)

        buf_shape = (1, Hk, length, D)
        k_buf, v_buf = current_workspace_manager().get_simultaneous(
            (buf_shape, torch.float16),
            (buf_shape, torch.float16),
        )
        k_cached = k_buf[:, :, :length, :]
        v_cached = v_buf[:, :, :length, :]

        grid = (length, Hk)
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
            start,
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
            k_flat = k_cached[0, :, :length, :].reshape(-1, D)
            k_flat = k_flat @ Pi_half
            k_cached_trim = k_flat.reshape(Hk, length, D).transpose(0, 1)
        else:
            k_cached_trim = k_cached[0, :, :length, :].transpose(0, 1)

        v_cached_trim = v_cached[0, :, :length, :].transpose(0, 1)
        return k_cached_trim.to(output_dtype), v_cached_trim.to(output_dtype)

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
        return self._dequant_cached_kv_range(
            layer,
            kv_cache,
            block_table,
            0,
            cache_len,
            output_dtype,
        )

    def _streaming_prefill_from_raw_kv(
        self,
        query: torch.Tensor,  # (q_len, Hq, D)
        key: torch.Tensor,  # (seq_len, Hk, D)
        value: torch.Tensor,  # (seq_len, Hk, D)
        *,
        query_start_pos: int,
        mm_prefix_ranges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Streaming causal/sliding/mm-prefix prefill over raw K/V tensors."""
        q_len, Hq, D = query.shape
        seq_len, Hk, _ = key.shape
        assert Hq % Hk == 0

        device = query.device
        output = torch.empty_like(query)
        kv_head_for_q = torch.arange(Hq, device=device, dtype=torch.int64) // (Hq // Hk)

        def update_tile(
            q: torch.Tensor,
            q_pos: torch.Tensor,
            m: torch.Tensor,
            denom: torch.Tensor,
            acc: torch.Tensor,
            k_tile: torch.Tensor,
            v_tile: torch.Tensor,
            key_start_pos: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            tile_len = k_tile.shape[0]
            key_pos = (
                torch.arange(tile_len, device=device, dtype=torch.int64) + key_start_pos
            )
            visible = key_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            if self.sliding_window is not None:
                visible = visible & (
                    (q_pos.unsqueeze(1) - key_pos.unsqueeze(0)) < self.sliding_window
                )
            if mm_prefix_ranges is not None:
                starts = mm_prefix_ranges[:, 0]
                ends = mm_prefix_ranges[:, 1]
                valid = starts < ends
                q_in_range = (
                    (q_pos[:, None] >= starts) & (q_pos[:, None] <= ends) & valid
                )
                k_in_range = (
                    (key_pos[:, None] >= starts) & (key_pos[:, None] <= ends) & valid
                )
                visible = visible | (
                    q_in_range.unsqueeze(1) & k_in_range.unsqueeze(0)
                ).any(dim=-1)

            k_by_q = k_tile.transpose(0, 1).index_select(0, kv_head_for_q).float()
            v_by_q = v_tile.transpose(0, 1).index_select(0, kv_head_for_q).float()
            scores = torch.bmm(q, k_by_q.transpose(1, 2)) * self.scale
            scores = scores.masked_fill(~visible.unsqueeze(0), -float("inf"))

            tile_m = scores.max(dim=-1).values
            valid_tile = torch.isfinite(tile_m)
            m_new = torch.maximum(m, torch.where(valid_tile, tile_m, m))
            finite_m_new = torch.isfinite(m_new)

            shifted = scores - torch.where(
                finite_m_new,
                m_new,
                torch.zeros_like(m_new),
            ).unsqueeze(-1)
            p = torch.exp(shifted)
            p = torch.where(visible.unsqueeze(0) & finite_m_new.unsqueeze(-1), p, 0.0)

            has_prev = torch.isfinite(m)
            alpha = torch.where(
                has_prev & finite_m_new,
                torch.exp(m - m_new),
                torch.zeros_like(m),
            )
            acc = acc * alpha.unsqueeze(-1) + torch.bmm(p, v_by_q)
            denom = denom * alpha + p.sum(dim=-1)
            return m_new, denom, acc

        query_tile = _RAW_PREFILL_QUERY_TILE
        kv_tile = _RAW_PREFILL_KV_TILE
        for q_start in range(0, q_len, query_tile):
            q_end = min(q_start + query_tile, q_len)
            q_block = query[q_start:q_end]
            block_len = q_end - q_start
            q = q_block.transpose(0, 1).float().contiguous()
            q_pos = (
                torch.arange(block_len, device=device, dtype=torch.int64)
                + query_start_pos
                + q_start
            )
            m = torch.full((Hq, block_len), -float("inf"), device=device)
            denom = torch.zeros((Hq, block_len), device=device)
            acc = torch.zeros((Hq, block_len, D), device=device)

            for k_start in range(0, seq_len, kv_tile):
                k_end = min(k_start + kv_tile, seq_len)
                m, denom, acc = update_tile(
                    q,
                    q_pos,
                    m,
                    denom,
                    acc,
                    key[k_start:k_end],
                    value[k_start:k_end],
                    k_start,
                )

            out = acc / denom.clamp_min(1e-20).unsqueeze(-1)
            output[q_start:q_end] = out.transpose(0, 1).to(query.dtype)

        return output

    def _streaming_prefill_from_tq_cache(
        self,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Streaming causal prefill over TQ cached prefix plus raw current chunk."""
        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        assert seq_len == cached_len + q_len
        assert Hq % Hk == 0

        device = query.device
        output = torch.empty_like(query)
        kv_head_for_q = torch.arange(Hq, device=device, dtype=torch.int64) // (Hq // Hk)

        def update_tile(
            q: torch.Tensor,
            q_pos: torch.Tensor,
            m: torch.Tensor,
            denom: torch.Tensor,
            acc: torch.Tensor,
            k_tile: torch.Tensor,
            v_tile: torch.Tensor,
            key_start_pos: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            tile_len = k_tile.shape[0]
            if tile_len == 0:
                return m, denom, acc

            key_pos = (
                torch.arange(tile_len, device=device, dtype=torch.int64) + key_start_pos
            )
            causal = key_pos.unsqueeze(0) <= q_pos.unsqueeze(1)

            k_by_q = k_tile.transpose(0, 1).index_select(0, kv_head_for_q).float()
            v_by_q = v_tile.transpose(0, 1).index_select(0, kv_head_for_q).float()
            scores = torch.bmm(q, k_by_q.transpose(1, 2)) * self.scale
            scores = scores.masked_fill(~causal.unsqueeze(0), -float("inf"))

            tile_m = scores.max(dim=-1).values
            valid_tile = torch.isfinite(tile_m)
            m_new = torch.maximum(m, torch.where(valid_tile, tile_m, m))
            finite_m_new = torch.isfinite(m_new)

            shifted = scores - torch.where(
                finite_m_new,
                m_new,
                torch.zeros_like(m_new),
            ).unsqueeze(-1)
            p = torch.exp(shifted)
            p = torch.where(causal.unsqueeze(0) & finite_m_new.unsqueeze(-1), p, 0.0)

            has_prev = torch.isfinite(m)
            alpha = torch.where(
                has_prev & finite_m_new,
                torch.exp(m - m_new),
                torch.zeros_like(m),
            )
            acc = acc * alpha.unsqueeze(-1) + torch.bmm(p, v_by_q)
            denom = denom * alpha + p.sum(dim=-1)
            return m_new, denom, acc

        query_tile = _STREAMING_PREFILL_QUERY_TILE
        kv_tile = _STREAMING_PREFILL_KV_TILE
        for q_start in range(0, q_len, query_tile):
            q_end = min(q_start + query_tile, q_len)
            q_block_len = q_end - q_start
            q = query[q_start:q_end].transpose(0, 1).float().contiguous()
            q_pos = (
                torch.arange(q_block_len, device=device, dtype=torch.int64)
                + cached_len
                + q_start
            )
            m = torch.full(
                (Hq, q_block_len),
                -float("inf"),
                device=device,
                dtype=torch.float32,
            )
            denom = torch.zeros((Hq, q_block_len), device=device, dtype=torch.float32)
            acc = torch.zeros((Hq, q_block_len, D), device=device, dtype=torch.float32)

            for tile_start in range(0, cached_len, kv_tile):
                tile_len = min(kv_tile, cached_len - tile_start)
                k_tile, v_tile = self._dequant_cached_kv_range(
                    layer,
                    kv_cache,
                    block_table,
                    tile_start,
                    tile_len,
                    query.dtype,
                )
                m, denom, acc = update_tile(
                    q, q_pos, m, denom, acc, k_tile, v_tile, tile_start
                )

            for local_start in range(0, q_len, kv_tile):
                local_end = min(local_start + kv_tile, q_len)
                m, denom, acc = update_tile(
                    q,
                    q_pos,
                    m,
                    denom,
                    acc,
                    key_chunk[local_start:local_end],
                    val_chunk[local_start:local_end],
                    cached_len + local_start,
                )

            out = acc / denom.clamp_min(1e-20).unsqueeze(-1)
            output[q_start:q_end] = out.transpose(0, 1).to(query.dtype)

        return output

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

        if self._can_use_flash_prefill(seq_len, mm_prefix_ranges):
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
        if self.sliding_window is None and mm_prefix_ranges is None:
            return self._sdpa_lower_right_causal_prefill(query, key, value)
        return self._sdpa_with_causal_and_sliding_mask(
            query,
            key,
            value,
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
        """Handle continuation chunk with tiled/full dequant over cached TQ K/V."""
        if self._should_use_streaming_prefill(
            query.shape[0], seq_len, mm_prefix_ranges
        ):
            return self._streaming_prefill_from_tq_cache(
                layer,
                query,
                key_chunk,
                val_chunk,
                kv_cache,
                block_table,
                cached_len,
                seq_len,
            )

        k_cached_trim, v_cached_trim = self._dequant_cached_kv(
            layer,
            kv_cache,
            block_table,
            cached_len,
            query.dtype,
        )

        q_len, _, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        qdtype = query.dtype
        k_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        v_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        k_full[:cached_len] = k_cached_trim.to(qdtype)
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached_trim.to(qdtype)
        v_full[cached_len:] = val_chunk

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
        )
        return result
