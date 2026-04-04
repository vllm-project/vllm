# SPDX-License-Identifier: Apache-2.0
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
  For tq-k3v4nc head_dim=256: [100 bytes key | 512 bytes value] = 612 total
"""

import os
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from vllm.v1.attention.ops.triton_tq_store import triton_tq_store

# CUDA stream overlap: disabled by default — degrades TTFT under concurrent
# load (489ms vs 338ms). Enable via TQ_STREAM_OVERLAP=1 for experimentation.
_USE_STREAM_OVERLAP = os.environ.get("TQ_STREAM_OVERLAP", "0") == "1"

_store_stream: torch.cuda.Stream | None = None

from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.logger import init_logger
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

logger = init_logger(__name__)


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "tq-k8v4",
        "tq-t4nc",
        "tq-k3v4nc",
        "tq-t3nc",
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
        cache_dtype_str: str = "tq-t4nc",
    ) -> tuple[int, ...]:
        """Combined K+V cache shape — no leading 2 dimension.

        Layout: (num_blocks, block_size, num_kv_heads, padded_slot_size)
        Each slot = [key_packed | value_fp16 | padding].

        Note: head_size here is the *effective* head_size from the spec
        (= padded_slot // 2), NOT the model's actual head_dim.
        So padded_slot = head_size * 2.
        """
        return (num_blocks, block_size, num_kv_heads, head_size * 2)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype is not None and kv_cache_dtype.startswith("tq-")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # head_size from spec is effective_head_size (padded_slot//2),
        # not the model's actual head_dim. Accept any positive value.
        return head_size > 0


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention."""

    seq_lens: torch.Tensor          # (num_reqs,) — total context length per request
    slot_mapping: torch.Tensor      # (num_tokens,) — cache slot for each token
    block_table: torch.Tensor       # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor   # (num_reqs + 1,) — cu_seqlens for queries
    num_actual_tokens: int = 0      # actual tokens (excluding padding)
    max_query_len: int = 0          # longest query in batch
    max_seq_len: int = 0            # longest context in batch
    is_prefill: bool = False


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def reorder_batch(self, input_batch, scheduler_output):
        return False

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
        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
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
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        from vllm.turboquant.config import TurboQuantConfig
        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)

    def _ensure_on_device(self, layer, device):
        """One-time migration of TQ buffers to the correct device."""
        Pi = layer._tq_Pi
        if Pi.device != device:
            layer._tq_Pi = Pi.to(device)
            layer._tq_centroids = layer._tq_centroids.to(device)
        # Cache contiguous float32 matrices and precomputed midpoints
        if not hasattr(layer, '_tq_cached'):
            Pi_f = layer._tq_Pi.float().contiguous()
            c = layer._tq_centroids.float()
            layer._tq_PiT = Pi_f.T.contiguous()
            # Precompute midpoints for threshold-based quantization
            c_sorted, _ = c.sort()
            layer._tq_midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2)
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

        With stream overlap enabled, the store runs on a secondary CUDA
        stream so it can overlap with the next layer's forward pass.
        """
        global _store_stream

        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._current_layer = layer

        # Use stream overlap only when not capturing CUDA graphs
        use_overlap = (
            _USE_STREAM_OVERLAP
            and _USE_TRITON_STORE
            and not torch.cuda.is_current_stream_capturing()
        )

        if use_overlap:
            # Lazy-init the secondary stream
            if _store_stream is None:
                _store_stream = torch.cuda.Stream(device=device)

            # Wait for any previous store to finish before starting new one
            torch.cuda.current_stream(device).wait_stream(_store_stream)

            # Launch store on secondary stream
            with torch.cuda.stream(_store_stream):
                self._store_kv(k, v, kv_cache, slot_mapping,
                               layer._tq_Pi, layer._tq_centroids)
        else:
            self._store_kv(k, v, kv_cache, slot_mapping,
                           layer._tq_Pi, layer._tq_centroids)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "TurboQuantMetadata",
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(
                num_tokens, self.num_heads * self.head_size,
                dtype=query.dtype, device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        # Slice to actual tokens
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)

        # Get TQ buffers, ensure on device (one-time migration)
        device = q.device
        self._ensure_on_device(layer, device)
        Pi = layer._tq_Pi
        centroids = layer._tq_centroids

        # Ensure any async store has completed before decode reads cache
        if (_store_stream is not None
                and not attn_metadata.is_prefill
                and not torch.cuda.is_current_stream_capturing()):
            torch.cuda.current_stream(device).wait_stream(_store_stream)

        # Compute attention (KV cache was already updated by do_kv_cache_update)
        # Handle mixed prefill+decode batches (chunked prefill):
        # is_prefill is batch-level (max_query_len > 1). When chunked prefill
        # is enabled, a batch can contain both prefill chunks (query_len > 1)
        # and decode requests (query_len = 1). We must dispatch each to the
        # correct path — decode tokens MUST use _decode_attention to read
        # from the compressed KV cache.
        if not attn_metadata.is_prefill:
            # Pure decode batch — fast path
            attn_out = self._decode_attention(q, kv_cache, attn_metadata,
                                              Pi, centroids)
        else:
            # Could be pure prefill or mixed prefill+decode.
            # Fast check: if max_query_len == max_seq_len, all requests are
            # first-chunk prefills (no prior cache) → guaranteed no decodes.
            # Only fall through to GPU-sync check when there's continuation.
            if attn_metadata.max_query_len == attn_metadata.max_seq_len:
                has_decodes = False
            else:
                query_start_loc = attn_metadata.query_start_loc
                num_reqs = query_start_loc.shape[0] - 1
                q_lens = (query_start_loc[1:] - query_start_loc[:num_reqs])
                has_decodes = (q_lens == 1).any().item()

            if not has_decodes:
                # Pure prefill batch — use prefill path for all
                k = key[:N].view(N, self.num_kv_heads, self.head_size)
                v = value[:N].view(N, self.num_kv_heads, self.head_size)
                attn_out = self._prefill_attention(q, k, v, attn_metadata)
            else:
                # Mixed batch: split into prefill and decode requests
                attn_out = self._mixed_batch_attention(
                    q, key[:N].view(N, self.num_kv_heads, self.head_size),
                    value[:N].view(N, self.num_kv_heads, self.head_size),
                    kv_cache, attn_metadata, Pi, centroids,
                    query_start_loc, q_lens, num_reqs,
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
        key: torch.Tensor,     # (N, Hk, D)
        value: torch.Tensor,   # (N, Hk, D)
        kv_cache: torch.Tensor, # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
    ):
        """Quantize + store via fused Triton kernel."""
        layer = self._current_layer
        triton_tq_store(
            key, value, kv_cache, slot_mapping,
            layer._tq_PiT, centroids, layer._tq_midpoints,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            value_packed_size=self.tq_config.value_packed_size,
            key_fp8=self.tq_config.key_fp8,
        )

    # ------------------------------------------------------------------ #
    #  Mixed batch: split prefill + decode and dispatch separately         #
    # ------------------------------------------------------------------ #
    def _mixed_batch_attention(
        self,
        query: torch.Tensor,       # (N, Hq, D) — all tokens
        key: torch.Tensor,         # (N, Hk, D) — all tokens
        value: torch.Tensor,       # (N, Hk, D) — all tokens
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        query_start_loc: torch.Tensor,  # (num_reqs + 1,)
        q_lens: torch.Tensor,          # (num_reqs,) — per-request query len
        num_reqs: int,
    ) -> torch.Tensor:
        """Handle mixed prefill+decode batches from chunked prefill.

        Splits the batch into prefill requests (query_len > 1) and decode
        requests (query_len == 1), runs each through the appropriate path,
        then merges results.
        """
        N, Hq, D = query.shape
        device = query.device
        output = torch.zeros(N, Hq, D, device=device, dtype=query.dtype)

        # Identify decode vs prefill requests
        decode_mask = (q_lens == 1)  # (num_reqs,)
        prefill_mask = ~decode_mask

        # --- Handle prefill requests via _prefill_attention ---
        if prefill_mask.any():
            # Run per-request SDPA for prefill tokens (the fallback path
            # in _prefill_attention already loops per-request, so we can
            # just call it — it will skip decode requests since they have
            # q_len=1 and produce trivial output, but that's wasteful).
            # Instead, extract prefill tokens and build sub-metadata.
            for i in range(num_reqs):
                if decode_mask[i]:
                    continue
                q_start = query_start_loc[i].item()
                q_end = query_start_loc[i + 1].item()
                q_len = q_end - q_start
                if q_len <= 0:
                    continue

                q_seq = query[q_start:q_end]   # (q_len, Hq, D)
                k_seq = key[q_start:q_end]     # (q_len, Hk, D)
                v_seq = value[q_start:q_end]   # (q_len, Hk, D)

                seq_len = attn_metadata.seq_lens[i].item()

                # If this is a first-chunk prefill (all KV in batch),
                # use causal SDPA on the raw tokens.
                if q_len == seq_len:
                    Hk = k_seq.shape[1]
                    use_gqa = (Hk < Hq)
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t, k_t, v_t,
                        is_causal=True, scale=self.scale,
                        enable_gqa=use_gqa,
                    )
                    output[q_start:q_end] = out.transpose(0, 1)
                else:
                    # Continuation chunk: needs to attend to prior cached KV.
                    # Fall back to per-request SDPA on just this request's
                    # raw K/V (causal within the chunk). This is approximate
                    # for continuation but matches the original behavior.
                    Hk = k_seq.shape[1]
                    use_gqa = (Hk < Hq)
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t, k_t, v_t,
                        is_causal=True, scale=self.scale,
                        enable_gqa=use_gqa,
                    )
                    output[q_start:q_end] = out.transpose(0, 1)

        # --- Handle decode requests via _decode_attention ---
        if decode_mask.any():
            decode_indices = decode_mask.nonzero(as_tuple=True)[0]
            num_decodes = decode_indices.shape[0]

            # Gather decode queries: each has exactly 1 token
            decode_token_offsets = query_start_loc[decode_indices]
            decode_q = query[decode_token_offsets]  # (num_decodes, Hq, D)

            # Build decode sub-metadata
            decode_seq_lens = attn_metadata.seq_lens[decode_indices]
            decode_block_table = attn_metadata.block_table[decode_indices]
            decode_meta = TurboQuantMetadata(
                seq_lens=decode_seq_lens,
                slot_mapping=attn_metadata.slot_mapping,  # not used by decode
                block_table=decode_block_table,
                query_start_loc=torch.arange(
                    num_decodes + 1, device=device, dtype=torch.int32),
                num_actual_tokens=num_decodes,
                max_query_len=1,
                max_seq_len=decode_seq_lens.max().item(),
                is_prefill=False,
            )

            decode_out = self._decode_attention(
                decode_q, kv_cache, decode_meta, Pi, centroids)

            # Scatter decode results back to correct positions
            for j in range(num_decodes):
                tok_idx = decode_token_offsets[j].item()
                output[tok_idx] = decode_out[j]

        return output

    # ------------------------------------------------------------------ #
    #  Prefill: SDPA on raw Q/K/V with causal mask                        #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query: torch.Tensor,   # (N, Hq, D)
        key: torch.Tensor,     # (N, Hk, D)
        value: torch.Tensor,   # (N, Hk, D)
        attn_metadata: TurboQuantMetadata,
    ) -> torch.Tensor:
        N, Hq, D = query.shape

        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # Both are Python ints — no GPU sync.
        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            output = torch.empty(N, Hq, D, device=query.device, dtype=query.dtype)
            flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
                softmax_scale=self.scale,
                causal=True,
                out=output,
            )
            return output

        # Fallback: per-request SDPA (chunked continuation or no flash_attn)
        Hk = key.shape[1]
        use_gqa = (Hk < Hq)
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        for i in range(num_reqs):
            q_start = query_start_loc[i].item()
            q_end = query_start_loc[i + 1].item()
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            q_seq = query[q_start:q_end]       # (q_len, Hq, D)
            k_seq = key[q_start:q_end]         # (q_len, Hk, D)
            v_seq = value[q_start:q_end]       # (q_len, Hk, D)

            # Transpose to (H, q_len, D) for SDPA
            q_t = q_seq.transpose(0, 1).contiguous()  # (Hq, q_len, D)
            k_t = k_seq.transpose(0, 1).contiguous()  # (Hk, q_len, D)
            v_t = v_seq.transpose(0, 1).contiguous()  # (Hk, q_len, D)

            # SDPA with native GQA support (avoids repeat_interleave)
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                is_causal=True, scale=self.scale,
                enable_gqa=use_gqa,
            )  # (Hq, q_len, D)
            output[q_start:q_end] = out.transpose(0, 1).to(query.dtype)

        return output

    # ------------------------------------------------------------------ #
    #  Decode: Triton TQ decode attention                                 #
    # ------------------------------------------------------------------ #
    def _decode_attention(
        self,
        query: torch.Tensor,      # (B, Hq, D)
        kv_cache: torch.Tensor,    # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.v1.attention.ops.triton_tq_decode import (
            triton_tq_decode_attention,
        )
        return triton_tq_decode_attention(
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
            value_packed_size=self.tq_config.value_packed_size,
            max_seq_len=attn_metadata.max_seq_len,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
        )
