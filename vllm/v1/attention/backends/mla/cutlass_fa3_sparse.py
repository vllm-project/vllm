# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUTLASS FA3 Sparse MLA Attention Backend for vLLM.

This backend uses the vendored CUTLASS FlashAttention3 Sm90 kernel from
sgl-attn to implement sparse MLA attention for DeepSeek-V3.2 and similar
models on SM90 (Hopper) GPUs.

Key differences from FlashMLASparseBackend:
  - Uses BF16 KV cache (576 bytes/token) instead of FP8 (656 bytes/token)
  - No head padding needed (FA3 handles arbitrary head counts natively)
  - Accepts Q_rope and Q_nope (qv) separately (no ConcatMLAQ kernel)
  - 3 sub-kernels: scheduler + main attention + combine
  - ~4x faster per transformer block (~16us vs ~64us)

All execution modes (decode, prefill, mixed) are handled identically:
each token is treated as an independent batch element with seqlen=1.
This simplifies metadata building and CUDA graph support.

Backend priority: Highest for SM90 with kv_cache_dtype="auto".
Graceful fallback to FlashMLA Sparse when FP8 cache requested or non-SM90.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.ops.cutlass_fa3 import is_cutlass_fa3_available

logger = logging.getLogger(__name__)

# Maximum batch size (number of tokens) for which CUTLASS FA3 is used.
# For larger batch sizes, fall back to FlashMLA BF16 sparse prefill kernel.
# FA3 is ~4x faster than FlashMLA for small batches (bs<=16) but regresses
# for larger batches due to higher per-token overhead from the 3-kernel
# launch pattern (scheduler + main + combine) and page_size=1 layout.
MAX_BATCH_SIZE_FOR_FA3 = 16

# FlashMLA sparse prefill kernel requires num_heads padded to this multiple
# on SM90 (Hopper). SM100 (Blackwell) requires 128.
_FLASHMLA_SM90_HEAD_PADDING = 64

# Check if FlashMLA BF16 sparse kernel is available for fallback
_flashmla_sparse_available = False
try:
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    _flashmla_sparse_available = True
except (ImportError, Exception):
    pass

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.kv_cache_interface import AttentionSpec


# ─── Backend Class ────────────────────────────────────────────────────


class CutlassFA3MLASparseBackend(AttentionBackend):
    """CUTLASS FA3 sparse MLA for SM90 (Hopper). BF16 KV cache only.

    When FP8 cache is requested, vLLM's backend selection falls back to
    FlashMLASparseBackend automatically since this backend only supports
    kv_cache_dtype="auto" (which maps to BF16 for MLA).
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "CUTLASS_FA3_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type[CutlassFA3MLASparseMetadataBuilder]:
        return CutlassFA3MLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[CutlassFA3MLASparseImpl]:
        return CutlassFA3MLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 9

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # BF16 cache: 576 bf16 elements per token = 1152 bytes
        # Layout per token: [kv_c_normed(512 bf16) | k_pe(64 bf16)]
        return (num_blocks, block_size, head_size)

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        use_per_head_quant_scales: bool,
        device_capability: DeviceCapability,
        attn_type: str,
        use_non_causal: bool = False,
    ) -> list[str]:
        invalid = super().validate_configuration(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            use_mla=use_mla,
            has_sink=has_sink,
            use_sparse=use_sparse,
            use_mm_prefix=use_mm_prefix,
            use_per_head_quant_scales=use_per_head_quant_scales,
            device_capability=device_capability,
            attn_type=attn_type,
            use_non_causal=use_non_causal,
        )
        if not is_cutlass_fa3_available():
            invalid.append("_cutlass_fa3_C not available (requires CUDA >= 12.4, SM90)")
        return invalid


# ─── Metadata ─────────────────────────────────────────────────────────


@dataclass
class CutlassFA3MLASparseMetadata(AttentionMetadata):
    """Flat metadata for CUTLASS FA3 sparse MLA attention.

    ALL tokens (decode/prefill/mixed) are treated as independent batch
    elements with seqlen=1. There are no nested Decode/Prefill sub-objects.

    This simplification is valid because:
    - Sparse MLA always routes through forward_mqa (not forward_mha)
    - Each token independently selects its top-K KV positions
    - The FA3 kernel handles variable-length sequences via cu_seqlens
    """

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor  # [num_reqs, max_blocks_per_req] int32
    req_id_per_token: torch.Tensor  # [T] int32

    block_size: int = 64
    topk_tokens: int = 2048

    # FA3-specific metadata (pre-allocated for CUDA graph safety)
    cache_seqlens: torch.Tensor | None = None  # [T] int32
    cu_seqlens_q: torch.Tensor | None = None  # [T+1] int32
    cu_seqlens_k: torch.Tensor | None = None  # [T+1] int32

    # For MLAAttention.forward_impl() routing: sparse -> all MQA
    # Setting num_decodes = num_reqs ensures all tokens go through
    # the forward_mqa path (no MHA prefill path).
    num_decodes: int | None = 0
    num_decode_tokens: int | None = 0
    num_prefills: int | None = 0
    num_prefill_tokens: int | None = 0


# ─── Metadata Builder ─────────────────────────────────────────────────


class CutlassFA3MLASparseMetadataBuilder(
    AttentionMetadataBuilder[CutlassFA3MLASparseMetadata]
):
    """Builds CutlassFA3MLASparseMetadata from CommonAttentionMetadata.

    Key design choices:
    - Pre-allocates GPU buffers in __init__ for CUDA graph compatibility
    - All tokens (decode + prefill) treated as independent seqlen=1 elements
    - Uses in-place .copy_() for buffer updates (safe for CUDA graph replay)
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.topk_tokens = 2048
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.block_size = kv_cache_spec.block_size

        # Enable speculative decoding support
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        # Pre-allocate GPU buffers (persist across CUDA graph replays).
        # These are updated in-place via .copy_() before each replay.
        self.req_id_buf = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        self.cache_seqlens_buf = torch.ones(
            max_tokens, dtype=torch.int32, device=device
        )
        self.cu_seqlens_q_buf = torch.arange(
            0, max_tokens + 1, dtype=torch.int32, device=device
        )
        self.cu_seqlens_k_buf = torch.zeros(
            max_tokens + 1, dtype=torch.int32, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CutlassFA3MLASparseMetadata:
        """Build metadata from common attention metadata.

        Converts the request-level metadata into per-token flat metadata:
        - req_id_per_token: maps each token to its request index
        - cache_seqlens: min(seq_len, topk) per token (topk-clipped)
        - cu_seqlens_q: [0, 1, 2, ..., T] (each token = seqlen 1)
        - cu_seqlens_k: cumsum of cache_seqlens
        """
        cm = common_attn_metadata
        T = cm.num_actual_tokens
        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lens = np.diff(starts)

        # req_id_per_token: map each token -> request index
        req_ids = np.repeat(np.arange(len(seg_lens), dtype=np.int32), seg_lens)
        # CUDA graph padding fix: T = cm.num_actual_tokens may include
        # padding tokens (e.g., T=32 when only 31 real tokens exist).
        # The computed req_ids array has sum(seg_lens) elements which
        # equals the real (unpadded) token count. We must:
        # 1) Zero-fill the entire buffer first (safe default for padding)
        # 2) Copy only the actual data using req_ids.shape[0]
        # 3) Slice to padded T for the metadata return
        # This matches the pattern used by FlashMLASparseMetadataBuilder,
        # FlashInferMLASparseMetadataBuilder, and all other sparse backends.
        actual_tokens = req_ids.shape[0]
        self.req_id_buf.fill_(0)
        self.req_id_buf[:actual_tokens].copy_(
            torch.from_numpy(req_ids), non_blocking=True
        )

        # cache_seqlens: UPPER BOUND = min(seq_len, topk) per token.
        # NOTE: This is a per-REQUEST uniform value, NOT the correct
        # per-token causal seqlen. For prefill, token i at position p
        # can only attend to min(p+1, topk) entries, but this gives
        # all tokens min(seq_len, topk). The actual per-token
        # cache_seqlens is computed in forward_mqa() using valid_counts
        # from the index conversion kernel, which correctly reflects
        # the number of valid KV entries per token.
        seq_lens_np = np.asarray(cm.seq_lens_cpu, dtype=np.int32)
        per_tok_seqlens = np.minimum(np.repeat(seq_lens_np, seg_lens), self.topk_tokens)
        # Same CUDA graph padding fix: zero-fill then copy actual data.
        # Default to 1 (safe minimum seqlen for FA3 kernel).
        self.cache_seqlens_buf.fill_(1)
        self.cache_seqlens_buf[:actual_tokens].copy_(
            torch.from_numpy(per_tok_seqlens), non_blocking=True
        )

        # cu_seqlens_q: [0, 1, 2, ..., T] — each token is seqlen=1
        cu_q = self.cu_seqlens_q_buf[: T + 1]

        # cu_seqlens_k: cumsum(cache_seqlens)
        self.cu_seqlens_k_buf[0] = 0
        self.cu_seqlens_k_buf[1 : T + 1].copy_(
            torch.cumsum(self.cache_seqlens_buf[:T], dim=0)
        )
        cu_k = self.cu_seqlens_k_buf[: T + 1]

        return CutlassFA3MLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=T,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=self.req_id_buf[:T],
            block_size=self.block_size,
            topk_tokens=self.topk_tokens,
            cache_seqlens=self.cache_seqlens_buf[:T],
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            # Route ALL tokens through MQA in forward_impl
            num_decodes=cm.num_reqs,
            num_decode_tokens=T,
            num_prefills=0,
            num_prefill_tokens=0,
        )


# ─── Implementation ───────────────────────────────────────────────────


class CutlassFA3MLASparseImpl(SparseMLAAttentionImpl[CutlassFA3MLASparseMetadata]):
    """CUTLASS FA3 sparse MLA attention implementation.

    This implementation replaces the FlashMLA C sparse_attn_fwd_kernel
    with the CUTLASS FA3 Sm90 kernel from sgl-attn, providing ~4x speedup
    per transformer block on Hopper GPUs.

    Key advantages over FlashMLASparseImpl:
    - No head padding (FA3 handles arbitrary head counts natively)
    - No Q concatenation kernel (FA3 accepts q_rope and qv separately)
    - BF16 KV cache (smaller footprint, no dequantization overhead)
    - SM90 warpgroup MMA + TMA for higher compute efficiency
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None = None,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        qk_head_dim: int = 192,
        v_head_dim: int = 128,
        kv_b_proj: ColumnParallelLinear | None = None,
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads  # 16 (per GPU for TP=8)
        self.head_size = head_size  # 576 (kv_lora_rank + qk_rope_head_dim)
        self.scale = float(scale)  # 192**-0.5
        self.num_kv_heads = num_kv_heads  # 1 (MQA)
        self.kv_cache_dtype = kv_cache_dtype  # "auto" (maps to BF16)
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.softmax_scale = scale
        self.topk_tokens = 2048
        self.num_splits = 0  # auto; CUDA-graph safe (deterministic per bs)
        self.logits_soft_cap = float(logits_soft_cap) if logits_soft_cap else 0.0

        # The indexer provides topk_indices_buffer shared across layers
        assert indexer is not None, (
            "CutlassFA3MLASparseImpl requires an indexer "
            "for sparse top-K index selection"
        )
        self.topk_indices_buffer = indexer.topk_indices_buffer

        # DCP (Decode Context Parallelism) requires softmax LSE from the
        # attention kernel. FA3's return_softmax_lse=True is not yet wired
        # through this backend. When DCP is needed, fall back to FlashMLA.
        # TODO: Wire return_softmax_lse=True through forward_mqa for DCP.

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: CutlassFA3MLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """FA3 sparse MLA attention with batch size gating.

        For batch sizes <= MAX_BATCH_SIZE_FOR_FA3 (16), uses the fast
        CUTLASS FA3 kernel. For larger batch sizes, falls back to the
        FlashMLA BF16 sparse prefill kernel which handles larger batches
        more efficiently.

        All execution modes (decode/prefill/mixed) are handled identically:
        each token is an independent batch element with seqlen=1.

        Input:  q = tuple(ql_nope[T, N, 512], q_pe[T, N, 64])
        Output: (attn_out[T, N, 512], None)

        The _v_up_proj in MLAAttention.forward_impl() handles the
        subsequent .view(-1, N, kv_lora_rank) correctly for 3D output.
        """
        # FA3 does not yet return LSE; DCP requires it.
        assert self.dcp_world_size <= 1, (
            "CutlassFA3MLASparseImpl does not support DCP (dcp_world_size > 1). "
            "Use FlashMLA Sparse instead."
        )

        # 1) Unpack Q components
        if isinstance(q, tuple):
            ql_nope, q_pe = q  # [T, N, 512], [T, N, 64]
        else:
            ql_nope = q[..., : self.kv_lora_rank]  # [T, N, 512]
            q_pe = q[..., self.kv_lora_rank :]  # [T, N, 64]
        T = ql_nope.shape[0]

        # 2) Convert topk_indices -> global cache slot indices
        #    Reuses vLLM's existing Triton kernel (no changes needed)
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_convert_req_index_to_global_index,
        )

        global_idx, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,  # [T] int32
            attn_metadata.block_table,  # [R, max_blocks] int32
            self.topk_indices_buffer[:T],  # [T, 2048] int32
            BLOCK_SIZE=attn_metadata.block_size,  # 64
            NUM_TOPK_TOKENS=self.topk_tokens,  # 2048
            return_valid_counts=True,
        )
        # global_idx: [T, 2048] int32 — flat cache slot IDs
        # valid_counts: [T] int32 — number of valid (non -1) entries per token

        # Replace -1 (invalid) page indices with 0 (a safe, valid page index)
        # IN-PLACE for CUDA graph friendliness (no extra allocation).
        global_idx.clamp_(min=0)

        # Use valid_counts as cache_seqlens instead of metadata.cache_seqlens.
        # CRITICAL FIX (Issue #1): metadata cache_seqlens = min(seq_len, topk)
        # can exceed actual valid topk entries for prefill tokens.
        valid_counts.clamp_(min=1)  # in-place; min=1 for seqlen_k safety
        cache_seqlens = valid_counts

        # 3) Route to FA3 or FlashMLA based on batch size
        #    FA3 is faster for small batches (bs<=16) but regresses for
        #    larger batches. FlashMLA BF16 sparse prefill handles larger
        #    batches more efficiently.
        use_fa3 = (T <= MAX_BATCH_SIZE_FOR_FA3) or not _flashmla_sparse_available
        if use_fa3:
            attn_out = self._forward_fa3(
                ql_nope,
                q_pe,
                kv_c_and_k_pe_cache,
                global_idx,
                cache_seqlens,
                attn_metadata,
            )
        else:
            attn_out = self._forward_flashmla_bf16_fallback(
                ql_nope,
                q_pe,
                kv_c_and_k_pe_cache,
                global_idx,
                cache_seqlens,
            )

        # Output: [T, N, 512] — already 3D
        return attn_out, None

    def _forward_fa3(
        self,
        ql_nope: torch.Tensor,  # [T, N, 512]
        q_pe: torch.Tensor,  # [T, N, 64]
        kv_c_and_k_pe_cache: torch.Tensor,
        global_idx: torch.Tensor,  # [T, 2048]
        cache_seqlens: torch.Tensor,  # [T]
        attn_metadata: CutlassFA3MLASparseMetadata,
    ) -> torch.Tensor:
        """CUTLASS FA3 kernel path — fast for small batch sizes (bs<=16).

        Accepts Q_rope and Q_nope (qv) separately, no head padding needed.
        Uses page_size=1 paged KV format with split-KV parallelism.
        """
        T = ql_nope.shape[0]
        S = kv_c_and_k_pe_cache.shape[0] * kv_c_and_k_pe_cache.shape[1]
        kv_flat = kv_c_and_k_pe_cache.reshape(S, self.head_size)  # [S, 576]

        # Split NoPE and RoPE, reshape for FA3 paged format (page_size=1)
        c_kv = kv_flat[:, : self.kv_lora_rank].reshape(
            S, 1, 1, self.kv_lora_rank
        )  # [S, 1, 1, 512]
        k_rope = kv_flat[:, self.kv_lora_rank :].reshape(
            S, 1, 1, self.qk_rope_head_dim
        )  # [S, 1, 1, 64]

        from vllm.v1.attention.ops.cutlass_fa3 import flash_attn_with_kvcache

        attn_out = flash_attn_with_kvcache(
            q=q_pe,  # [T, N, 64]
            k_cache=k_rope,  # [S, 1, 1, 64]
            v_cache=c_kv,  # [S, 1, 1, 512]
            qv=ql_nope,  # [T, N, 512]
            page_table=global_idx,  # [T, 2048]
            cache_seqlens=cache_seqlens,  # [T]
            cu_seqlens_q=attn_metadata.cu_seqlens_q,  # [T+1]
            cu_seqlens_k_new=None,
            max_seqlen_q=1,
            softmax_scale=self.softmax_scale,  # 192**-0.5
            causal=True,
            window_size=(-1, -1),
            softcap=self.logits_soft_cap,
            num_splits=self.num_splits,
        )
        return attn_out  # [T, N, 512]

    def _forward_flashmla_bf16_fallback(
        self,
        ql_nope: torch.Tensor,  # [T, N, 512]
        q_pe: torch.Tensor,  # [T, N, 64]
        kv_c_and_k_pe_cache: torch.Tensor,
        global_idx: torch.Tensor,  # [T, 2048]
        cache_seqlens: torch.Tensor,  # [T]
    ) -> torch.Tensor:
        """FlashMLA BF16 sparse prefill fallback — for larger batch sizes.

        Used when T > MAX_BATCH_SIZE_FOR_FA3 (16). The FlashMLA BF16 sparse
        prefill kernel handles larger batches more efficiently than FA3's
        3-kernel launch pattern (scheduler + main + combine).

        This path:
        1. Concatenates Q components: [ql_nope | q_pe] -> [T, N, 576]
        2. Pads heads to 64 (SM90 FlashMLA requirement)
        3. Reshapes KV cache to [S, 1, 576] (flattened, MQA format)
        4. Reshapes indices to [T, 1, topk] (MQA format)
        5. Calls flash_mla_sparse_fwd with topk_length for valid bounds
        6. Unpads output heads back to N

        The BF16 KV cache format [kv_c_normed(512) | k_pe(64)] is identical
        between FA3 and FlashMLA, so no cache format conversion is needed.
        """
        T = ql_nope.shape[0]
        N = self.num_heads

        # 1) Concatenate Q: [ql_nope(512) | q_pe(64)] -> [T, N, 576]
        q_concat = torch.cat([ql_nope, q_pe], dim=-1)  # [T, N, 576]

        # 2) Pad heads to _FLASHMLA_SM90_HEAD_PADDING (64 for SM90)
        padded_heads = _FLASHMLA_SM90_HEAD_PADDING
        if padded_heads > N:
            q_padded = q_concat.new_zeros((T, padded_heads, q_concat.shape[-1]))
            q_padded[:, :N, :] = q_concat
            q_concat = q_padded

        # 3) Reshape KV cache: (num_blocks, block_size, 576) -> (S, 1, 576)
        S = kv_c_and_k_pe_cache.shape[0] * kv_c_and_k_pe_cache.shape[1]
        kv = kv_c_and_k_pe_cache.reshape(S, 1, self.head_size)  # [S, 1, 576]

        # 4) Reshape indices for MQA: (T, 2048) -> (T, 1, 2048)
        indices = global_idx.unsqueeze(1)  # [T, 1, 2048]

        # 5) Call FlashMLA BF16 sparse prefill kernel
        #    NOTE: Unlike FlashMLASparseImpl._bf16_flash_mla_kernel which does
        #    not pass topk_length (it relies on all indices being valid), we
        #    pass topk_length=valid_counts because our indices have been
        #    clamped (global_idx.clamp_(min=0)), so entries beyond valid_counts
        #    are 0 (valid but irrelevant data). topk_length prevents the kernel
        #    from processing these clamped entries, saving compute and ensuring
        #    correctness.
        output = flash_mla_sparse_fwd(
            q_concat,  # [T, padded_heads, 576]
            kv,  # [S, 1, 576]
            indices,  # [T, 1, 2048]
            self.softmax_scale,  # 192**-0.5
            d_v=self.kv_lora_rank,  # 512
            topk_length=cache_seqlens,  # [T] valid entry counts
        )[0]  # extract output tensor from (output, max_logits, lse) tuple

        # 6) Unpad heads: (T, padded_heads, 512) -> (T, N, 512)
        return output[:, :N, :]

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """BF16 KV cache write using existing vLLM kernel.

        kv_cache_dtype MUST be "auto" which maps to Fp8KVCacheDataType::kAuto
        in the C++ dispatch, performing a direct BF16 copy (no quantization).
        Passing "bfloat16" would crash because concat_and_cache_mla expects
        the "auto" string for the non-quantized path.
        """
        if kv_cache.numel() == 0:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype="auto",
            scale=k_scale,
        )
