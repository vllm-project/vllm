# SPDX-License-Identifier: Apache-2.0
"""Triton-based block-sparse attention backend for LMCache CacheBlend.

Drop-in replacement for LMCFlashInferSparseBackend that works on ROCm.
Uses Triton kernels instead of flashinfer for block-sparse attention.
"""

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.abstract import AttentionInterface
from lmcache.v1.compute.attention.metadata import (
    LMCAttnMetadata,
    LMCTritonSparseMetadata,
)
from lmcache.v1.compute.attention.triton_kernels import (
    block_sparse_attention,
    causal_prefill_attention,
)

logger = init_logger(__name__)


class LMCTritonSparseBackend(AttentionInterface):
    """Triton-based block-sparse attention backend for LMCache.

    Drop-in replacement for LMCFlashInferSparseBackend.
    Works on both CUDA (via Triton) and ROCm (via Triton).
    Does NOT require flashinfer.
    """

    def __init__(self, vllm_attn: "torch.nn.Module") -> None:
        """Initialize from a vLLM Attention module.

        Args:
            vllm_attn: vLLM Attention layer (for extracting config)
        """
        self.vllm_attn = vllm_attn
        impl = vllm_attn.impl

        self.num_qo_heads = impl.num_heads
        self.num_kv_heads = impl.num_kv_heads
        self.head_dim = impl.head_size
        self.sm_scale = impl.scale

        # Optional scale factors for FP8 KV cache
        self.k_scale = getattr(impl, "k_scale", None)
        self.v_scale = getattr(impl, "v_scale", None)

        idx = torch_dev.current_device()
        self.device = torch.device(f"{torch_device_type}:{idx}")

        logger.info(
            f"Initialized LMCTritonSparseBackend: "
            f"heads={self.num_qo_heads}, kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, scale={self.sm_scale:.4f}"
        )

    def forward_contiguous(
        self,
        query: torch.Tensor,  # [M, H, D]
        key: torch.Tensor,  # [N, H, D]
        value: torch.Tensor,  # [N, H, D]
        output: torch.Tensor,  # [M, H, D]
        attn_metadata: LMCAttnMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Compute attention using Triton kernels.

        For causal (full prefill): uses causal_prefill_attention.
        For non-causal (sparse CacheBlend): uses block_sparse_attention.
        """
        assert isinstance(attn_metadata, LMCTritonSparseMetadata)

        if attn_metadata.is_causal:
            output = causal_prefill_attention(
                query,
                key,
                value,
                sm_scale=self.sm_scale,
                block_size=64,
            )
        else:
            assert attn_metadata.block_indices is not None, (
                "Sparse metadata not initialized. Call update_from_top_indices() first."
            )
            output, lse = block_sparse_attention(
                query,
                key,
                value,
                block_indices=attn_metadata.block_indices,
                block_indptr=attn_metadata.block_indptr,
                sm_scale=self.sm_scale,
                # block_size controls BLOCK_M and BLOCK_N in the kernel.
                # Use sparse_blk_row_size to match the CSR row structure
                # built by update_from_top_indices.
                # Note: sparse_blk_row_size == sparse_blk_col_size (both 32)
                # in the current implementation.
                block_size=attn_metadata.sparse_blk_row_size,
            )
            # Store LSE for downstream blending if needed
            attn_metadata.lse = lse

        return output

    def init_attn_metadata(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> LMCTritonSparseMetadata:
        """Initialize attention metadata.

        Creates a TritonSparseMetadata with default causal mode.
        CacheBlend will call update_from_top_indices() to switch to sparse.
        """
        seq_len = len(input_ids)

        sparse_blk_row_size = 32
        sparse_blk_col_size = 32

        num_block_col = (seq_len + sparse_blk_col_size - 1) // sparse_blk_col_size
        block_col_sizes = torch.full(
            (num_block_col,),
            sparse_blk_col_size,
            dtype=torch.int32,
            device=self.device,
        )
        # Last block may be smaller
        remainder = seq_len % sparse_blk_col_size
        if remainder > 0:
            block_col_sizes[-1] = remainder

        return LMCTritonSparseMetadata(
            seq_len=seq_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            sparse_blk_row_size=sparse_blk_row_size,
            sparse_blk_col_size=sparse_blk_col_size,
            is_causal=True,
            block_col_sizes=block_col_sizes,
        )
