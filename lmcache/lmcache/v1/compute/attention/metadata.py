# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import abc

# Third Party
import torch

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.compute.attention.flash_infer_sparse import HackBSAWrapper


@dataclass
class LMCAttnMetadata(metaclass=abc.ABCMeta):
    @abstractmethod
    def update_from_top_indices(self, top_indices: torch.Tensor):
        raise NotImplementedError("This method should be implemented in subclasses.")


@dataclass
class LMCFlashAttnMetadata(LMCAttnMetadata):
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_query_len: torch.Tensor
    max_seq_len: torch.Tensor

    def update_from_top_indices(self, top_indices: torch.Tensor):
        top_k_num = len(top_indices)
        self.max_query_len = top_k_num
        device = self.query_start_loc.device
        dtype = self.query_start_loc.dtype
        self.query_start_loc = torch.tensor([0, top_k_num], dtype=dtype, device=device)


@dataclass
class LMCFlashInferSparseMetadata(LMCAttnMetadata):
    wrapper: "HackBSAWrapper"
    seq_len: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    block_col_sizes: torch.Tensor
    sparse_blk_row_size: int = 32  # TODO(Jiayi): make this tunable
    sparse_blk_col_size: int = 32  # TODO(Jiayi): make this tunable
    is_causal: bool = True
    q_data_dtype: torch.dtype = torch.bfloat16  # TODO(Jiayi): remove hardcode

    def update_from_top_indices(self, top_indices: torch.Tensor):
        # self.is_causal = False
        device = top_indices.device
        top_k_num = len(top_indices)
        num_block_row = top_k_num // self.sparse_blk_row_size
        block_row_sizes = torch.tensor(
            [self.sparse_blk_row_size] * num_block_row, device=device
        )
        block_row_sizes[-1] += top_k_num % self.sparse_blk_row_size

        block_mask_map = torch.zeros(
            num_block_row, len(self.block_col_sizes), dtype=torch.bool, device=device
        )
        cols = torch.arange(block_mask_map.size(1), device=device).expand(
            block_mask_map.size(0), -1
        )

        # NOTE(Jiayi): select every `sparse_blk_row_size`-th index from top_indices
        # to approximate the attention mask at block level.
        top_indices_slice = top_indices[
            self.sparse_blk_row_size - 1 :: self.sparse_blk_row_size
        ]
        top_indices_slice //= self.sparse_blk_col_size
        mask = cols < top_indices_slice.unsqueeze(1)
        block_mask_map[mask] = 1
        self.wrapper.plan(
            block_mask_map.expand(self.num_kv_heads, -1, -1),
            block_row_sizes.expand(self.num_kv_heads, -1),
            self.block_col_sizes.expand(self.num_kv_heads, -1),
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            q_data_type=self.q_data_dtype,
        )


def _block_mask_to_csr(
    block_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a 2D boolean block mask to CSR sparse format.

    Args:
        block_mask: [num_q_blocks, num_kv_blocks] bool tensor
        device: target device

    Returns:
        block_indices: [nnz] int32 column indices
        block_indptr: [num_q_blocks + 1] int32 row pointers
    """
    num_q_blocks = block_mask.shape[0]
    nnz_per_row = block_mask.sum(dim=1)
    block_indptr = torch.zeros(num_q_blocks + 1, dtype=torch.int32, device=device)
    torch.cumsum(nnz_per_row, dim=0, out=block_indptr[1:])
    # Vectorized CSR column index extraction — torch.where naturally
    # returns an empty tensor when the mask is all-False, so no early
    # return is needed (and we avoid a .item() CPU-GPU sync).
    block_indices = torch.where(block_mask)[1].to(torch.int32)
    return block_indices, block_indptr


@dataclass
class LMCTritonSparseMetadata(LMCAttnMetadata):
    """Metadata for Triton-based block-sparse attention.

    Drop-in replacement for LMCFlashInferSparseMetadata.
    Works on both CUDA and ROCm via Triton.
    """

    seq_len: int = 0
    num_qo_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    sparse_blk_row_size: int = 32
    sparse_blk_col_size: int = 32
    is_causal: bool = True

    # Sparse indices (populated by update_from_top_indices)
    block_indices: Optional[torch.Tensor] = None
    block_indptr: Optional[torch.Tensor] = None

    # Block column sizes for variable-size last block
    block_col_sizes: Optional[torch.Tensor] = None

    # LSE from sparse attention (for downstream merge if needed)
    lse: Optional[torch.Tensor] = None

    # Match flashinfer metadata interface
    q_data_dtype: torch.dtype = torch.bfloat16

    def update_from_top_indices(self, top_indices: torch.Tensor):
        """Convert top_indices to CSR sparse structure for Triton kernel.

        Called by CacheBlend when it determines which tokens have cache hits.
        """
        self.is_causal = False
        device = top_indices.device
        top_k_num = len(top_indices)

        # Use ceiling division so all query positions are covered.
        # The last block row may be partial (fewer than sparse_blk_row_size
        # queries) — the kernel handles this via q_mask.
        num_block_row = (
            top_k_num + self.sparse_blk_row_size - 1
        ) // self.sparse_blk_row_size
        num_block_col = (
            self.seq_len + self.sparse_blk_col_size - 1
        ) // self.sparse_blk_col_size

        block_mask = torch.zeros(
            num_block_row, num_block_col, dtype=torch.bool, device=device
        )

        top_indices_slice = top_indices[
            self.sparse_blk_row_size - 1 :: self.sparse_blk_row_size
        ]
        top_indices_block = top_indices_slice // self.sparse_blk_col_size

        cols = torch.arange(num_block_col, device=device).expand(
            len(top_indices_block), -1
        )
        mask = cols < top_indices_block.unsqueeze(1)
        block_mask[: len(top_indices_block)] = mask

        self.block_indices, self.block_indptr = _block_mask_to_csr(block_mask, device)
