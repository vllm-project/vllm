# SPDX-License-Identifier: Apache-2.0

# Helper functions for 3D sparse pattern
# These function are not optimized and very inefficient.
# Avoid calling them too frequent or use a cache mechanism.

from functools import lru_cache

import numpy as np
import torch
import triton


class csr_matrix:
    """Simple implementation of CSR matrix conversion without scipy.
    This replaced scipy.sparse.csr_matrix() previously used."""

    def __init__(self, input_array):
        if not isinstance(input_array, np.ndarray):
            raise ValueError("Input must be a NumPy array")

        self.shape = input_array.shape
        rows, cols = self.shape
        data = []
        indices = []
        indptr = [0]

        for i in range(rows):
            for j in range(cols):
                if input_array[i, j]:
                    data.append(input_array[i, j])
                    indices.append(j)
            indptr.append(len(indices))

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)


def dense_to_crow_col(x: torch.Tensor):
    """Turning a 2D/3D torch tensor (x) to CSR rows/cols indexing.
    NOTE: col_indices padded -1
    """
    device = x.device
    pad = -1
    dim = x.dim()
    assert x.dim() in (2, 3)
    if x.dim() == 2:
        x = x[None]
    x = [csr_matrix(xi.bool().cpu().numpy()) for xi in x]
    crows = torch.vstack([torch.from_numpy(xi.indptr) for xi in x])
    cols = [torch.from_numpy(xi.indices) for xi in x]
    max_cols = max(len(xi) for xi in cols)
    cols = [
        torch.cat([xi, pad + xi.new_zeros(max_cols - xi.shape[0])])
        for xi in cols
    ]
    cols = torch.vstack(cols)
    if dim == 2:
        crows = crows[0]
        cols = cols[0]
    return crows.to(device), cols.to(device)


def crow_col_to_dense(crows: torch.Tensor,
                      cols: torch.Tensor,
                      dtype: torch.dtype = torch.float16):
    dim = crows.dim()
    if dim == 1:
        crows = crows[None]
        cols = cols[None]
    device = crows.device
    crows, cols = crows.cpu(), cols.cpu()  # faster in cpu
    shape = (crows.shape[0], crows.shape[1] - 1, cols.max() + 1)
    x = torch.zeros(shape, dtype=dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x[i, j, cols[i, crows[i, j]:crows[i, j + 1]]] = 1
    if dim == 1:
        x = x[0]
    return x.to(device)


def dense_to_ccol_row(x: torch.Tensor):
    """Similar, but to CSC format"""
    x = x.transpose(-2, -1)
    return dense_to_crow_col(x)


def ccol_row_to_dense(ccol: torch.Tensor,
                      rows: torch.Tensor,
                      dtype: torch.dtype = torch.float16):
    return crow_col_to_dense(ccol, rows, dtype).permute(0, 2, 1).contiguous()


def _get_sparse_attn_mask_homo_head(
    q_len: int,
    max_seqlen: int,
    dtype: torch.dtype,
    device: torch.device,
    block_size: int = 128,
    local_blocks: int = 4,
    vert_stride: int = 4,
    return_dense: bool = False,
):
    """
    :return: a tuple of 3:
        - tuple of crow_indices, col_indices representation
            of CSR format.
        - block dense mask
        - all token dense mask (be aware that it can be
            OOM if it is too big) if `return_dense==True`,
            otherwise, None
    """
    with torch.no_grad():
        num_blocks = triton.cdiv(max_seqlen, block_size)
        q_pos = torch.arange(num_blocks)[:, None]
        k_pos = torch.arange(num_blocks)[None]
        mask_vert_strided = (torch.arange(num_blocks) + 1) % vert_stride == 0
        block_mask_dense = (((q_pos >= k_pos)
                             & ((q_pos - k_pos < local_blocks)
                                | mask_vert_strided)).to(device).to(dtype))
        num_blocks_q = triton.cdiv(q_len, block_size)
        block_mask_dense_output = (dense_to_crow_col(
            block_mask_dense[-num_blocks_q:].contiguous()))
    if return_dense:
        mask_dense = torch.kron(
            block_mask_dense,
            block_mask_dense.new_ones((block_size, block_size)),
        )
        causal_mask = torch.tril(torch.ones(
            max_seqlen, max_seqlen)).type_as(mask_dense)[-q_len:]
        mask_dense = mask_dense[-q_len:, :max_seqlen] * causal_mask
        return (
            block_mask_dense_output,
            block_mask_dense,
            mask_dense,
        )
    else:
        return (
            block_mask_dense_output,
            block_mask_dense,
            None,
        )


def binary_mask_to_bias(mask_dense: torch.Tensor):
    mask_dense = 1 - mask_dense
    mask_dense.masked_fill_(mask_dense.bool(), -torch.inf)
    return mask_dense


def get_head_sliding_step(n_heads: int,
                          vert_stride: int,
                          homo_head: bool = False):
    if homo_head:
        return 0
    return max(1, int(vert_stride / n_heads))


@lru_cache
def get_sparse_attn_mask(
    n_heads: int,
    q_len: int,
    max_seqlen: int,
    dtype: torch.dtype,
    device: torch.device,
    block_size: int = 64,
    local_blocks: int = 4,
    vert_stride: int = 4,
    homo_head: bool = True,
    return_dense: bool = False,
    dense_mask_type: str = "binary",
):
    """
    :param dense_mask_type: "binary" (0 for skip token, 1 for others)
        or "bias" (-inf for skip token, 0 or others)
    :return: a tuple of 3:
        - tuple of crow_indices, col_indices representation
            of CSR format.
        - block dense mask
        - all token dense mask (be aware that it can be OOM if it
            is too big) if `return_dense==True`, otherwise, None
    """
    assert dense_mask_type in ("binary", "bias")
    if homo_head:
        with torch.no_grad():
            (crow, col), block_mask_dense, mask_dense = (
                _get_sparse_attn_mask_homo_head(
                    q_len,
                    max_seqlen,
                    dtype,
                    device,
                    block_size,
                    local_blocks,
                    vert_stride,
                    return_dense,
                ))
            crow = crow[None].expand(n_heads, crow.shape[0])
            col = col[None].expand(n_heads, col.shape[0])
            if return_dense:
                mask_dense = mask_dense[None].expand(n_heads,
                                                     *mask_dense.shape)
                if dense_mask_type == "bias":
                    mask_dense = binary_mask_to_bias(mask_dense)
            return (crow, col), block_mask_dense, mask_dense

    with torch.no_grad():
        num_blocks = triton.cdiv(max_seqlen, block_size)
        q_pos = torch.arange(num_blocks)[None, :, None]
        k_pos = torch.arange(num_blocks)[None, None]
        head_sliding_step = get_head_sliding_step(n_heads, vert_stride)
        mask_vert_strided = [
            (torch.arange(num_blocks) + h * head_sliding_step + 1) %
            vert_stride == 0 for h in range(n_heads)
        ]
        mask_vert_strided = torch.vstack(mask_vert_strided).unsqueeze(1)
        block_mask_dense = (((q_pos >= k_pos)
                             & ((q_pos - k_pos < local_blocks)
                                | mask_vert_strided)).to(device).to(dtype))
        num_blocks_q = triton.cdiv(q_len, block_size)
        block_mask_dense_output = block_mask_dense[:, -num_blocks_q:]
    if return_dense:
        mask_dense = torch.kron(
            block_mask_dense,
            block_mask_dense.new_ones((block_size, block_size)),
        )
        causal_mask = torch.tril(torch.ones(
            max_seqlen, max_seqlen)).type_as(mask_dense)[-q_len:]
        mask_dense = mask_dense[..., -q_len:, :max_seqlen] * causal_mask[None]
        if dense_mask_type == "bias":
            mask_dense = binary_mask_to_bias(mask_dense)

        return (
            dense_to_crow_col(block_mask_dense_output),
            block_mask_dense,
            mask_dense,
        )
    else:
        return (
            dense_to_crow_col(block_mask_dense_output),
            block_mask_dense,
            None,
        )
