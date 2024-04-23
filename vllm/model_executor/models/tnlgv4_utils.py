import torch
import triton
import numpy as np
from functools import lru_cache


# helper functions for 3D sparse pattern
# these function are not optimized and very inefficient. Avoid calling them too frequent.
# currently, it is only called within `get_local_strided_sparse_attention_op`, which is cached.
def dense_to_crow_col(x):
    ''' Turning a 2D/3D torch tensor (x) to CSR rows/cols indexing.
    param:
    TODO:
        1. improve efficiency, is it faster if done in CPU, or customize a cuda kernel for it?
    NOTE: col_indices padded -1
    '''
    pad = -1
    dim = x.dim()
    assert x.dim() in (2, 3)
    if x.dim() == 2:
        x = x[None]
    x = [xi.to_sparse_csr() for xi in x]
    crows = torch.vstack([xi.crow_indices() for xi in x])
    cols = [xi.col_indices() for xi in x]
    max_cols = max(len(xi) for xi in cols)
    cols = [torch.cat([xi, pad + xi.new_zeros(max_cols - xi.shape[0])]) for xi in cols]
    cols = torch.vstack(cols)
    if dim == 2:
        crows = crows[0]
        cols = cols[0]
    return crows, cols


def crow_col_to_dense(crows, cols, dtype=torch.float16):
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
            x[i, j, cols[i, crows[i, j]:crows[i, j+1]]] = 1
    if dim == 1:
        x = x[0]
    return x.to(device)


def dense_to_ccol_row(x):
    '''Similar, but to CSC format
    '''
    x = x.transpose(-2, -1)
    return dense_to_crow_col(x)


def ccol_row_to_dense(ccol, rows, dtype=torch.float16):
    return crow_col_to_dense(ccol, rows, dtype).permute(0, 2, 1).contiguous()


def _get_sparse_attn_mask_homo_head(q_len, N_CTX, dtype, device, BLOCK=128, local_blocks=4, vert_stride=4, return_dense=False):
    '''
    :return: a tuple of 3:
        - tuple of crow_indices, col_indices representation of CSR format.
        - block dense mask
        - all token dense mask (be aware that it can be OOM if it is too big) if `return_dense==True`, otherwise, None
    '''
    with torch.no_grad():
        N_BLOCK = triton.cdiv(N_CTX, BLOCK)
        q_pos = torch.arange(N_BLOCK)[:, None]
        k_pos = torch.arange(N_BLOCK)[None]
        mask_vert_strided = (torch.arange(N_BLOCK) + 1) % vert_stride == 0
        block_mask_dense = ((q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided)).to(device).to(dtype)
        N_BLOCK_Q = triton.cdiv(q_len, BLOCK)
        block_mask_dense_output = block_mask_dense[-N_BLOCK_Q:].contiguous().to_sparse_csr()
    if return_dense:
        mask_dense = torch.kron(block_mask_dense, block_mask_dense.new_ones((BLOCK, BLOCK)))
        causal_mask = torch.tril(torch.ones(N_CTX, N_CTX)).type_as(mask_dense)[-q_len:]
        mask_dense = mask_dense[-q_len:, :N_CTX] * causal_mask
        return (block_mask_dense_output.crow_indices(), block_mask_dense_output.col_indices()), block_mask_dense, mask_dense
    else:
        return (block_mask_dense_output.crow_indices(), block_mask_dense_output.col_indices()), block_mask_dense, None


def binary_mask_to_bias(mask_dense):
    mask_dense = 1 - mask_dense
    mask_dense.masked_fill_(mask_dense.bool(), -torch.inf)
    return mask_dense


@lru_cache
def _get_sparse_attn_mask(n_heads, q_len, N_CTX, dtype, device, BLOCK=128, local_blocks=4, vert_stride=4, homo_head=True, return_dense=False, dense_mask_type='binary'):
    '''
    :param dense_mask_type: "binary" (0 for skip token, 1 for others) or "bias" (-inf for skip token, 0 or others)
    :return: a tuple of 3:
        - tuple of crow_indices, col_indices representation of CSR format.
        - block dense mask
        - all token dense mask (be aware that it can be OOM if it is too big) if `return_dense==True`, otherwise, None
    '''
    assert dense_mask_type in ('binary', 'bias')
    if homo_head:
        with torch.no_grad():
            (crow, col), block_mask_dense, mask_dense = _get_sparse_attn_mask_homo_head(q_len, N_CTX, dtype, device, BLOCK, local_blocks, vert_stride, return_dense)
            crow = crow[None].expand(n_heads, crow.shape[0])
            col = col[None].expand(n_heads, col.shape[0])
            if return_dense:
                mask_dense = mask_dense[None].expand(n_heads, *mask_dense.shape)
                if dense_mask_type == 'bias':
                    mask_dense = binary_mask_to_bias(mask_dense)
            return (crow, col), block_mask_dense, mask_dense

    with torch.no_grad():
        N_BLOCK = triton.cdiv(N_CTX, BLOCK)
        q_pos = torch.arange(N_BLOCK)[None, :, None]
        k_pos = torch.arange(N_BLOCK)[None, None]
        head_sliding_step = max(1, int(vert_stride / n_heads))  # if vert_stride <= n_heads, rotating the heads
        mask_vert_strided = [(torch.arange(N_BLOCK) + h * head_sliding_step + 1) % vert_stride == 0 for h in range(n_heads)]
        mask_vert_strided = torch.vstack(mask_vert_strided).unsqueeze(1)
        block_mask_dense = ((q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided)).to(device).to(dtype)
        N_BLOCK_Q = triton.cdiv(q_len, BLOCK)
        block_mask_dense_output = block_mask_dense[:, -N_BLOCK_Q:]
    if return_dense:
        mask_dense = torch.kron(block_mask_dense, block_mask_dense.new_ones((BLOCK, BLOCK)))
        causal_mask = torch.tril(torch.ones(N_CTX, N_CTX)).type_as(mask_dense)[-q_len:]
        mask_dense = mask_dense[..., -q_len:, :N_CTX] * causal_mask[None]
        if dense_mask_type == 'bias':
            mask_dense = binary_mask_to_bias(mask_dense)

        return dense_to_crow_col(block_mask_dense_output), block_mask_dense, mask_dense
    else:
        return dense_to_crow_col(block_mask_dense_output), block_mask_dense, None


def get_sparse_attn_mask(q, N_CTX, *args, **kwargs):
    return _get_sparse_attn_mask(q.size(1), q.size(2), N_CTX, q.dtype, q.device, *args, **kwargs)
