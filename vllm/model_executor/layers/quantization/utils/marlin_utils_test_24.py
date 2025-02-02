# SPDX-License-Identifier: Apache-2.0
"""Utility functions used for tests and benchmarks"""

import random
from typing import List

import numpy
import torch

from vllm.scalar_type import ScalarType

from .marlin_utils_test import marlin_weights
from .quant_utils import gptq_quantize_weights


# This is PyTorch implementation of main part of reorder_meta()
# function, from tools/util/include/cutlass/util/host_reorder.h file
# of CUTLASS source tree.  Furthermore, CUTLASS template for sparse
# GEMM decides upon layout of this matrix, and at the moment for the
# sparse GEMM executed on tensor cores, this is layout described by
# ColumnMajorInterleaved<2> data structure, in
# include/cutlass/layout/matrix.h of CUTLASS source tree.  The
# reordering of meta matrix into meta_reordered matrix calculated
# according to these segments of CUTLASS code is re-implemented here.
# Note that this calculation produces offsets for scattering metadata
# matrix elements into reordered metadata matrix elements (or,
# equivalently, for gathering reordered metadata matrix element back
# into metadata matrix elements).
def _calculate_meta_reordering_scatter_offsets(m, meta_ncols, meta_dtype,
                                               device):
    dst_rows = torch.arange(0, m, device=device)[:, None].repeat(1, meta_ncols)
    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(m, 1)

    # Reorder the rows, then swizzle the 2x2 blocks.
    group_x = 64
    group_y = 32 if meta_dtype.itemsize == 2 else 16

    dst_rows = (dst_rows // group_x * group_x + (dst_rows % 2) * 2 +
                (dst_rows % 8) // 4 + ((dst_rows % group_y) % 4) // 2 * 32 +
                ((dst_rows % group_x) // 8) * 4)

    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    dst_rows += topright - bottomleft
    dst_cols -= topright - bottomleft

    # Assumed that meta tensor is to be stored in CUTLASS
    # InterleavedColumnMajor layout, and reverse engineered
    # corresponding code to store values into this tensor.
    interleave = 2
    cols_maj = dst_cols // interleave
    cols_min = dst_cols % interleave
    return (cols_maj * m * interleave + dst_rows * interleave +
            cols_min).view(-1)


# This function converts dense matrix into sparse semi-structured
# representation, producing "compressed" matrix, in the layout used by
# CUTLASS backend, and corresponding metadata matrix.
def sparse_semi_structured_from_dense_cutlass(dense):
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"  # noqa: E501
        )

    m, k = dense.shape
    device = dense.device

    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16, torch.float, torch.int32]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError(
            "Invalid number of elements per meta element calculated")

    if meta_dtype == torch.int32:
        if m % 16 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 16")
    else:
        if m % 32 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 32")
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}"  # noqa: E501
        )

    if dense.dtype != torch.float:
        ksparse = 4
        dense_4 = dense.view(-1, k // ksparse, ksparse)
        m0, m1, m2, m3 = (dense_4 != 0).unbind(-1)
    else:
        ksparse = 2
        dense_2 = dense.view(-1, k // ksparse, ksparse)
        m0, m2 = m1, m3 = (dense_2 != 0).unbind(-1)
    meta_ncols = k // (ksparse * quadbits_per_meta_elem)

    # Encoding quadruples of True/False values as follows:
    #     [True,  True,  False, False] -> 0b0100
    #     [True,  False, True,  False] -> 0b1000
    #     [False, True,  True,  False] -> 0b1001
    #     [True,  False, False, True ] -> 0b1100
    #     [False, True,  False, True ] -> 0b1101
    #     [False, False, True,  True ] -> 0b1110
    # Thus, lower two bits in the encoding are index of the True value
    # at the lowest index in the quadruple, and the higher two bits in
    # the encoding are index of the other True value in the quadruple.
    # In case there are less than two True values, than False value or
    # values at some index or indices are considered True for the
    # encoding.  In case there are more than two True values, then the
    # excess True value(s) at some indices are considered False for
    # the encoding.  The exact encodings used for these cases are as
    # follows:
    #     [False, False, False, False] -> 0b1110
    #     [False, False, False, True ] -> 0b1110
    #     [False, False, True,  False] -> 0b1110
    #     [False, True,  False, False] -> 0b1001
    #     [False, True,  True,  True ] -> 0b1101
    #     [True,  False, False, False] -> 0b1000
    #     [True,  False, True,  True ] -> 0b1100
    #     [True,  True,  False, True ] -> 0b0100
    #     [True,  True,  True,  False] -> 0b0100
    #     [True,  True,  True,  True ] -> 0b0100
    # These particular encodings are chosen, with the help of Espresso
    # logic minimizer software, for the purpose of minimization of
    # corresponding Boolean functions, that translate non-zero flags
    # into encoding bits.  Note also possible choices for the first
    # and last of these encodings were limited only to (0b0100,
    # 0b1110), in order to produce valid encodings for 1:2 sparsity
    # case.

    expr0 = m0 & m1
    expr1 = ~m0 & m1
    expr2 = ~m0 & ~m1
    bit0 = expr1
    bit1 = expr2
    bit2 = expr0 | expr2 | m3
    bit3 = expr1 | ~m1
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)

    if dense.dtype != torch.float:
        sparse0 = dense_4.gather(
            -1, idxs0.unsqueeze(-1))  # type: ignore[possibly-undefined]
        sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
        sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    else:
        sparse = dense_2.gather(-1,
                                idxs0.unsqueeze(-1) // 2).view(
                                    m,
                                    k // 2)  # type: ignore[possibly-undefined]

    meta_4 = idxs0 | (idxs1 << 2)
    meta_n = meta_4.view(
        (-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)

    if quadbits_per_meta_elem == 4:
        meta = (meta_n[:, :, 0]
                | (meta_n[:, :, 1] << 4)
                | (meta_n[:, :, 2] << 8)
                | (meta_n[:, :, 3] << 12))
    elif quadbits_per_meta_elem == 8:
        meta = (meta_n[:, :, 0]
                | (meta_n[:, :, 1] << 4)
                | (meta_n[:, :, 2] << 8)
                | (meta_n[:, :, 3] << 12)
                | (meta_n[:, :, 4] << 16)
                | (meta_n[:, :, 5] << 20)
                | (meta_n[:, :, 6] << 24)
                | (meta_n[:, :, 7] << 28))

    # Reorder meta tensor elements.
    meta_reordered = meta.new_empty(
        (m * meta_ncols, ))  # type: ignore[possibly-undefined]
    meta_offsets = _calculate_meta_reordering_scatter_offsets(
        m, meta_ncols, meta_dtype, device)
    meta_reordered.scatter_(0, meta_offsets, meta.view(-1))

    return (sparse, meta_reordered.view(m, meta_ncols))


# This function performs reverse of the function above - it
# reconstructs dense matrix from a pair of "compressed" matrix, given
# in the layout used by CUTLASS backend, and accompanying metadata
# matrix.
def sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered):
    if sparse.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor"  # noqa: E501
        )

    m, k = sparse.shape
    device = sparse.device

    if meta_reordered.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor"  # noqa: E501
        )
    if meta_reordered.device != device:
        raise RuntimeError(
            f"Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device"  # noqa: E501
        )

    meta_dtype = meta_reordered.dtype
    if meta_dtype not in (torch.int16, torch.int32):
        raise RuntimeError(f"Invalid datatype {meta_dtype} of meta matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4

    ksparse = 4 if sparse.dtype != torch.float else 2

    meta_nrows, meta_ncols = meta_reordered.shape
    if meta_nrows != m:
        raise RuntimeError(
            f"Number of rows of meta matrix {meta_nrows} must be equal to number of columns of spase matrix {m}"  # noqa: E501
        )
    if meta_ncols * ksparse * quadbits_per_meta_elem != 2 * k:
        raise RuntimeError(
            f"Number of columns of sparse matrix {k} different from the {meta_ncols * ksparse * quadbits_per_meta_elem // 2}, "  # noqa: E501
            "expected according to the number of columns of meta matrix")

    # Undo meta tensor elements reordering.
    meta_offsets = _calculate_meta_reordering_scatter_offsets(
        m, meta_ncols, meta_dtype, device)
    meta = torch.gather(meta_reordered.view(-1), 0,
                        meta_offsets).view(m, meta_ncols)

    # Unpack sparse tensor back to original dense tensor, using
    # information provided by meta tensor.  Note that torch.float
    # datatype is handled pretty much the same as
    # torch.half/torch.bfloat16, as metadata for a pair of torch.float
    # value is encoded as if underlying 8 bytes contain four
    # torch.half/torch.bfloat16 values, where either first two or last
    # two are zeros.
    meta_2 = torch.empty(
        (m, meta_ncols, 2 * quadbits_per_meta_elem),
        dtype=meta_dtype,
        device=device,
    )
    if quadbits_per_meta_elem == 4:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
    elif quadbits_per_meta_elem == 8:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
        meta_2[:, :, 8] = (meta >> 16) & 0b11
        meta_2[:, :, 9] = (meta >> 18) & 0b11
        meta_2[:, :, 10] = (meta >> 20) & 0b11
        meta_2[:, :, 11] = (meta >> 22) & 0b11
        meta_2[:, :, 12] = (meta >> 24) & 0b11
        meta_2[:, :, 13] = (meta >> 26) & 0b11
        meta_2[:, :, 14] = (meta >> 28) & 0b11
        meta_2[:, :, 15] = (meta >> 30) & 0b11

    dense_offsets = meta_2.view(-1) + (
        torch.arange(0, 2 * m * k // ksparse, device=device) * 4).view(
            -1, 1).repeat(1, 2).view(-1)

    dense = torch.zeros((m * 2 * k, ), dtype=sparse.dtype, device=device)
    if sparse.dtype != torch.float:
        # dense.scatter_(0, dense_offsets, sparse.view(-1))
        dense.scatter_(0, dense_offsets, sparse.reshape(-1))
    else:
        dense.view(torch.half).scatter_(0, dense_offsets,
                                        sparse.view(torch.half).view(-1))

    return dense.view(m, 2 * k)


def mask_creator(tensor):
    """
    Class for creating N:M sparsity masks.
    Masks will be created using the N:M ratio, where for every block of 
    M weights, N will be pruned based on ranked weight value. Each mask 
    will correspond to the given tensor.

    :param N: The number of weights in a group to keep
    :param M: The size of a weight group
    """
    N = 2
    M = 4

    mask = None
    # for i, tensor in enumerate(tensors):
    if tensor.numel() % M != 0:
        raise ValueError(
            f"Tensor of size {tensor.shape} can't be evenly divided into "
            f"{M} groups")

    num_groups = tensor.numel() // M

    # N:M sparsity for linear layers
    tensor_temp = tensor.detach().abs().reshape(num_groups, M)
    index = torch.argsort(tensor_temp, dim=1)[:, :int(M - N)]

    w_b = torch.ones(tensor_temp.shape, device=tensor_temp.device)
    mask = w_b.scatter_(dim=1, index=index, value=0).reshape(tensor.shape)

    return mask


def inject_24(w, size_k, size_n):
    assert w.shape == (size_k, size_n)

    mask = mask_creator(w.t()).t().cuda().bool()

    return (mask * w).contiguous(), mask.contiguous()


def check_24(w, num_rows_to_sample=50, _verbose=False):
    BLOCK_SIZE = 4
    MAX_NON_ZEROS = 2

    w = w.t().contiguous()

    print("check_24: w.shape = {}".format(w.shape))

    num_rows, num_cols = w.shape
    sampled_row_idxs = random.choices(range(num_rows), k=num_rows_to_sample)
    if _verbose:
        print(f"Sampled row idxs = {sampled_row_idxs}")

    total_segments = 0
    non_24_segments = 0
    for i in sampled_row_idxs:
        for j in range(0, num_cols - BLOCK_SIZE, BLOCK_SIZE):
            total_segments += 1
            block = w[i, j:j + BLOCK_SIZE]
            num_nonzero = torch.count_nonzero(block)
            if num_nonzero > MAX_NON_ZEROS:
                print("i = {} j = {} block = {}".format(i, j, block))
                non_24_segments += 1

    print(f"{non_24_segments} / {total_segments} do not have 2:4 structure.")


def compress_quantized_24_weight(q_24, size_k, size_n, wtype: ScalarType):
    assert q_24.shape == (size_k, size_n)

    # Remove bias to normalize over 0
    q_24_no_zp = q_24 - wtype.bias

    # Compress
    q_24_no_zp = q_24_no_zp.t().contiguous()
    q_24_no_zp_comp, meta = sparse_semi_structured_from_dense_cutlass(
        q_24_no_zp)
    q_24_no_zp_comp = q_24_no_zp_comp.t().contiguous()

    # Restore bias
    q_24_comp = q_24_no_zp_comp + wtype.bias

    # Resize meta to its actual shape (without moving any data)
    meta = meta.resize_(meta.shape[1] // 2, meta.shape[0] * 2)

    return q_24_comp, meta


def get_scale_perms_24():
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i * 8 + j for j in [0, 4, 1, 5, 2, 6, 3, 7]])
    scale_perm_single: List[int] = []
    for i in range(8):
        scale_perm_single.extend([8 * i + j for j in [0, 1, 2, 3, 4, 5, 6, 7]])
    return scale_perm, scale_perm_single


def get_weight_perm_24(num_bits: int):
    perm_list: List[int] = []
    for i in range(32):
        perm1: List[int] = []
        col = i // 4
        col_o = col // 2
        for block in [0, 1]:
            for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col_o * 256 + 8 * (col % 2) +
                             4 * block)
        for j in range(4):
            perm_list.extend([p + 1 * j for p in perm1])
    perm = numpy.array(perm_list)

    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise ValueError("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


def marlin_permute_scales_24(s: torch.Tensor, size_k: int, size_n: int,
                             group_size: int) -> torch.Tensor:

    scale_perm, scale_perm_single = get_scale_perms_24()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def marlin_24_quantize(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Inject 2:4 sparsity
    w_24, mask_24 = inject_24(w, size_k, size_n)

    # Quantize
    w_24_ref, q_w_24, s, g_idx, rand_perm = gptq_quantize_weights(
        w_24, quant_type, group_size, act_order=False)

    # Compress quantized weight
    q_w_24_comp, meta = compress_quantized_24_weight(q_w_24, size_k, size_n,
                                                     quant_type)
    size_k_comp = size_k // 2

    # Reformat to marlin
    weight_perm = get_weight_perm_24(quant_type.size_bits)
    marlin_24_q_w_comp = marlin_weights(q_w_24_comp, size_k_comp, size_n,
                                        quant_type.size_bits, weight_perm)
    marlin_24_s = marlin_permute_scales_24(s, size_k, size_n, group_size)

    # Create result
    res_list = [w_24_ref, marlin_24_q_w_comp, meta, marlin_24_s]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list
