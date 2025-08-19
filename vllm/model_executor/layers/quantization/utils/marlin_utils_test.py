# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions used for tests and benchmarks"""

from typing import Optional

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.scalar_type import ScalarType, scalar_types

from .marlin_utils import (GPTQ_MARLIN_TILE, marlin_permute_scales,
                           marlin_zero_points)
from .quant_utils import (get_pack_factor, gptq_quantize_weights,
                          quantize_weights, sort_weights)


class MarlinWorkspace:

    def __init__(self, out_features, min_thread_n, max_parallel):
        assert (out_features % min_thread_n == 0), (
            "out_features = {} is undivisible by min_thread_n = {}".format(
                out_features, min_thread_n))

        max_workspace_size = ((out_features // min_thread_n) * max_parallel)

        self.scratch = torch.zeros(max_workspace_size,
                                   dtype=torch.int,
                                   device="cuda")


def marlin_permute_weights(q_w,
                           size_k,
                           size_n,
                           perm,
                           tile=GPTQ_MARLIN_TILE,
                           is_a_8bit=False):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    if is_a_8bit:
        q_w = q_w.reshape(
            (size_k // (tile * 2), tile * 2, size_n // tile, tile))
    else:
        q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w


def marlin_weights(q_w, size_k, size_n, num_bits, perm, is_a_8bit):
    # Permute
    q_w = marlin_permute_weights(q_w,
                                 size_k,
                                 size_n,
                                 perm,
                                 is_a_8bit=is_a_8bit)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor),
                        dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_packed


def get_weight_perm(num_bits: int, is_a_8bit: bool):
    perm_list: list[int] = []
    if is_a_8bit:
        perm_list: list[int] = []
        for i in range(32):
            perm1: list[int] = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                        4 * (i % 4),
                        4 * (i % 4) + 1,
                        4 * (i % 4) + 2,
                        4 * (i % 4) + 3,
                        4 * (i % 4 + 4),
                        4 * (i % 4 + 4) + 1,
                        4 * (i % 4 + 4) + 2,
                        4 * (i % 4 + 4) + 3,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(2):
                perm_list.extend([p + 512 * j for p in perm1])
    else:
        for i in range(32):
            perm1: list[int] = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                        2 * (i % 4),
                        2 * (i % 4) + 1,
                        2 * (i % 4 + 4),
                        2 * (i % 4 + 4) + 1,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)

    if num_bits == 4:
        if is_a_8bit:
            interleave = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        else:
            interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        if is_a_8bit:
            interleave = np.array([0, 1, 2, 3])
        else:
            interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


def marlin_quantize(w: torch.Tensor,
                    quant_type: ScalarType,
                    group_size: int,
                    act_order: bool,
                    test_perm: Optional[torch.Tensor] = None,
                    is_a_8bit: bool = False):

    if quant_type == scalar_types.uint8b128:
        assert not is_a_8bit, "UINT8B128 is only supported when is_a_8bit = false"
    if quant_type == scalar_types.int8:
        assert is_a_8bit, "INT8 is only supported when is_a_8bit = true"
        # we still use uint8b128 to quantize weight
        # and then convert the weight to int8
        quant_type = scalar_types.uint8b128

    size_k, size_n = w.shape
    num_bits = quant_type.size_bits

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = gptq_quantize_weights(
        w, quant_type, group_size, act_order, test_perm)

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    # Reformat to marlin
    weight_perm = get_weight_perm(num_bits, is_a_8bit)
    marlin_q_w = marlin_weights(q_w,
                                size_k,
                                size_n,
                                num_bits,
                                weight_perm,
                                is_a_8bit=is_a_8bit)
    marlin_s = marlin_permute_scales(s,
                                     size_k,
                                     size_n,
                                     group_size,
                                     is_a_8bit=is_a_8bit)

    if is_a_8bit and quant_type == scalar_types.uint8b128:
        # uint8b128 -> int8
        marlin_q_w = marlin_q_w.view(torch.int8) - 128
        marlin_q_w = marlin_q_w.view(torch.int32)

    if is_a_8bit and quant_type == scalar_types.uint4b8:
        # to fit the dequantizition method of GPTQ-W4A8
        marlin_q_w0 = (marlin_q_w & 0x0F0F0F0F | 0x80808080) - 0x08080808
        marlin_q_w0 = marlin_q_w0 & 0x0F0F0F0F

        marlin_q_w1 = (marlin_q_w & 0xF0F0F0F0 | 0x08080808) - 0x80808080
        marlin_q_w1 = marlin_q_w1 & 0xF0F0F0F0

        marlin_q_w = marlin_q_w0 | marlin_q_w1

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, rand_perm]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list


def awq_marlin_quantize(w: torch.Tensor,
                        quant_type: ScalarType,
                        group_size: int,
                        is_a_8bit: bool = False):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Detect num groups
    assert size_k % group_size == 0
    num_groups = size_k // group_size

    # Quantize with zp
    w_ref, q_w, s, zp = quantize_weights(w,
                                         quant_type,
                                         group_size,
                                         zero_points=True)

    # Reformat to marlin
    weight_perm = get_weight_perm(quant_type.size_bits, is_a_8bit)

    num_bits = 4
    q_w = q_w.T.contiguous()
    if num_bits == 4:
        q_w = q_w[:, ::2] + q_w[:, 1::2] * 16
    q_w = q_w.to(torch.int8).view(torch.int32).T.contiguous()

    marlin_q_w = ops.gptq_marlin_repack(
        q_w, torch.empty(0, dtype=torch.int, device=w.device), size_k, size_n,
        num_bits, is_a_8bit)

    # marlin_q_w = marlin_weights(q_w, size_k, size_n, quant_type.size_bits,
    #                             weight_perm, is_a_8bit=is_a_8bit)
    marlin_s = marlin_permute_scales(s,
                                     size_k,
                                     size_n,
                                     group_size,
                                     is_a_8bit=is_a_8bit)
    marlin_zp = marlin_zero_points(zp,
                                   num_groups,
                                   size_n,
                                   quant_type.size_bits,
                                   is_a_8bit=is_a_8bit)

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, marlin_zp]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list
