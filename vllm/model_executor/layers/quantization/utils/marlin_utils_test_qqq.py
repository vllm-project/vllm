# SPDX-License-Identifier: Apache-2.0

import numpy
import torch

from .marlin_utils_test import marlin_permute_weights
from .quant_utils import get_pack_factor, qqq_quantize_weights


def marlin_qqq_weights(q_w, size_k, size_n, num_bits, perm, group_size):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_packed = numpy.zeros((q_w.shape[0], q_w.shape[1] // pack_factor),
                           dtype=numpy.uint32)
    if group_size == size_k:
        for i in range(pack_factor):
            q_packed |= (q_w[:, i::pack_factor] & 0xF) << num_bits * i
    else:
        for i in range(pack_factor):
            q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(numpy.int32)).to(orig_device)

    return q_packed


def get_qqq_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


# NOTE(HandH1998): QQQ employs different perms for per-group and per-channel weight quantization. # noqa: E501
def get_qqq_weight_perm(num_bits: int, quant_type: str):
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
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = numpy.array(perm_list)

    assert quant_type in ["per-channel",
                          "per-group"], "not supported quantization type"
    if num_bits == 4:
        if quant_type == "per-channel":
            interleave = numpy.array([4, 0, 5, 1, 6, 2, 7, 3])
        else:
            interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    else:
        raise Exception("num_bits must be 4, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


def marlin_qqq_permute_scales(s_group, s_channel, size_k, size_n, group_size):
    scale_perm, scale_perm_single = get_qqq_scale_perms()
    if group_size < size_k and group_size != -1:
        s_group = s_group.reshape((-1, len(scale_perm)))[:, scale_perm]
        s_channel = s_channel.reshape(
            (-1, len(scale_perm_single)))[:, scale_perm_single]
        s_group = s_group.reshape((-1, size_n)).contiguous()
    else:
        s_channel = s_channel.reshape(
            (-1, len(scale_perm_single)))[:, scale_perm_single]
    s_channel = s_channel.reshape((-1, size_n)).contiguous()

    return s_group, s_channel


def marlin_qqq_quantize(
    w: torch.Tensor,
    num_bits: int,
    group_size: int,
):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k
    quant_type = "per-channel" if group_size == size_k else "per-group"

    # Quantize
    w_ref, q_w, s_group, s_channel = qqq_quantize_weights(
        w, num_bits, group_size)

    # Reformat to marlin_qqq
    weight_perm = get_qqq_weight_perm(num_bits, quant_type)
    marlin_qqq_q_w = marlin_qqq_weights(q_w, size_k, size_n, num_bits,
                                        weight_perm, group_size)
    marlin_qqq_s_group, marlin_qqq_s_channel = marlin_qqq_permute_scales(
        s_group, s_channel, size_k, size_n, group_size)

    # Create result
    res_list = [
        w_ref, marlin_qqq_q_w, marlin_qqq_s_group, marlin_qqq_s_channel
    ]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list
