# ruff: noqa
# Copyright (c) the vLLM team.
# Copyright (c) the PygmalionAI team.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This script was initially developed for sub-byte MX dtypes (FP4 E2M1, FP6 E3M2, and FP6 E2M3).
# It has been refactored to support any sub-byte FP dtypes. However, some behaviors of MX dtypes remain:
#   1. No encodings are reserved for special values (+/-inf, NaN).
#   2. When downcasting from FP32 to FPx,
#      - Rounding mode is round to nearest, ties to even.
#      - Values outside the representable range of FPx after rounding are clamped to the maximum FPx
#      magnitude (sign is preserved).
from functools import reduce
from typing import Tuple

import torch
from torch import Tensor


def _n_ones(n: int) -> int:
    return (1 << n) - 1


EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)

# https://github.com/microsoft/DeepSpeed/blob/3a3a6db3332e339cc9fd94efd4982f6d60635a3d/deepspeed/inference/v2/kernels/core_ops/cuda_linear/cuda_linear.py
_SPLIT_K_MAP = [
    {  # tokens: [1, 64]
        3072: 18,
        4096: 13,
        5120: 10,
        6144: 9,
        8192: 6,
        10240: 5,
        14336: 7,
        28672: 7,
        57344: 7
    },
    {  # tokens: [65:128]
        3072: 9,
        4096: 6,
        5120: 5,
        6144: 9,
        8192: 3,
        10240: 5,
        14336: 7,
        28672: 7,
        57344: 6
    },
    {  # tokens: [129:192]
        3072: 6,
        4096: 4,
        5120: 7,
        6144: 3,
        8192: 2,
        10240: 5,
        14336: 5,
        28672: 5,
        57344: 4
    },
    {  # tokens: [193:256]
        3072: 9,
        4096: 3,
        5120: 5,
        6144: 2,
        8192: 5,
        10240: 4,
        14336: 8,
        28672: 6,
        57344: 4
    },
    {  # tokens: [257:320]
        3072: 7,
        4096: 5,
        5120: 2,
        6144: 5,
        8192: 4,
        10240: 1,
        14336: 3,
        28672: 3,
        57344: 4
    },
    {  # tokens: [321:384]
        3072: 3,
        4096: 2,
        5120: 5,
        6144: 3,
        8192: 1,
        10240: 8,
        14336: 3,
        28672: 4,
        57344: 3
    },
    {  # tokens: [385:448]
        3072: 5,
        4096: 7,
        5120: 3,
        6144: 5,
        8192: 7,
        10240: 3,
        14336: 1,
        28672: 1,
        57344: 3
    },
    {  # tokens: [449:512]
        3072: 2,
        4096: 5,
        5120: 4,
        6144: 1,
        8192: 5,
        10240: 2,
        14336: 6,
        28672: 4,
        57344: 1
    },
    {  # tokens: [513:576]
        3072: 2,
        4096: 3,
        5120: 1,
        6144: 1,
        8192: 3,
        10240: 3,
        14336: 3,
        28672: 1,
        57344: 1
    },
    {  # tokens: [577:640]
        3072: 5,
        4096: 4,
        5120: 1,
        6144: 4,
        8192: 2,
        10240: 1,
        14336: 1,
        28672: 1,
        57344: 1
    },
    {  # tokens: [641:704]
        3072: 3,
        4096: 1,
        5120: 2,
        6144: 2,
        8192: 1,
        10240: 2,
        14336: 1,
        28672: 1,
        57344: 1
    },
    {  # tokens: [705:768]
        3072: 3,
        4096: 1,
        5120: 3,
        6144: 2,
        8192: 1,
        10240: 1,
        14336: 1,
        28672: 1,
        57344: 1
    }
]


def _f32_to_fpx_unpacked(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers with the given
    number of exponent and mantissa bits.
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding
    Note: there are no special values (NaN, inf) support in this code. Values
    outside the representable range of FPx after rounding are clamped to the
    maximum FPx magnitude (sign is preserved).
    Code below is an adaptation of https://fburl.com/code/ciwofcg4
    Background 1: last answer in https://stackoverflow.com/questions/8981913/how-to-perform-round-to-even-with-floating-point-numbers  # noqa: E501
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    # TODO document this better
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # all E bits and M bits are 1s
    max_normal = (2**(_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) /
                                                    (2**mbits)))

    # E bits = 1, M bits = 0
    min_normal = 2**(1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1)
    denorm_mask_int = denorm_exp << MBITS_F32

    # reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int,
                                     dtype=torch.int32).view(torch.float32)

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign

    # TODO: can the branch floating point comparisons below be done without
    # converting to float? probably but need to verify
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask),
                                      x < min_normal)
    normal_mask = torch.logical_not(
        torch.logical_or(saturate_mask, denormal_mask))

    #
    # branch 1: saturate to max val - handled later in the code which combines
    #   the branches
    #

    #
    # branch 2: to conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


# TODO(alpin): check if LUT for everything is faster than bit shifting,
# especially for fp4 (only 2^4=16 unique values).
def _fpx_unpacked_to_f32(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Convert sub-byte floating point numbers with the given number of
    exponent and mantissa bits to FP32.

    Input: torch.Tensor of dtype uint8, where the bit encoding is stored
        in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    assert x.dtype == torch.uint8
    assert 1 + ebits + mbits <= 8

    sign_mask = 1 << (ebits + mbits)
    exp_bias = _n_ones(ebits - 1)
    mantissa_mask = _n_ones(mbits)

    # save the sign
    sign_lp = x & sign_mask

    # set everything to positive, will add sign back at the end
    x_pos = x ^ sign_lp

    #
    # 1. Calculate zero mask
    #
    zero_mask = x_pos == 0

    #
    # 2. Calculate the denormal path mask
    #
    denormal_mask = torch.logical_and((x_pos > 0), ((x_pos >> mbits) == 0))

    #
    # 3. Calculate the normal path
    #

    # calculate the new exponent and shift it to bits 2:9 of the result
    exp_biased_lp = x_pos >> mbits
    exp_biased_f32 = exp_biased_lp - exp_bias + F32_EXP_BIAS
    exp_biased_f32 = exp_biased_f32.to(torch.int32) << MBITS_F32

    # shift the mantissa to bits 10:32 of the result
    mantissa_lp_int32 = (x_pos & mantissa_mask).to(torch.int32)
    mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - mbits)
    result = exp_biased_f32 | mantissa_f32

    #
    # 4. Add the zero and denormal casts to the already casted normal path
    #
    result[zero_mask] = 0

    denormal_exp_biased = 1 - exp_bias + F32_EXP_BIAS

    # fast path.
    # without this, performance for FP4_E2M1 is slower by 2x
    if mbits == 1:
        result[denormal_mask] = (denormal_exp_biased - mbits) << MBITS_F32

    else:
        # iterate over all possible values of mantissa
        # i=0, j=1
        # i=1, j=10,11
        # i=2, j=100,101,110,111
        # and so on
        for i in range(mbits):
            for mantissa_cmp in range(1 << i, 1 << (i + 1)):
                # left shift mantissa until it overflows (create an implicit 1)
                # subtract exponent by the same amount
                left_shift = mbits - i
                mantissa_f32 = (mantissa_cmp -
                                (1 << i)) << (left_shift + MBITS_F32 - mbits)
                exp_biased_f32 = (denormal_exp_biased -
                                  left_shift) << MBITS_F32

                # we can update this in-place since the values won't overlap
                # torch.compile() may complain unsupported operand type(s)
                # for |: 'SymInt' and 'int', thus we use + instead of | here
                mantissa_lp_int32[mantissa_lp_int32 == mantissa_cmp] = (
                    exp_biased_f32 + mantissa_f32)

        result = torch.where(denormal_mask, mantissa_lp_int32, result)

    # add sign back
    sign_f32 = sign_lp.to(
        torch.int32) << (MBITS_F32 - mbits + EBITS_F32 - ebits)
    result = result | sign_f32

    return result.view(torch.float)


_ONES_TABLE = [_n_ones(i) for i in range(8)]


def _pack(x: Tensor, n_bits: int) -> Tensor:
    return reduce(torch.bitwise_or, [
        x[..., i::(8 // n_bits)] << (8 - (i + 1) * n_bits)
        for i in range(8 // n_bits)
    ])


def _unpack(x: Tensor, n_bits: int) -> Tensor:
    return torch.stack([(x >> (8 - (i + 1) * n_bits)) & ((1 << n_bits) - 1)
                        for i in range(8 // n_bits)],
                       dim=-1).flatten(-2)


# https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/utils/weight_prepacking.h#L87-L116
def _bit_interleave(x: Tensor, n_bits: int, undo: bool = False) -> Tensor:
    # the original code unpacks/packs the values from/to uint32 while we
    # unpack/pack the values from/to uint8
    # thus, we need to reverse byte order within a uint32 word.
    x = x.reshape(-1, 4).flip(1)

    x = _unpack(x, n_bits)
    x = x.view(-1, 4 * (8 // n_bits))

    if not undo:
        bit_order = {
            1: [
                1, 5, 9, 13, 17, 21, 25, 29, 3, 7, 11, 15, 19, 23, 27, 31, 0,
                4, 8, 12, 16, 20, 24, 28, 2, 6, 10, 14, 18, 22, 26, 30
            ],
            2: [1, 5, 9, 13, 3, 7, 11, 15, 0, 4, 8, 12, 2, 6, 10, 14],
            4: [1, 5, 3, 7, 0, 4, 2, 6],
        }[n_bits]

    else:
        # this is inverse of the above, obtained by running
        # [v.index(i) for i in range(len(v))]
        bit_order = {
            1: [
                16, 0, 24, 8, 17, 1, 25, 9, 18, 2, 26, 10, 19, 3, 27, 11, 20,
                4, 28, 12, 21, 5, 29, 13, 22, 6, 30, 14, 23, 7, 31, 15
            ],
            2: [8, 0, 12, 4, 9, 1, 13, 5, 10, 2, 14, 6, 11, 3, 15, 7],
            4: [4, 0, 6, 2, 5, 1, 7, 3],
        }[n_bits]

    x = x[:, bit_order]
    x = _pack(x, n_bits)

    # reverse byte order within a uint32 word again.
    x = x.reshape(-1, 4).flip(1)
    return x.flatten()


# this is a literal adaptation of FP6-LLM ahead-of-time bit-level pre-packing
# https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/utils/weight_prepacking.h
def _pack_tc_fpx(tensor: Tensor, nbits: int) -> Tensor:
    assert tensor.ndim == 2, tensor.dtype == torch.uint8
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    # Pass 1 from original code
    tensor = tensor.view(M // 64, 4, 2, 8, N // 16, 2, 8)
    tensor = tensor.permute(0, 4, 1, 5, 2, 3, 6)
    tensor = tensor.reshape(-1, 32, 2)
    tensor = tensor.permute(1, 0, 2)
    tensor = tensor.flatten()

    used_bits = 0
    fragments = []

    for y in [1, 2, 4]:
        if nbits & y:
            mask = (1 << y) - 1
            tensor_ybit = (tensor >> (nbits - used_bits - y)) & mask
            tensor_ybit = _pack(tensor_ybit, y)

            tensor_ybit = tensor_ybit.view(32, -1, 4).permute(1, 0, 2).flip(2)
            tensor_ybit = _bit_interleave(tensor_ybit.flatten(), y)
            fragments.append(tensor_ybit)
            used_bits += y

    return torch.cat(fragments, dim=0).view(M, -1)


# more optimized version of _pack_tc_fpx() for FP6 by merging ops
def _pack_tc_fp6(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2, tensor.dtype == torch.uint8
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    tensor = tensor.view(M // 64, 2, 2, 2, 8, N // 16, 2, 8)
    tensor = tensor.flip(3)

    tensor_2bit = (tensor >> 4) & 0b11
    tensor_2bit = tensor_2bit.permute(0, 5, 1, 4, 7, 3, 2, 6)
    tensor_2bit = _pack(tensor_2bit.flatten(), 2)

    tensor_4bit = tensor & 0b1111
    tensor_4bit = tensor_4bit.permute(0, 5, 1, 2, 4, 7, 3, 6)
    tensor_4bit = _pack(tensor_4bit.flatten(), 4)

    return torch.cat([tensor_2bit, tensor_4bit], dim=0).view(M, -1)


# currently only optimize for TC-FP6 packing
def pack_tc_fpx(tensor: Tensor, nbits: int) -> Tensor:
    if nbits == 6:
        return _pack_tc_fp6(tensor)
    return _pack_tc_fpx(tensor, nbits)


def to_scaled_tc_fpx(tensor: Tensor, ebits: int,
                     mbits: int) -> Tuple[Tensor, Tensor]:
    # _n_ones() is not compatible with torch.compile() due to << operator
    # https://github.com/pytorch/pytorch/issues/119152
    # exp_bias = _n_ones(ebits - 1)
    # max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2 ** mbits))

    # workaround: global lookup table
    exp_bias = _ONES_TABLE[ebits - 1]
    max_normal = (2**(_ONES_TABLE[ebits] - exp_bias) *
                  (_ONES_TABLE[mbits + 1] / (2**mbits)))

    tensor = tensor.float()
    scale = tensor.abs().amax(1).clamp(min=1e-12) / max_normal
    tensor_fpx = _f32_to_fpx_unpacked(tensor / scale.view(-1, 1), ebits, mbits)
    tensor_tc_fpx = pack_tc_fpx(tensor_fpx, 1 + ebits + mbits)
    return tensor_tc_fpx, scale.half()


# inverse of _pack_tc_fpx()
def _unpack_tc_fpx(tensor: Tensor, nbits: int) -> Tensor:
    assert tensor.ndim == 2 and tensor.dtype == torch.uint8
    M = tensor.shape[0]
    size = tensor.numel()
    tensor = tensor.flatten()
    offset = 0
    used_bits = 0

    tensor_fpx = None

    for y in [1, 2, 4]:
        if nbits & y:
            size_ybit = size // nbits * y
            tensor_ybit = tensor[offset:offset + size_ybit]
            offset += size_ybit

            # undo Pass 3
            tensor_ybit = _bit_interleave(tensor_ybit, y, undo=True)
            # undo Pass 2
            tensor_ybit = tensor_ybit.view(-1, 32, 4).flip(2).permute(1, 0, 2)

            tensor_ybit = _unpack(tensor_ybit.flatten(), y)
            tensor_ybit = tensor_ybit << (nbits - used_bits - y)
            used_bits += y

            if tensor_fpx is None:
                tensor_fpx = tensor_ybit
            else:
                tensor_fpx |= tensor_ybit

    # undo Pass 1
    # NOTE(alpin): the view op here fails for FP8
    if tensor_fpx is None:
        tensor_fpx = torch.zeros_like(tensor_ybit)
    tensor_fpx = tensor_fpx.view(32, -1, 2).permute(1, 0, 2)
    tensor_fpx = tensor_fpx.reshape(M // 64, -1, 4, 2, 2, 8, 8)
    tensor_fpx = tensor_fpx.permute(0, 2, 4, 5, 1, 3, 6)
    tensor_fpx = tensor_fpx.reshape(M, -1)
    return tensor_fpx


# more optimized version of _unpack_tc_fpx() for FP6 by merging ops
# inverse of _unpack_tc_fp6()
def _unpack_tc_fp6(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2 and tensor.dtype == torch.uint8
    M = tensor.shape[0]
    N = tensor.shape[1] // 3 * 4
    assert (M % 64 == 0) and (N % 64 == 0)
    size_2bit = M * N // 4
    size_4bit = M * N // 2
    tensor = tensor.view(-1)
    assert tensor.numel() == size_2bit + size_4bit

    tensor_2bit, tensor_4bit = tensor.split([size_2bit, size_4bit])

    tensor_2bit = _unpack(tensor_2bit, 2)
    tensor_2bit = tensor_2bit.view(M // 64, N // 16, 2, 8, 8, 2, 2, 2)
    tensor_2bit = tensor_2bit.permute(0, 2, 6, 5, 3, 1, 7, 4)

    tensor_4bit = _unpack(tensor_4bit, 4)
    tensor_4bit = tensor_4bit.view(M // 64, N // 16, 2, 2, 8, 8, 2, 2)
    tensor_4bit = tensor_4bit.permute(0, 2, 3, 6, 4, 1, 7, 5)

    tensor_fp6 = (tensor_2bit << 4) | tensor_4bit
    tensor_fp6 = tensor_fp6.flip(3).reshape(M, N)
    return tensor_fp6


def unpack_tc_fpx(tensor: Tensor, nbits: int) -> Tensor:
    if nbits == 6:
        return _unpack_tc_fp6(tensor)
    return _unpack_tc_fpx(tensor, nbits)


def from_scaled_tc_fpx(tensor: Tensor,
                       ebits: int,
                       mbits: int,
                       scale=None) -> Tensor:
    fpx_unpacked = unpack_tc_fpx(tensor, 1 + ebits + mbits)
    tensor = _fpx_unpacked_to_f32(fpx_unpacked, ebits, mbits)
    if scale is not None:
        tensor = tensor * scale.float().view(-1, 1)
    return tensor
