# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This file is used for /tests and /benchmarks"""
from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, NamedTuple, Optional

import numpy
import torch

from vllm._custom_ops import cutlass_scaled_mm_supports_fp4
from vllm.model_executor.layers.quantization.qqq import (
    MARLIN_QQQ_SUPPORTED_NUM_BITS)
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types


# Use proxy as NamedTuple direct subclasses cannot have static members
class _GroupShape(NamedTuple):
    row: int
    col: int


class GroupShape(_GroupShape):
    """
    This class describes the quantization group shape.
    It includes static members for common shapes (per-tensor, per-token).
    """

    # Aliases for common quantization group shapes
    PER_TENSOR: ClassVar['GroupShape']
    PER_TOKEN: ClassVar['GroupShape']


GroupShape.PER_TENSOR = GroupShape(-1, -1)
GroupShape.PER_TOKEN = GroupShape(1, -1)


# Normalize the group_shape to the full extent for any dims that are -1
def _normalize_quant_group_shape(x: torch.Tensor, group_shape: GroupShape):
    # -1 means full extent
    return (group_shape[0] if group_shape[0] > 0 else x.shape[-2],
            group_shape[1] if group_shape[1] > 0 else x.shape[-1])


# Useful when treating N-dimensional group scaling as extended numpy-style
# broadcasting in numpy simply stretches dimensions with an extent of 1 to match
# the target shape by repeating the data along that dimension (broadcasting)
# , we extend these semantics to say if the extent of a dimension in the
# source shape is not 1 and does not match the target shape we repeat each
# element along that dimension src_shape[dim] // target_shape[dim] times
# example if we have:
#       a = [[1, 2], and target_shape = (2, 4)
#            [3, 4]]
# then we would expand a to:
#       a = [[1, 1, 2, 2],
#            [3, 3, 4, 4]]
# NOTE this function this function does not explicitly broadcast dimensions
# with an extent of 1, since this can be done implicitly by pytorch
def group_broadcast(t, shape):
    for i, s in enumerate(shape):
        if t.shape[i] != s and t.shape[i] != 1:
            assert s % t.shape[i] == 0
            t = t.unsqueeze(i + 1)\
                .expand(*t.shape[:i+1], s // t.shape[i], *t.shape[i+1:])\
                .flatten(i, i + 1)
    return t


# Quantize assuming once scale per group of elements with shape group_shape,
# example group shapes:
#  * (-1, -1)   for per-tensor quantization
#  * (1, -1)    for per-row quantization
#  * (-1, 1)    for per-column quantization
#  * (128, 128) for 128x128 deepseek style block quantization
#  * (1, 128)   for deepseek style activation quantization
#               (i.e. per-token-per-group)
def scaled_quantize(
    x: torch.Tensor,
    group_shape: GroupShape,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_shape = _normalize_quant_group_shape(x, group_shape)
    assert quant_dtype.is_floating_point, \
        "currently `scaled_quantize` only supports floating point dtypes " \
        "but could be extended to support other dtypes"

    finfo = torch.finfo(quant_dtype)

    # Reshape (M, N) into (BLK_M, BLOCK_SIZE_M, BLK_N, BLOCK_SIZE_N)
    assert x.ndim == 2
    assert x.shape[0] % group_shape[0] == 0 and x.shape[1] % group_shape[1] == 0
    blk_m, blk_n = x.shape[0] // group_shape[0], x.shape[1] // group_shape[1]
    x_blkd = x.reshape(blk_m, group_shape[0], blk_n, group_shape[1])

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    x_blkd_permd = x_blkd.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    x_blkd_permd = x_blkd_permd.flatten(start_dim=2)

    # Compute scales
    min_val, max_val = x_blkd_permd.aminmax(dim=-1)
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax

    # Apply scale and convert form:
    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    x_scl_sat = (x_blkd_permd * scale.unsqueeze(-1))\
        .clamp(min=finfo.min, max=finfo.max)\
        .reshape(blk_m, blk_n, group_shape[0], group_shape[1])\
        .permute(0, 2, 1, 3)\
        .reshape(x.shape)

    return x_scl_sat.to(quant_dtype).contiguous(), scale.float().reciprocal()


# inverses `scaled_quantize`
def scaled_dequantize(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    group_shape: Optional[GroupShape] = None,
    out_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if group_shape is not None:
        group_shape = _normalize_quant_group_shape(x_q, group_shape)

    if x_s.ndim == 0:  # scalar
        x_s = x_s.unsqueeze(-1).unsqueeze(-1)  # convert to (1, 1) tensor
    if x_s.ndim == 1:
        if group_shape is None:
            raise AssertionError(
                "if x_s is 1D tensor, group_shape must be provided otherwise "
                "its ambiguous which dimension to broadcast x_s to")
        # unsqueeze the scales for the dimension where we want to broadcast
        # across the full extent
        if group_shape[0] == x_q.shape[-2]:
            x_s = x_s.unsqueeze(-2)
        elif group_shape[1] == x_q.shape[-1]:
            x_s = x_s.unsqueeze(-1)
        else:
            raise AssertionError(
                "if x_s is a vector we should be broadcasting it to the full "
                "extent of one of the dimensions")

    if group_shape is not None:
        assert x_s.shape[-1] == x_q.shape[-1] // group_shape[1]
        assert x_s.shape[-2] == x_q.shape[-2] // group_shape[0]
    x_s = group_broadcast(x_s.to(torch.float32), x_q.shape)
    return (x_q.to(torch.float32) * x_s).to(out_dtype)


def pack_quantized_values_into_int32(w_q: torch.Tensor,
                                     wtype: ScalarType,
                                     packed_dim: int = 0):
    # move dim to pack to the end
    perm = (*[i for i in range(len(w_q.shape)) if i != packed_dim], packed_dim)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    w_q_perm = w_q.permute(perm)

    pack_factor = 32 // wtype.size_bits
    mask = (1 << wtype.size_bits) - 1

    new_shape_perm = list(w_q_perm.shape)
    assert w_q_perm.shape[-1] % pack_factor == 0
    new_shape_perm[-1] //= pack_factor

    res = torch.zeros(new_shape_perm, dtype=torch.int32, device=w_q.device)
    for i in range(pack_factor):
        res |= (w_q_perm[..., i::pack_factor] & mask) << wtype.size_bits * i

    return res.permute(inv_perm)


def unpack_quantized_values_into_int32(w_q: torch.Tensor,
                                       wtype: ScalarType,
                                       packed_dim: int = 0):
    # move dim to pack to the end
    perm = (*[i for i in range(len(w_q.shape)) if i != packed_dim], packed_dim)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    w_q_perm = w_q.permute(perm)

    pack_factor = 32 // wtype.size_bits
    mask = (1 << wtype.size_bits) - 1

    new_shape_perm = list(w_q_perm.shape)
    new_shape_perm[-1] *= pack_factor

    res = torch.zeros(new_shape_perm, dtype=torch.int32, device=w_q.device)
    for i in range(pack_factor):
        res[..., i::pack_factor] = (w_q_perm >> wtype.size_bits * i) & mask

    return res.permute(inv_perm)


def is_layer_skipped(
    prefix: str,
    ignored_layers: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({})
) -> bool:
    # prefix: model.layers.0.self_attn.q_proj
    # proj_name: q_proj
    proj_name = prefix.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = shard_prefix in ignored_layers

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision.")
    else:
        is_skipped = prefix in ignored_layers

    assert is_skipped is not None
    return is_skipped


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def permute_rows(q_w: torch.Tensor,
                 w_ref: torch.Tensor,
                 group_size: int,
                 test_perm: Optional[torch.Tensor] = None):
    assert q_w.shape == w_ref.shape

    orig_device = q_w.device
    k_size, _ = q_w.shape

    g_idx = torch.zeros((k_size, ), dtype=torch.int32)
    for i in range(k_size):
        g_idx[i] = i // group_size

    # Simulate act_order by doing a random permutation on K
    rand_perm = test_perm if test_perm is not None else torch.randperm(k_size)

    g_idx = g_idx[rand_perm].contiguous()
    q_w = q_w[rand_perm, :].contiguous()
    w_ref = w_ref[rand_perm, :].contiguous()

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        rand_perm.to(device=orig_device),
    )


def quantize_weights(w: torch.Tensor,
                     quant_type: ScalarType,
                     group_size: Optional[int],
                     zero_points: bool = False,
                     ref_zero_points_after_scales: bool = False):
    assert quant_type.is_integer(), \
        "Floating point quantization may work but has not been tested"
    assert not zero_points or group_size is not None, \
        "to have group zero points, group_size must be provided "\
        "(-1 group_size is channelwise)"

    orig_device = w.device
    orig_type = w.dtype
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k

    # Reshape to [groupsize, -1]
    if group_size is not None and group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    # Compute scale for each group
    max_val = torch.max(w, 0, keepdim=True).values
    min_val = torch.min(w, 0, keepdim=True).values

    max_q_val = quant_type.max()
    min_q_val = quant_type.min()

    w_s = torch.Tensor([1.0]).to(w.device)  # unscaled case
    maybe_w_zp = None
    if group_size is not None:
        if zero_points:
            assert not quant_type.is_signed() and quant_type.max() > 0
            w_s = (max_val - min_val).clamp(min=1e-5) / quant_type.max()
            maybe_w_zp = torch.round(torch.abs(min_val / w_s)) \
                .clamp(min_q_val, max_q_val).int()
        else:
            # If the bias is such that there are no possible negative/positive
            #  values, set the max value to inf to avoid divide by 0
            w_s = torch.max(
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)))

    # Quantize
    w_q = torch.round(w / w_s).int() + (maybe_w_zp if zero_points else 0)
    w_q = torch.clamp(w_q, min_q_val, max_q_val)

    # Compute ref (dequantized)
    # For some kernels (namely Machete) the zero-points are applied after the
    # scales are applied, for this case computing the reference in similar way
    # allows us to use tighter error tolerances in our unit tests.
    if ref_zero_points_after_scales and maybe_w_zp is not None:
        w_ref = w_q.to(orig_type) * w_s - maybe_w_zp.to(orig_type) * w_s
    else:
        w_ref = (w_q - (maybe_w_zp if zero_points else 0)).to(orig_type) * w_s

    if quant_type.has_bias():
        w_q += quant_type.bias

    # Restore original shapes
    if group_size is not None and group_size < size_k:

        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        w_q = reshape_w(w_q)
        w_ref = reshape_w(w_ref)
        w_s = w_s.reshape((-1, size_n)).contiguous()

    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous()
        maybe_w_zp = maybe_w_zp.to(device=orig_device)

    return (
        w_ref.to(device=orig_device),
        w_q.to(device=orig_device),
        w_s if group_size is not None else None,
        maybe_w_zp,
    )


SUPPORTED_GPTQ_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


def gptq_quantize_weights(w: torch.Tensor,
                          quant_type: ScalarType,
                          group_size: int,
                          act_order: bool,
                          test_perm: Optional[torch.Tensor] = None):
    size_k, _ = w.shape

    assert w.is_floating_point(), "w must be float"
    assert quant_type in SUPPORTED_GPTQ_QUANT_TYPES, \
        f"Unsupported gptq type = {quant_type}"
    assert group_size in SUPPORTED_GROUP_SIZES + [
        size_k
    ], f"Unsupported groupsize = {group_size}"

    w_ref, w_q, w_s, _ = quantize_weights(w, quant_type, group_size)

    # Apply act_order
    g_idx = torch.empty(0, dtype=torch.int, device=w.device)
    rand_perm = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        assert (
            group_size < size_k
        ), "For act_order, groupsize = {} must be less than size_k = {}".format(
            group_size, size_k)

        w_ref, w_q, g_idx, rand_perm = permute_rows(w_q, w_ref, group_size,
                                                    test_perm)

    return w_ref, w_q, w_s, g_idx, rand_perm


# QQQ employs different quant schemes for per-group and
# per-channel quantization.
def qqq_quantize_weights(w: torch.Tensor, num_bits: int, group_size: int):
    orig_device = w.device
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"
    assert num_bits in MARLIN_QQQ_SUPPORTED_NUM_BITS, \
           f"Unsupported num_bits = {num_bits}"
    assert group_size in SUPPORTED_GROUP_SIZES + [
        size_k
    ], f"Unsupported groupsize = {group_size}"

    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    if group_size < size_k:
        # Reshape to [groupsize, -1]
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

        max_q_val = 2**num_bits - 1
        half_q_val = (max_q_val + 1) // 2

        # Compute scale for each group
        s_group = torch.max(torch.abs(w), 0, keepdim=True)[0]
        s_group *= 2 / max_q_val  # 2 => symmetric

        # Quantize
        q_w = torch.round(w / s_group).int()
        q_w += half_q_val
        q_w = torch.clamp(q_w, 0, max_q_val)
        # Compute ref (dequantized)
        w_ref = (q_w - half_q_val).half() * s_group

        # Restore original shapes
        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        q_w = reshape_w(q_w)
        w_ref = reshape_w(w_ref)

        # Compute int8 quantization scale for each channel
        s_channel = torch.max(torch.abs(w_ref), 0, keepdim=True)[0]
        s_channel /= 127.0
        t_int8 = (w_ref / s_channel).round().clamp(-128, 127).to(torch.int8)
        w_ref = t_int8.half() * s_channel
        s_channel = s_channel.reshape(1, -1).to(dtype=torch.float)

        # Fuse scales
        s_group = (s_group.reshape(-1, size_n).contiguous() /
                   s_channel).to(dtype=torch.half)
    else:
        max_q_val = 2**(num_bits - 1) - 1

        # Compute scale for each channel
        s_channel = torch.max(torch.abs(w), 0, keepdim=True)[0]
        s_channel /= max_q_val

        # Quantize
        q_w = torch.round(w / s_channel).int()
        q_w = torch.clamp(q_w, -max_q_val, max_q_val)
        # Compute ref (dequantized)
        w_ref = q_w.half() * s_channel

        s_group = torch.tensor([], dtype=torch.half)
        # div 2 ** (8 - self.bits)) to offset right shift in unpacking
        s_channel /= (2**(8 - num_bits))
        s_channel = s_channel.reshape(-1, size_n).contiguous().to(torch.float)

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        s_group.to(device=orig_device),
        s_channel.to(device=orig_device),
    )


def sort_weights(q_w: torch.Tensor, g_idx: torch.Tensor):
    orig_device = q_w.device

    sort_indices = torch.argsort(g_idx).to(
        dtype=torch.int32)  # Sort based on g_idx

    g_idx = g_idx[sort_indices].contiguous()
    q_w = q_w[sort_indices, :].contiguous()

    return (
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        sort_indices.to(device=orig_device),
    )


def pack_rows(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_k % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k // pack_factor, size_n), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[i::pack_factor, :] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    return q_res


def pack_cols(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def unpack_cols(
    packed_q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0
    assert packed_q_w.shape == (
        size_k, size_n // pack_factor
    ), "packed_q_w.shape = {} size_k = {}, size_n = {} pack_Factor = {}".format(
        packed_q_w.shape, size_k, size_n, pack_factor)

    orig_device = packed_q_w.device

    packed_q_w_cpu = packed_q_w.cpu().numpy().astype(numpy.uint32)
    q_res = numpy.zeros((size_k, size_n), dtype=numpy.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = packed_q_w_cpu & mask
        packed_q_w_cpu >>= num_bits
        q_res[:, i::pack_factor] = vals

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def gptq_pack(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    return pack_rows(q_w, num_bits, size_k, size_n)


def awq_pack(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_w = q_w.reshape((-1, len(interleave)))[:, interleave].ravel()
    q_w = q_w.reshape((-1, size_n)).contiguous()

    return pack_cols(q_w, num_bits, size_k, size_n)


def swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    """
    Pad and block-interleave the FP4 block-scales so that they match the data
    layout expected by the CUTLASS / FlashInfer kernels.

    Parameters
    ----------
    scale: torch.Tensor

    Returns
    -------
    torch.Tensor
        The swizzled tensor with the same logical shape as *scale*.
    """
    assert scale.dtype == torch.float8_e4m3fn, (
        "swizzle_blockscale expects the input tensor to be in "
        "torch.float8_e4m3fn format.")

    scale_ndim = scale.ndim
    if scale_ndim == 2:
        scale = scale.unsqueeze(0)  # (1, M, K)
    assert scale.ndim == 3, "Expected a 2-D or 3-D tensor for block scales."

    B, M, K = scale.shape

    def _round_up(x: int, m: int) -> int:
        return (x + m - 1) // m * m

    M_padded = _round_up(M, 128)
    K_padded = _round_up(K, 4)

    padded = torch.zeros((B, M_padded, K_padded),
                         dtype=scale.dtype,
                         device=scale.device)
    padded[:B, :M, :K] = scale

    # Reshape / permute to the layout required by the kernel.
    padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()

    if scale_ndim == 2:
        return swizzled.reshape(M, K)
    return swizzled.reshape(B, M, K)


def cutlass_fp4_supported() -> bool:
    if not current_platform.is_cuda():
        return False
    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()
    return cutlass_scaled_mm_supports_fp4(capability)
