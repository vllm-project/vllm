# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This file is used for /tests and /benchmarks"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import numpy
import torch
from torch import fx

from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

if TYPE_CHECKING:
    from vllm.model_executor.layers.linear import LinearBase

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def get_fp8_min_max() -> tuple[float, float]:
    """Get the min and max values for FP8 quantization."""
    # Using the default value (240.0) from pytorch will cause accuracy
    # issue on dynamic quantization models on ROCm. Here, use 224.0 for fnuz
    # on ROCm platforms that use the torch.float8_e4m3fnuz dtype.
    if current_platform.is_fp8_fnuz():
        return -224.0, 224.0
    finfo = torch.finfo(current_platform.fp8_dtype())
    return finfo.min, finfo.max


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
    PER_TENSOR: ClassVar["GroupShape"]
    PER_TOKEN: ClassVar["GroupShape"]
    PER_CHANNEL: ClassVar["GroupShape"]

    def is_per_tensor(self) -> bool:
        return self.row == -1 and self.col == -1

    def is_per_token(self) -> bool:
        return self.row == 1 and self.col == -1

    def is_per_channel(self) -> bool:
        return self.row == -1 and self.col == 1

    def is_per_group(self) -> bool:
        return self.row == 1 and self.col >= 1


GroupShape.PER_TENSOR = GroupShape(-1, -1)
GroupShape.PER_TOKEN = GroupShape(1, -1)
GroupShape.PER_CHANNEL = GroupShape(-1, 1)


@dataclass(frozen=True)
class ScaleDesc:
    """
    Class for describing a single quantization scaling factor.
    dtype: data type of the scale
    static: static scale if True, dynamic if False
    group_shape: group shape of the scale
    """

    dtype: torch.dtype
    static: bool
    group_shape: GroupShape

    def __str__(self):
        d = {
            GroupShape.PER_TENSOR: "per_tensor",
            GroupShape.PER_TOKEN: "per_token",
            GroupShape.PER_CHANNEL: "per_channel",
        }
        group_shape = d.get(self.group_shape, str(self.group_shape))
        return (
            f"{fx.graph.dtype_abbrs[self.dtype]},"
            f"{'static' if self.static else 'dynamic'},{group_shape}"
        )


@dataclass(frozen=True)
class QuantKey:
    """
    Class for identifying the type of quantization.
    dtype: quantized data type
    scale: scale descriptor
    scale2: second-level scale descriptor
    symmetric: symmetric if True, asymmetric if False
    """

    dtype: torch.dtype
    scale: ScaleDesc
    scale2: ScaleDesc | None = None
    symmetric: bool = True

    def __str__(self):
        scale2_str = f"scale2({self.scale2})," if self.scale2 else ""
        return (
            f"QuantKey({fx.graph.dtype_abbrs[self.dtype]},"
            f"scale({self.scale}),{scale2_str}"
            f"{'a' if not self.symmetric else ''}symmetric)"
        )


kStaticTensorScale = ScaleDesc(torch.float32, True, GroupShape.PER_TENSOR)
kFp8StaticTensorSym = QuantKey(FP8_DTYPE, kStaticTensorScale, symmetric=True)

kDynamicTensorScale = ScaleDesc(torch.float32, False, GroupShape.PER_TENSOR)
kFp8DynamicTensorSym = QuantKey(FP8_DTYPE, kDynamicTensorScale, symmetric=True)

kStaticTokenScale = ScaleDesc(torch.float32, True, GroupShape.PER_TOKEN)
kFp8StaticTokenSym = QuantKey(FP8_DTYPE, kStaticTokenScale, symmetric=True)

kStaticChannelScale = ScaleDesc(torch.float32, True, GroupShape.PER_CHANNEL)
kFp8StaticChannelSym = QuantKey(FP8_DTYPE, kStaticChannelScale, symmetric=True)

kDynamicTokenScale = ScaleDesc(torch.float32, False, GroupShape.PER_TOKEN)
kFp8DynamicTokenSym = QuantKey(FP8_DTYPE, kDynamicTokenScale, symmetric=True)

kNvfp4DynamicGroupScale = ScaleDesc(FP8_DTYPE, False, GroupShape(1, 16))
kNvfp4Dynamic = QuantKey(
    FP4_DTYPE, scale=kNvfp4DynamicGroupScale, scale2=kStaticTensorScale
)

kNvfp4StaticGroupScale = ScaleDesc(FP8_DTYPE, True, GroupShape(1, 16))
kNvfp4Static = QuantKey(
    FP4_DTYPE, scale=kNvfp4StaticGroupScale, scale2=kStaticTensorScale
)

kDynamic128Scale = ScaleDesc(torch.float32, False, GroupShape(1, 128))
kFp8Dynamic128Sym = QuantKey(FP8_DTYPE, kDynamic128Scale, symmetric=True)

kStatic128BlockScale = ScaleDesc(torch.float32, True, GroupShape(128, 128))
kFp8Static128BlockSym = QuantKey(FP8_DTYPE, kStatic128BlockScale, symmetric=True)

kDynamic64Scale = ScaleDesc(torch.float32, False, GroupShape(1, 64))
kFp8Dynamic64Sym = QuantKey(FP8_DTYPE, kDynamic64Scale, symmetric=True)

kDynamic128x128Scale = ScaleDesc(torch.float32, False, GroupShape(128, 128))
kFp8Dynamic128x128Sym = QuantKey(FP8_DTYPE, kDynamic128x128Scale, symmetric=True)


def create_fp8_quant_key(
    static: bool,
    group_shape: GroupShape,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float32,
) -> QuantKey:
    scale_desc = ScaleDesc(scale_dtype, static, group_shape)
    return QuantKey(FP8_DTYPE, scale_desc, symmetric=symmetric)


# Normalize the group_shape to the full extent for any dims that are -1
def _normalize_quant_group_shape(x: torch.Tensor, group_shape: GroupShape):
    # -1 means full extent
    return (
        group_shape[0] if group_shape[0] > 0 else x.shape[-2],
        group_shape[1] if group_shape[1] > 0 else x.shape[-1],
    )


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
# NOTE this function does not explicitly broadcast dimensions
# with an extent of 1, since this can be done implicitly by pytorch
def group_broadcast(t, shape):
    for i, s in enumerate(shape):
        # If tensor has fewer dimensions than target shape, treat missing
        # dimensions as size 1 (standard PyTorch broadcasting behavior)
        t_dim_size = t.shape[i] if i < t.ndim else 1
        if t_dim_size != s and t_dim_size != 1:
            assert s % t_dim_size == 0
            t = (
                t.unsqueeze(i + 1)
                .expand(*t.shape[: i + 1], s // t_dim_size, *t.shape[i + 1 :])
                .flatten(i, i + 1)
            )
    return t


def prep_scale_for_group_broadcast(
    scale: torch.Tensor,
    x: torch.Tensor,
    group_shape: GroupShape | None,
) -> torch.Tensor:
    """
    Prepare the input quantization scale for group broadcasting.

    Args:
        scale: The scale tensor (scalar or 1D).
        x: Target tensor whose shape determines broadcast dimensions.
        group_shape: GroupShape to broadcast over.

    Returns:
        scale reshaped for correct broadcasting.
    """
    if scale.numel() == 1:
        # For per-tensor quant, keep the scale as a scalar (not reshaped to (1, 1)).
        # This avoids misclassifying it as channelwise quant in Fp8LinearOp.apply,
        # where the "per_tensor_activations" check relies on "x_scale.dim() < 2":
        #   per_tensor_activations = (x_scale.numel() == 1) and x_scale.dim() < 2
        # For all other cases, reshape scalar scales to (1, 1) for broadcasting.
        return (
            scale
            if group_shape is not None and group_shape.is_per_tensor()
            else scale.reshape(1, 1)
        )
    if scale.ndim == 1:
        assert group_shape is not None, (
            "group_shape must be provided to correctly broadcast 1D scale"
        )
        rows, cols = _normalize_quant_group_shape(x, group_shape)
        # Determine broadcasting dimension: either rows or columns match group size
        if rows == x.shape[-2]:
            scale = scale.unsqueeze(-2)
        elif cols == x.shape[-1]:
            scale = scale.unsqueeze(-1)
        else:
            raise ValueError(
                f"1D scale with shape {scale.shape} cannot be broadcast to x with shape"
                f" {x.shape}, group_shape={(rows, cols)}"
            )
    return scale


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
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: Input tensor to quantize
        group_shape: Shape of quantization groups
        quant_dtype: Target quantized dtype (e.g., torch.float8_e4m3fn)
        compute_dtype: Optional dtype for intermediate computations.
            If None, uses input dtype. Use torch.float32 for higher precision.
    """
    group_shape = _normalize_quant_group_shape(x, group_shape)
    assert quant_dtype.is_floating_point, (
        "currently `scaled_quantize` only supports floating point dtypes "
        "but could be extended to support other dtypes"
    )

    finfo = torch.finfo(quant_dtype)

    # Convert to compute dtype if specified
    x_compute = x if compute_dtype is None else x.to(compute_dtype)

    # Reshape (M, N) into (BLK_M, BLOCK_SIZE_M, BLK_N, BLOCK_SIZE_N)
    assert x.ndim == 2
    assert x.shape[0] % group_shape[0] == 0 and x.shape[1] % group_shape[1] == 0
    blk_m, blk_n = x.shape[0] // group_shape[0], x.shape[1] // group_shape[1]
    x_blkd = x_compute.reshape(blk_m, group_shape[0], blk_n, group_shape[1])

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    x_blkd_permd = x_blkd.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    x_blkd_permd = x_blkd_permd.flatten(start_dim=2)

    # Compute scales
    min_val, max_val = x_blkd_permd.aminmax(dim=-1)
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    _, fp8_max = get_fp8_min_max()
    scale = fp8_max / amax

    # Apply scale and convert from:
    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    x_scl_sat = (
        (x_blkd_permd * scale.unsqueeze(-1))
        .clamp(min=finfo.min, max=finfo.max)
        .reshape(blk_m, blk_n, group_shape[0], group_shape[1])
        .permute(0, 2, 1, 3)
        .reshape(x.shape)
    )

    return x_scl_sat.to(quant_dtype).contiguous(), scale.float().reciprocal()


# inverses `scaled_quantize`
def scaled_dequantize(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    group_shape: GroupShape | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    x_s = prep_scale_for_group_broadcast(x_s, x_q, group_shape)
    if group_shape is not None:
        assert x_s.shape[-1] == x_q.shape[-1] // group_shape[1]
        assert x_s.shape[-2] == x_q.shape[-2] // group_shape[0]
    x_s = group_broadcast(x_s.to(torch.float32), x_q.shape)
    return (x_q.to(torch.float32) * x_s).to(out_dtype)


def get_attribute_fallback(obj, attributes: list[str]):
    for attr in attributes:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    raise AttributeError(f"'{obj}' has no recognized attributes: {attributes}.")


def get_and_maybe_dequant_weights(
    layer: "LinearBase", out_dtype: torch.dtype = torch.float32
):
    """Return layer's unquantized weights in [out, in] layout"""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    weight = get_attribute_fallback(layer, ["weight", "qweight", "weight_packed"])

    # Unquantized layer: just return base weights
    if layer.quant_method is None or isinstance(
        layer.quant_method, UnquantizedLinearMethod
    ):
        return weight.to(out_dtype)

    # Simple Fp8 case: rescale with tensor or block weight scales
    if (
        isinstance(layer.quant_method, Fp8LinearMethod)
        and not layer.quant_method.use_marlin
        # DeepGEMM transforms the scales using `transform_sf_into_required_layout` into
        # a layout that is not compatible with `scaled_dequantize`.
        and not layer.quant_method.use_deep_gemm
    ):
        weight_scales = get_attribute_fallback(
            layer, ["weight_scale", "weight_scale_inv"]
        )
        dequant_weights = scaled_dequantize(
            weight,
            weight_scales,
            group_shape=layer.weight_block_size,
            out_dtype=out_dtype,
        )
        # per-tensor scaling stores weights in [in, out] layout
        if not layer.quant_method.block_quant:
            dequant_weights = dequant_weights.T
        return dequant_weights

    # NOTE: Most generic base case
    # - Call the layer with identity matrix which returns unquantized weights.
    # - Must be used with extra care when dealing with static activation quantization:
    #   quantizing 1.0 may lead to over/underflows
    # - Should only be used offline, since it's O(N^3)
    assert hasattr(layer, "input_size_per_partition")
    eye = torch.eye(
        layer.input_size_per_partition,
        dtype=out_dtype,
        device=weight.device,
    )
    dequant_weights = layer.quant_method.apply(layer, eye, bias=None).to(out_dtype)
    return dequant_weights.T


def pack_quantized_values_into_int32(
    w_q: torch.Tensor, wtype: ScalarType, packed_dim: int = 0
):
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


def unpack_quantized_values_into_int32(
    w_q: torch.Tensor, wtype: ScalarType, packed_dim: int = 0
):
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
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
    *,
    skip_with_substr: bool = False,
) -> bool:
    def prefix_full_match(prefix: str, ignored_layers: list[str]) -> bool:
        return prefix in ignored_layers

    # For case like: ignored_layers = ["self_attn"]
    def substr_match(prefix: str, ignored_layers: list[str]) -> bool:
        return any(layer in prefix for layer in ignored_layers)

    match_func = substr_match if skip_with_substr else prefix_full_match

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
            is_shard_skipped = match_func(shard_prefix, ignored_layers)

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    elif "experts" in prefix and not skip_with_substr:
        expert_ignore_layers = filter(
            lambda layer_name: "experts" in layer_name, ignored_layers
        )
        return any(
            prefix in layer_name if not skip_with_substr else layer_name in prefix
            for layer_name in expert_ignore_layers
        )
    else:
        is_skipped = match_func(prefix, ignored_layers)

    assert is_skipped is not None
    return is_skipped


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def permute_rows(
    q_w: torch.Tensor,
    w_ref: torch.Tensor,
    group_size: int,
    test_perm: torch.Tensor | None = None,
):
    assert q_w.shape == w_ref.shape

    orig_device = q_w.device
    k_size, _ = q_w.shape

    g_idx = torch.zeros((k_size,), dtype=torch.int32)
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


def quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int | None,
    zero_points: bool = False,
    ref_zero_points_after_scales: bool = False,
):
    assert quant_type.is_integer(), (
        "Floating point quantization may work but has not been tested"
    )
    assert not zero_points or group_size is not None, (
        "to have group zero points, group_size must be provided "
        "(-1 group_size is channelwise)"
    )

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
            maybe_w_zp = (
                torch.round(torch.abs(min_val / w_s)).clamp(min_q_val, max_q_val).int()
            )
        else:
            # If the bias is such that there are no possible negative/positive
            #  values, set the max value to inf to avoid divide by 0
            w_s = torch.max(
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)),
            )

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


def gptq_quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: torch.Tensor | None = None,
):
    size_k, _ = w.shape

    assert w.is_floating_point(), "w must be float"
    assert quant_type in SUPPORTED_GPTQ_QUANT_TYPES, (
        f"Unsupported gptq type = {quant_type}"
    )
    assert group_size in SUPPORTED_GROUP_SIZES + [size_k], (
        f"Unsupported groupsize = {group_size}"
    )

    w_ref, w_q, w_s, _ = quantize_weights(w, quant_type, group_size)

    # Apply act_order
    g_idx = torch.empty(0, dtype=torch.int, device=w.device)
    rand_perm = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        assert group_size < size_k, (
            "For act_order, groupsize = {} must be less than size_k = {}".format(
                group_size, size_k
            )
        )

        w_ref, w_q, g_idx, rand_perm = permute_rows(w_q, w_ref, group_size, test_perm)

    return w_ref, w_q, w_s, g_idx, rand_perm


def sort_weights(q_w: torch.Tensor, g_idx: torch.Tensor):
    orig_device = q_w.device

    sort_indices = torch.argsort(g_idx).to(dtype=torch.int32)  # Sort based on g_idx

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
    assert packed_q_w.shape == (size_k, size_n // pack_factor), (
        "packed_q_w.shape = {} size_k = {}, size_n = {} pack_Factor = {}".format(
            packed_q_w.shape, size_k, size_n, pack_factor
        )
    )

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


def convert_bf16_scales_to_fp8(
    quant_fp8: Callable, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a BF16 scale tensor into the pair of (fp8_scales, channel_scales)
    expected by W4A8 GEMM kernels.
    """
    assert scales.is_contiguous(), (
        f"scale tensor must be contiguous, got {scales.stride()=}"
    )
    assert scales.is_cuda, "scales must be on gpu"

    orig_shape = scales.shape
    k_groups = orig_shape[-1]
    flat_scales = scales.view(-1, k_groups)

    fp8_scales, chan_scales = quant_fp8(flat_scales)
    fp8_scales = (fp8_scales.float() / 8.0).to(torch.float8_e4m3fn)
    chan_scales *= 8.0

    # restore original shape
    fp8_scales = fp8_scales.view(orig_shape)
    chan_scales = chan_scales.view(orig_shape[:-1], -1)

    return fp8_scales, chan_scales


def convert_packed_uint4b8_to_signed_int4_inplace(t: torch.Tensor) -> torch.Tensor:
    """
    Convert int4b8 (packed to int32) to signed int4
    """
    assert t.is_cuda, "tensor must be on gpu"
    assert t.dtype == torch.int32, f"expected int32 packed weights but got {t.dtype}"

    # loop through the 8 4-bit nibbles in each int32 entry
    for i in range(8):
        shift = 4 * i
        # extract the i-th 4-bit nibble
        nib = (t >> shift) & 0xF
        # clear the original nibble by masking out
        t &= ~(0xF << shift)
        # convert int4b8 [0..15] to signed int4 [-8..7] by subtracting 8
        # and update in-place
        t |= ((nib - 8) & 0xF) << shift

    return t
