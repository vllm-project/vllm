# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dequantization of the fused DFlash context-KV weights.

``_project_context_kv`` runs one fused ``F.linear`` across every layer's K/V
weights, bypassing ``quant_method.apply()``, so ``_dequant_kv_slice`` has to
materialize those rows itself. It is the only place that reads a checkpoint's
raw quantized storage directly, and the layouts it must accept differ in shape
rather than in name: a per-tensor scheme on a *fused* qkv stores one scalar per
shard, which is neither a scalar nor one entry per row.

Getting a layout wrong here does not raise -- it broadcasts the wrong scale over
the wrong rows and shows up only as degraded draft acceptance, so each accepted
layout is pinned and each rejected one is required to name what it saw.
"""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.base import (
    pack_to_int32,
    unpack_from_int32,
)

from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

dequant_kv_slice = DFlashQwen3Model._dequant_kv_slice

ACT_DTYPE = torch.bfloat16
Q_SIZE, KV_SIZE, HIDDEN = 128, 32, 256
OUT_SIZE = Q_SIZE + 2 * KV_SIZE
SHARDS = [Q_SIZE, KV_SIZE, KV_SIZE]


def _attn(**qkv_attrs):
    qkv_attrs.setdefault("weight", None)
    qkv_attrs.setdefault("input_size", HIDDEN)
    return SimpleNamespace(qkv_proj=SimpleNamespace(**qkv_attrs), q_size=Q_SIZE)


def _quantized_weight():
    return torch.randn(OUT_SIZE, HIDDEN).to(torch.float8_e4m3fn)


def test_unquantized_weight_is_passed_through():
    weight = torch.randn(OUT_SIZE, HIDDEN, dtype=ACT_DTYPE)
    out = dequant_kv_slice(_attn(weight=weight, weight_scale=None), ACT_DTYPE)
    assert torch.equal(out, weight[Q_SIZE:])


def test_per_tensor_scale_applies_to_every_row():
    weight = _quantized_weight()
    scale = torch.tensor(0.03)
    out = dequant_kv_slice(_attn(weight=weight, weight_scale=scale), ACT_DTYPE)
    assert torch.equal(out, weight[Q_SIZE:].to(ACT_DTYPE) * scale.to(ACT_DTYPE))


def test_per_channel_scale_is_sliced_to_the_kv_rows():
    weight = _quantized_weight()
    scale = torch.rand(OUT_SIZE) * 0.05 + 0.001
    out = dequant_kv_slice(_attn(weight=weight, weight_scale=scale), ACT_DTYPE)
    expected = weight[Q_SIZE:].to(ACT_DTYPE) * scale[Q_SIZE:].to(ACT_DTYPE).reshape(
        -1, 1
    )
    assert torch.equal(out, expected)


@pytest.mark.parametrize("sizes_attr", ["output_partition_sizes", "output_sizes"])
def test_per_shard_scale_on_a_fused_qkv(sizes_attr):
    """A per-tensor scheme on a merged qkv stores ``weight_scale`` as (3,).

    K and V own different scalars, so expanding each shard's scalar over the rows
    it owns has to happen before the slice -- taking ``scale[q_size:]`` of a (3,)
    tensor would silently hand V's scalar to K.
    """
    weight = _quantized_weight()
    scale = torch.tensor([0.02, 0.05, 0.07])
    out = dequant_kv_slice(
        _attn(weight=weight, weight_scale=scale, **{sizes_attr: SHARDS}), ACT_DTYPE
    )

    per_row = torch.cat([scale[i].expand(n) for i, n in enumerate(SHARDS)])
    expected = weight[Q_SIZE:].to(ACT_DTYPE) * per_row[Q_SIZE:].to(ACT_DTYPE).reshape(
        -1, 1
    )
    assert torch.equal(out, expected)


def test_scale_that_maps_onto_no_layout_is_rejected():
    attn = _attn(weight=_quantized_weight(), weight_scale=torch.rand(7))
    with pytest.raises(ValueError, match=r"weight_scale of \(7,\)"):
        dequant_kv_slice(attn, ACT_DTYPE)


def test_quantized_weight_without_a_scale_is_rejected():
    attn = _attn(weight=_quantized_weight(), weight_scale=None)
    with pytest.raises(ValueError, match="no weight_scale"):
        dequant_kv_slice(attn, ACT_DTYPE)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [128, HIDDEN])
def test_packed_slice_matches_unpacking_the_whole_weight(bits, group_size):
    """Slicing before the unpack must not change the result, only its cost.

    Both tensors are row-major over output features, so the q rows can be dropped
    up front instead of being unpacked into fp32 and discarded.
    """
    limit = 2 ** (bits - 1)
    values = torch.randint(-limit, limit, (OUT_SIZE, HIDDEN), dtype=torch.int8)
    packed = pack_to_int32(values, bits, packed_dim=1)
    scale = torch.rand(OUT_SIZE, HIDDEN // group_size, dtype=ACT_DTYPE) * 0.05 + 0.001

    out = dequant_kv_slice(_attn(weight_packed=packed, weight_scale=scale), ACT_DTYPE)

    whole = unpack_from_int32(
        packed, bits, torch.Size([OUT_SIZE, HIDDEN]), packed_dim=1
    )
    expected = (
        whole.to(torch.float32).reshape(OUT_SIZE, HIDDEN // group_size, group_size)
        * scale.to(torch.float32)[..., None]
    ).reshape(OUT_SIZE, HIDDEN)
    assert torch.equal(out, expected.to(ACT_DTYPE)[Q_SIZE:])


@pytest.mark.parametrize(
    "scale,input_size",
    [
        (torch.rand(OUT_SIZE), HIDDEN),  # group scale must be 2-D
        (torch.rand(OUT_SIZE, 2), HIDDEN + 1),  # bit width would not divide evenly
    ],
)
def test_unreadable_packed_layout_is_rejected(scale, input_size):
    """The bit width is derived from the column count, so it cannot fail on its
    own: a floor division turns an unreadable layout into a plausible width and
    then into silent garbage."""
    values = torch.randint(-8, 8, (OUT_SIZE, HIDDEN), dtype=torch.int8)
    attn = _attn(
        weight_packed=pack_to_int32(values, 4, packed_dim=1),
        weight_scale=scale,
        input_size=input_size,
    )
    with pytest.raises(ValueError, match="cannot read a .* packed weight"):
        dequant_kv_slice(attn, ACT_DTYPE)


def test_nvfp4_style_uint8_packing_is_rejected():
    """A uint8 container is the case the bit-width arithmetic cannot see.

    NVFP4 exports pack two 4-bit values per uint8, so half as many columns as the
    int32 assumption predicts. For real geometry (5120 input features, 2560
    columns) that yields bits=16 with no remainder -- clean, and wrong. Only the
    container dtype distinguishes it.
    """
    packed = torch.zeros(OUT_SIZE, HIDDEN // 2, dtype=torch.uint8)
    attn = _attn(
        weight_packed=packed,
        weight_scale=torch.rand(OUT_SIZE, HIDDEN // 16, dtype=ACT_DTYPE),
    )
    assert divmod(32 * packed.shape[1], HIDDEN) == (16, 0)  # the trap, pinned
    with pytest.raises(ValueError, match="torch.uint8 packed weight"):
        dequant_kv_slice(attn, ACT_DTYPE)


BLOCK = 128


def _block_scaled(rows=OUT_SIZE, cols=HIDDEN, block=(BLOCK, BLOCK)):
    weight = torch.randn(rows, cols).to(torch.float8_e4m3fn)
    scale = torch.rand(-(-rows // block[0]), -(-cols // block[1])) * 0.05 + 0.001
    return weight, scale


def test_block_scaled_fp8_matches_expanding_the_whole_scale():
    """Block-scaled FP8 stores the scale as `weight_scale_inv` over 2-D tiles.

    Real geometry: the drafter's fused qkv is 4096 + 2*1024 rows over 5120
    features, so q ends exactly on a 128-row block boundary and the K/V rows can
    be taken before the scale is expanded.
    """
    q_size, kv_size, hidden = 4096, 1024, 5120
    rows = q_size + 2 * kv_size
    weight, scale = _block_scaled(rows, hidden)
    attn = SimpleNamespace(
        qkv_proj=SimpleNamespace(
            weight=weight,
            weight_scale_inv=scale,
            weight_block_size=[BLOCK, BLOCK],
            input_size=hidden,
        ),
        q_size=q_size,
    )

    out = dequant_kv_slice(attn, ACT_DTYPE)

    full = scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[:rows, :hidden]
    expected = (weight.to(torch.float32) * full)[q_size:].to(ACT_DTYPE)
    assert torch.equal(out, expected)


def test_block_scaled_fp8_refuses_a_cut_inside_a_block():
    """A q_size off the block grid would hand K part of q's scale."""
    rows, hidden, q_size = 6144, 5120, 4000  # 4000 % 128 != 0
    weight, scale = _block_scaled(rows, hidden)
    attn = SimpleNamespace(
        qkv_proj=SimpleNamespace(
            weight=weight,
            weight_scale_inv=scale,
            weight_block_size=[BLOCK, BLOCK],
            input_size=hidden,
        ),
        q_size=q_size,
    )
    with pytest.raises(ValueError, match="falls inside a 128-row block"):
        dequant_kv_slice(attn, ACT_DTYPE)


def test_block_size_that_does_not_explain_the_scale_is_rejected():
    """Which axis each block size describes is a convention, so it is checked."""
    weight, scale = _block_scaled(6144, 5120)
    attn = SimpleNamespace(
        qkv_proj=SimpleNamespace(
            weight=weight,
            weight_scale_inv=scale,
            weight_block_size=[64, BLOCK],  # would imply twice as many scale rows
            input_size=5120,
        ),
        q_size=4096,
    )
    with pytest.raises(ValueError, match="implies a scale of"):
        dequant_kv_slice(attn, ACT_DTYPE)


def test_non_square_blocks_pin_which_axis_is_which():
    """Real checkpoints use 128x128, where swapping the two block sizes is
    invisible. A non-square block is the only case that pins the convention."""
    rows, hidden, q_size = 6144, 5120, 4096
    weight, scale = _block_scaled(rows, hidden, block=(BLOCK, 64))
    assert tuple(scale.shape) == (48, 80)
    attn = SimpleNamespace(
        qkv_proj=SimpleNamespace(
            weight=weight,
            weight_scale_inv=scale,
            weight_block_size=[BLOCK, 64],
            input_size=hidden,
        ),
        q_size=q_size,
    )

    out = dequant_kv_slice(attn, ACT_DTYPE)

    full = scale.repeat_interleave(BLOCK, 0).repeat_interleave(64, 1)[:rows, :hidden]
    expected = (weight.to(torch.float32) * full)[q_size:].to(ACT_DTYPE)
    assert torch.equal(out, expected)
