# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 MoE weights as a source for online block-FP8 requantization."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.quantization.online.fp8 as online_fp8
from tests.quantization.utils import quant_config_args
from tests.single_rank_group import SingleRankGroup, override_parallel_groups
from vllm.config.kernel import KernelConfig
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config, Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils import mxfp4_utils
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequantize_mxfp4_moe_weights,
    dequantize_mxfp4_weight,
)
from vllm.platforms import current_platform

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _reference(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode MXFP4 in float64, one nibble at a time."""
    raw = weight.view(torch.uint8).long()
    codes = torch.empty(*raw.shape[:-1], raw.shape[-1] * 2, dtype=torch.long)
    codes[..., 0::2] = raw & 0xF
    codes[..., 1::2] = raw >> 4
    lut = torch.tensor(E2M1, dtype=torch.float64)
    values = torch.where(codes >= 8, -lut[codes & 7], lut[codes & 7])
    factors = torch.pow(2.0, scale.view(torch.uint8).double() - 127)
    return values * factors.repeat_interleave(32, dim=-1)


def _mxfp4(rows: int, cols: int, low: int, high: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randint(
        0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator
    )
    scale = torch.randint(
        low, high + 1, (rows, cols // 32), dtype=torch.uint8, generator=generator
    )
    return weight, scale


def test_dequantize_mxfp4_weight_is_exact_with_positive_zero() -> None:
    # Scales above 2**123 would overflow bf16 for the largest codes.
    weight, scale = _mxfp4(64, 256, 0, 250)
    expected = _reference(weight, scale)

    actual = dequantize_mxfp4_weight(weight, scale, torch.float32)

    assert torch.equal(actual.double(), expected)
    assert not torch.signbit(actual[actual == 0]).any()
    bf16 = dequantize_mxfp4_weight(weight, scale, torch.bfloat16)
    assert torch.equal(bf16.double(), expected)


def test_dequantize_mxfp4_weight_rejects_nan_scale() -> None:
    weight, scale = _mxfp4(32, 64, 100, 130)
    scale[3, 1] = 255
    with pytest.raises(ValueError, match="NaN"):
        dequantize_mxfp4_weight(weight, scale, torch.bfloat16)


@pytest.mark.parametrize("empty_partition", [False, True])
def test_repack_fnuz_scales_follow_checkpoint(empty_partition: bool) -> None:
    # The largest source scale belongs to a group with no nonzero payload.
    packed = torch.full((1, 128, 64), 0x11, dtype=torch.uint8)
    scales = torch.full((1, 128, 4), 127, dtype=torch.uint8)
    packed[:, :, :16] = 0
    scales[:, :, 0] = 131
    if empty_partition:
        packed.zero_()

    codes, factors = mxfp4_utils.repack_mxfp4_fnuz_checkpoint_blocks(
        packed, scales, empty_partition=empty_partition
    )

    assert codes.dtype == torch.float8_e4m3fnuz
    assert factors.item() == (2.0 if empty_partition else 0.5)
    assert torch.equal(codes.double() * factors.item(), _reference(packed, scales))


def test_dequantize_mxfp4_moe_weights_covers_every_expert(monkeypatch) -> None:
    monkeypatch.setattr(mxfp4_utils, "_DEQUANT_CHUNK_BYTES", 1)
    w13, w13_scale = _mxfp4(3 * 2 * 64, 128, 110, 130)
    w2, w2_scale = _mxfp4(3 * 128, 64, 110, 130, seed=1)
    w13, w13_scale = w13.view(3, 128, 64), w13_scale.view(3, 128, 4)
    w2, w2_scale = w2.view(3, 128, 32), w2_scale.view(3, 128, 2)

    chunks = list(
        dequantize_mxfp4_moe_weights(w13, w13_scale, w2, w2_scale, torch.bfloat16)
    )

    assert [c[0] for c in chunks] == [slice(0, 1), slice(1, 2), slice(2, 3)]
    assert torch.equal(
        torch.cat([c[1] for c in chunks]).double(), _reference(w13, w13_scale)
    )
    assert torch.equal(
        torch.cat([c[2] for c in chunks]).double(), _reference(w2, w2_scale)
    )


@pytest.mark.parametrize(
    "fp8_dtype, fp8_max",
    [(torch.float8_e4m3fn, 448.0), (torch.float8_e4m3fnuz, 224.0)],
)
def test_power_of_two_block_fp8_keeps_mxfp4_exact(
    monkeypatch, fp8_dtype: torch.dtype, fp8_max: float
) -> None:
    monkeypatch.setattr(
        online_fp8, "current_platform", SimpleNamespace(fp8_dtype=lambda: fp8_dtype)
    )
    monkeypatch.setattr(online_fp8, "get_fp8_min_max", lambda: (-fp8_max, fp8_max))
    # A TP shard of 288 channels padded to 384, with E8M0 spanning 14 per block.
    weight, scale = _mxfp4(288, 256, 113, 127)
    weight[0, 0] = 0x77  # 6 * 2**0 sets the block maximum
    scale[0, 0] = 127
    weight[1, 0] = 0x01  # 0.5 * 2**-14, the smallest value
    scale[1, 0] = 113
    values = torch.zeros(384, 256, dtype=torch.bfloat16)
    values[:288] = dequantize_mxfp4_weight(weight, scale, torch.bfloat16)

    quantized, block_scale = online_fp8._per_block_cast_to_fp8_pow2(values, [128, 128])

    assert quantized.dtype == fp8_dtype and block_scale.shape == (3, 2)
    mantissa, _ = torch.frexp(block_scale)
    assert torch.all(mantissa == 0.5)
    restored = (
        quantized.float().view(3, 128, 2, 128) * block_scale[:, None, :, None]
    ).view(384, 256)
    assert torch.equal(restored, values.float())


@pytest.mark.parametrize("dispatcher", [False, True])
@pytest.mark.parametrize("stage1", [False, True])
def test_block_fp8_silu_limit_requires_dispatch_and_ck_stage1(
    monkeypatch, dispatcher, stage1
):
    import sys
    from types import ModuleType
    from typing import Any

    from vllm._aiter_ops import rocm_aiter_ops

    fused: Any = ModuleType("aiter.fused_moe")
    # Older AITER already accepts the argument on the outer fused_moe.
    fused.fused_moe = lambda swiglu_limit: None
    if dispatcher:
        fused.swiglu_limit_tag = lambda limit: str(limit)
    ck: Any = ModuleType("aiter.ops.moe_op")
    ck.ck_moe_stage1_fwd = (lambda swiglu_limit: None) if stage1 else (lambda: None)
    monkeypatch.setitem(sys.modules, fused.__name__, fused)
    monkeypatch.setitem(sys.modules, ck.__name__, ck)
    probe = rocm_aiter_ops.supports_fp8_block_silu_limit
    # Exercise each installed interface independently of the cached result.
    assert probe.__wrapped__() is (dispatcher and stage1)


# 192 is not a multiple of the 128 block: the shard is zero-padded to 256.
@pytest.mark.parametrize("intermediate_size", [256, 192])
@pytest.mark.skip_global_cleanup
@override_parallel_groups(SingleRankGroup())
def test_online_block_fp8_moe_requantizes_mxfp4_checkpoint_exactly(
    intermediate_size: int, default_vllm_config
) -> None:
    """MXFP4 experts targeted by `fp8_per_block` keep every weight value when the
    E8M0 scales of a 128x128 block span at most 2**14."""
    from tests.quantization.test_online import _materialize_requantization_source

    # Triton keeps the natural weight layout, so values compare directly.
    default_vllm_config.kernel_config = KernelConfig(moe_backend="triton")
    quant_config = Mxfp4Config()
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        quant_config_args(moe="fp8_per_block")
    )
    num_experts, hidden_size = 3, 256
    torch.manual_seed(0)
    with torch.device(current_platform.device_type):
        layer = FusedMoEFactory(
            num_experts=num_experts,
            top_k=2,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="model.layers.0.mlp.experts",
        ).routed_experts
        method = layer.quant_method
        assert isinstance(method, online_fp8.Fp8PerBlockOnlineMoEMethod)
        assert isinstance(method.requantization_source, Mxfp4MoEMethod)
        _materialize_requantization_source(layer)

        source = {}
        for name in ("w13", "w2"):
            weight = getattr(layer, f"{name}_weight")
            scale = getattr(layer, f"{name}_weight_scale")
            weight.data.copy_(torch.randint(0, 256, weight.shape, dtype=torch.uint8))
            scale.data.copy_(torch.randint(113, 128, scale.shape, dtype=torch.uint8))
            source[name] = (weight.data.cpu(), scale.data.cpu())
        method.process_weights_after_loading(layer)

    # The source parameters are released once requantized.
    assert not hasattr(layer, "w13_weight_scale")
    assert not hasattr(layer, "w2_weight_scale")
    for name in ("w13", "w2"):
        weight = getattr(layer, f"{name}_weight")
        assert weight.dtype == current_platform.fp8_dtype()
        scale = getattr(layer, f"{name}_weight_scale_inv").double().cpu()
        scale = scale.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
        restored = weight.double().cpu() * scale
        if name == "w13":
            restored = restored.unflatten(1, (2, -1))
        assert not restored[:, :, intermediate_size:].any()
        reference = _reference(*source[name])
        restored = restored[:, :, :intermediate_size].reshape(reference.shape)
        assert torch.equal(restored, reference)
