# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for loading ModelOpt mixed-precision weights into BF16 modules."""

from typing import Any

import pytest
import torch

from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.models.glm5next.common.modelopt_dequant import (
    alias_mtp_quantized_layers,
    dequantize_bf16_targets,
    dequantize_to_bf16,
    stacked_bf16_target_fn,
)

pytestmark = pytest.mark.cpu_test

_E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)


def _fp8(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float8_e4m3fn)


def test_mxfp8_matches_reference():
    torch.manual_seed(0)
    w = _fp8(torch.randn(4, 64))
    scale = torch.randint(120, 135, (4, 2), dtype=torch.uint8)
    ref = w.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, dim=1)
    out = dequantize_to_bf16({"weight": w, "weight_scale": scale})
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref.to(torch.bfloat16), atol=0, rtol=0)


def test_mxfp8_rejects_nan_scale():
    w = _fp8(torch.ones(1, 32))
    with pytest.raises(ValueError, match="NaN"):
        dequantize_to_bf16(
            {"weight": w, "weight_scale": torch.full((1, 1), 255, dtype=torch.uint8)}
        )


@pytest.mark.parametrize("scale_shape", [(2, 2), (4,), (2, 2, 1)])
def test_fp8_block_with_partial_last_block(scale_shape):
    # 200 x 130: both dims end with a partial 128-block, as kv_a_proj_with_mqa does.
    torch.manual_seed(0)
    w = _fp8(torch.randn(200, 130))
    blocks = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    full = blocks.repeat_interleave(128, 0).repeat_interleave(128, 1)[:200, :130]
    out = dequantize_to_bf16({"weight": w, "weight_scale": blocks.reshape(scale_shape)})
    torch.testing.assert_close(
        out, (w.float() * full).to(torch.bfloat16), atol=0, rtol=0
    )


def test_nvfp4_low_nibble_first():
    codes = torch.randint(0, 16, (3, 32))
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    scale = _fp8(torch.tensor([[1.0, 2.0], [0.5, 4.0], [1.5, 1.0]]))
    gs = torch.tensor(0.25)
    ref = _E2M1[codes] * scale.float().repeat_interleave(16, dim=1) * gs
    out = dequantize_to_bf16(
        {"weight": packed, "weight_scale": scale, "weight_scale_2": gs}
    )
    torch.testing.assert_close(out, ref.to(torch.bfloat16), atol=0, rtol=0)


def test_stream_buffers_out_of_order_and_passes_others_through():
    w = _fp8(torch.ones(2, 32))
    s = torch.full((2, 1), 127, dtype=torch.uint8)
    stream = [
        ("a.weight_scale", s),
        ("x.bias", torch.zeros(3)),
        ("a.weight", w),
        ("a.input_scale", torch.ones(1)),  # dropped: the BF16 module has no use for it
        ("bf.weight", torch.zeros(2, 2, dtype=torch.bfloat16)),
        ("native.weight", w),  # quantized but not a BF16 target: untouched
        ("native.weight_scale", s),
    ]
    out = list(dequantize_bf16_targets(stream, lambda module: module != "native"))
    assert [n for n, _ in out] == [
        "x.bias",
        "a.weight",
        "bf.weight",
        "native.weight",
        "native.weight_scale",
    ]
    assert out[1][1].dtype == torch.bfloat16
    assert torch.equal(out[1][1], torch.ones(2, 32, dtype=torch.bfloat16))
    assert out[3][1].dtype == torch.float8_e4m3fn


def test_stream_keeps_loader_kwargs_on_passthrough():
    item: tuple[Any, ...] = ("x.bias", torch.zeros(1), {"shard_id": 0})
    assert list(dequantize_bf16_targets([item], lambda module: True)) == [item]


def test_stream_raises_on_incomplete_module():
    with pytest.raises(ValueError, match="incomplete"):
        list(
            dequantize_bf16_targets(
                [("m.weight", _fp8(torch.ones(1, 32)))], lambda module: True
            )
        )


def test_stacked_target_detection():
    params = {
        "layers.0.self_attn.in_proj_qkvbfg_a.weight": None,
        "layers.0.mlp.gate_up_proj.weight": None,
        "layers.0.mlp.gate_up_proj.weight_scale": None,
        "layers.0.self_attn.o_proj.weight": None,
    }
    mapping = [
        (".in_proj_qkvbfg_a", ".q_proj", 0),
        (".gate_up_proj", ".gate_proj", 0),
    ]
    is_bf16 = stacked_bf16_target_fn(params, mapping)
    assert is_bf16("layers.0.self_attn.q_proj")  # fused BF16 target
    assert is_bf16("layers.0.self_attn.o_proj")  # direct BF16 target
    assert not is_bf16("layers.0.mlp.gate_proj")  # quantized target
    assert not is_bf16("layers.0.mlp.experts.3.gate_proj")  # no own param


@pytest.mark.parametrize(
    "text_prefix", ["model.language_model.layers.", "language_model.model.layers."]
)
def test_mtp_aliases(text_prefix):
    # The second form is what the quant config holds after the target model's
    # hf_to_vllm_mapper has been applied to it.
    cfg = ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    f"{text_prefix}45.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
                    f"{text_prefix}45.eh_proj": {"quant_algo": "MXFP8"},
                    f"{text_prefix}3.mlp.experts": {"quant_algo": "NVFP4"},
                },
            }
        }
    )
    alias_mtp_quantized_layers(cfg, 45, 1)
    assert cfg.quantized_layers["model.layers.45.mlp.experts"] == {
        "quant_algo": "W4A16_NVFP4"
    }
    assert cfg.quantized_layers["model.layers.45.eh_proj"] == {"quant_algo": "MXFP8"}
    assert "model.layers.3.mlp.experts" not in cfg.quantized_layers
