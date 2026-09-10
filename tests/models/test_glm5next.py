# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the GLM-5.3-Flash (glm5next) Quark MXFP4 loading."""

import torch

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.quark.quark import (
    QuarkConfig,
    QuarkW8A8Fp8PerBlock,
)
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForConditionalGeneration,
    _dequant_fp8_block,
    _try_load_fp8_attn_proj,
)

_MXFP4_WEIGHT = {
    "dtype": "fp4",
    "qscheme": "per_group",
    "group_size": 32,
    "scale_format": "e8m0",
    "is_dynamic": False,
}
_BLOCK_FP8_WEIGHT = {
    "dtype": "fp8_e4m3",
    "qscheme": "per_block",
    "is_dynamic": False,
    "block_size": [128, 128],
    "symmetric": True,
}
_BLOCK_FP8_INPUT = {
    "dtype": "fp8_e4m3",
    "qscheme": "per_group",
    "is_dynamic": True,
    "group_size": 128,
    "symmetric": True,
}
_GATE_UP = "language_model.model.layers.0.mlp.gate_up_proj"


def _mixed_precision_config() -> QuarkConfig:
    """Global MXFP4 with the dense-MLP gate/up shards marked block-FP8."""
    fp8 = {"weight": _BLOCK_FP8_WEIGHT, "input_tensors": _BLOCK_FP8_INPUT}
    return QuarkConfig(
        {
            "global_quant_config": {"weight": _MXFP4_WEIGHT, "input_tensors": None},
            "layer_type_quant_config": {},
            "layer_quant_config": {"*mlp.gate_proj": fp8, "*mlp.up_proj": fp8},
            "exclude": [],
        }
    )


class _RecordingParam:
    """Stand-in for a BF16 target param that records what gets loaded."""

    def __init__(self):
        self.loaded_weight: torch.Tensor | None = None
        self.loaded_shard: object = "unset"

    def weight_loader(self, param, loaded_weight, shard_id=None):
        self.loaded_weight = loaded_weight
        self.loaded_shard = shard_id


def _block_fp8(out_dim: int, in_dim: int, block: int = 128):
    """Return a (fp8 weight, f32 per-block scale) pair."""
    weight = (torch.randn(out_dim, in_dim) * 0.1).to(torch.float8_e4m3fn)
    scale = torch.rand(out_dim // block, in_dim // block, dtype=torch.float32) + 0.5
    return weight, scale


def test_gate_up_proj_mapping_is_genuine_fusion():
    mapping = Glm5NextForConditionalGeneration.packed_modules_mapping
    assert mapping["gate_up_proj"] == ["gate_proj", "up_proj"]


def test_genuine_fusion_resolves_gate_up_to_block_fp8():
    # The fused module must expand to its real shard names so each resolves to the
    # per-layer block-FP8 entry (rather than the global MXFP4 scheme).
    config = _mixed_precision_config()
    config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    _, _, scheme_cls = config.get_scheme_cls(LinearBase, _GATE_UP)
    assert scheme_cls is QuarkW8A8Fp8PerBlock


def test_attn_loader_accepts_quark_weight_scale():
    # Regression for KeyError on '...kv_a_proj_with_mqa.weight_scale': the fused
    # q_a/kv_a projection is kept BF16 and dequantized on load; the Quark scale
    # name must be recognized and routed to the correct fused shard.
    prefix = "layers.3.self_attn"
    target = _RecordingParam()
    params_dict = {f"{prefix}.fused_qkv_a_proj.weight": target}
    buf: dict = {}
    loaded: set = set()

    weight, scale = _block_fp8(256, 256)
    # fp8 weight arrives first -> buffered, nothing loaded yet.
    assert (
        _try_load_fp8_attn_proj(
            f"{prefix}.kv_a_proj_with_mqa.weight", weight, buf, params_dict, loaded, 0
        )
        is True
    )
    assert target.loaded_weight is None
    # Quark-named scale completes the pair -> dequantize + load.
    assert (
        _try_load_fp8_attn_proj(
            f"{prefix}.kv_a_proj_with_mqa.weight_scale",
            scale,
            buf,
            params_dict,
            loaded,
            0,
        )
        is True
    )
    assert target.loaded_weight is not None
    assert target.loaded_weight.dtype == torch.bfloat16
    assert target.loaded_shard == 1  # kv_a is shard 1 of fused_qkv_a_proj
    assert torch.equal(target.loaded_weight, _dequant_fp8_block(weight, scale, 128))
    assert f"{prefix}.fused_qkv_a_proj.weight" in loaded


def test_attn_loader_accepts_deepseek_weight_scale_inv():
    # DeepSeek scale name keeps working unchanged
    prefix = "layers.7.self_attn"
    target = _RecordingParam()
    params_dict = {f"{prefix}.o_proj.weight": target}
    buf: dict = {}
    loaded: set = set()

    weight, scale = _block_fp8(128, 256)
    _try_load_fp8_attn_proj(
        f"{prefix}.o_proj.weight", weight, buf, params_dict, loaded, 0
    )
    assert (
        _try_load_fp8_attn_proj(
            f"{prefix}.o_proj.weight_scale_inv", scale, buf, params_dict, loaded, 0
        )
        is True
    )
    assert target.loaded_shard is None  # o_proj is a direct (non-fused) proj
    assert torch.equal(target.loaded_weight, _dequant_fp8_block(weight, scale, 128))
