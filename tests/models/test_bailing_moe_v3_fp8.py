# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import pytest
import torch

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.models import bailing_moe_v3


class _FakeQuantConfig:
    def __init__(
        self,
        *,
        name: str = "fp8",
        serialized: bool = True,
        block_size: list[int] | None = None,
        ignored_layers: list[str] | None = None,
    ) -> None:
        self.name = name
        self.is_checkpoint_fp8_serialized = serialized
        self.weight_block_size = block_size
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping: dict[str, list[str]] = {}

    def get_name(self) -> str:
        return self.name


def _quant_config(**kwargs) -> QuantizationConfig:
    return cast(QuantizationConfig, _FakeQuantConfig(**kwargs))


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("model.layers.5.self_attn.kv_b_proj", True),
        ("model.layers.5.self_attn.g_proj", True),
        ("model.layers.6.self_attn.g_proj", False),
        ("model.layers.0.self_attn.b_proj", True),
        ("model.layers.0.self_attn.q_proj", False),
    ],
)
def test_ling_block_fp8_exclusion_aliases(prefix: str, expected: bool):
    quant_config = _quant_config(
        block_size=[128, 128],
        ignored_layers=[
            "kv_b_proj",
            "b_proj",
            "5.attention.g_proj",
        ],
    )

    assert bailing_moe_v3._is_fp8_module_excluded(quant_config, prefix) is expected


def test_ling_block_fp8_uses_checkpoint_modules_to_not_convert():
    quant_config = Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["b_proj", "5.attention.g_proj"],
        }
    )

    assert bailing_moe_v3._is_fp8_module_excluded(
        quant_config, "model.layers.0.self_attn.b_proj"
    )
    assert bailing_moe_v3._is_fp8_module_excluded(
        quant_config, "model.layers.5.self_attn.g_proj"
    )


def test_ling_fp8_exclusions_do_not_change_other_precision_paths():
    quant_config = _quant_config(
        name="some_other_quantization",
        block_size=[128, 128],
        ignored_layers=["kv_b_proj"],
    )

    assert (
        bailing_moe_v3._get_linear_quant_config(
            quant_config, "model.layers.5.self_attn.kv_b_proj"
        )
        is quant_config
    )
    assert (
        bailing_moe_v3._get_linear_quant_config(
            None, "model.layers.5.self_attn.kv_b_proj"
        )
        is None
    )


@pytest.mark.parametrize(
    ("serialized", "block_size"),
    [(False, [128, 128]), (True, None)],
)
def test_ling_kda_split_requires_serialized_block_fp8(
    serialized: bool,
    block_size: list[int] | None,
):
    quant_config = _quant_config(
        serialized=serialized,
        block_size=block_size,
        ignored_layers=["b_proj"],
    )

    assert not bailing_moe_v3._is_fp8_module_excluded(
        quant_config, "model.layers.0.self_attn.b_proj"
    )


def test_ling_kda_keeps_bf16_and_fp8_packed_mappings():
    packed_mapping = bailing_moe_v3.BailingMoeV3ForCausalLM.packed_modules_mapping

    assert packed_mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
    assert packed_mapping["qkvb_proj"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
    ]


@pytest.mark.parametrize(
    ("tp_size", "expected"),
    [(1, 768), (2, 768), (4, 1024), (8, 1024)],
)
def test_ling_block_fp8_shared_expert_padding(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    expected: int,
):
    monkeypatch.setattr(
        bailing_moe_v3,
        "get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )
    quant_config = _quant_config(block_size=[128, 128])
    prefix = "model.layers.2.mlp.shared_experts"

    assert (
        bailing_moe_v3._get_block_fp8_mlp_padded_intermediate_size(
            quant_config, 768, prefix
        )
        == expected
    )
    assert (
        bailing_moe_v3._get_block_fp8_mlp_padded_intermediate_size(None, 768, prefix)
        == 768
    )
    online_quant_config = _quant_config(
        serialized=False,
        block_size=[128, 128],
    )
    assert (
        bailing_moe_v3._get_block_fp8_mlp_padded_intermediate_size(
            online_quant_config, 768, prefix
        )
        == 768
    )


def test_ling_shared_expert_padding_requires_an_fp8_projection(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        bailing_moe_v3,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    prefix = "model.layers.2.mlp.shared_experts"
    quant_config = _quant_config(
        block_size=[128, 128],
        ignored_layers=["gate_up_proj", "down_proj"],
    )

    assert (
        bailing_moe_v3._get_block_fp8_mlp_padded_intermediate_size(
            quant_config, 768, prefix
        )
        == 768
    )

    quant_config = _quant_config(
        block_size=[128, 128],
        ignored_layers=["gate_up_proj"],
    )
    assert (
        bailing_moe_v3._get_block_fp8_mlp_padded_intermediate_size(
            quant_config, 768, prefix
        )
        == 1024
    )

    quant_config = _quant_config(
        block_size=[128, 128],
        ignored_layers=["down_proj"],
    )
    assert (
        bailing_moe_v3._get_block_fp8_mlp_padded_intermediate_size(
            quant_config, 768, prefix
        )
        == 1024
    )


@pytest.mark.parametrize(
    ("name", "shape", "expected_shape", "dim"),
    [
        ("shared_experts.gate_proj.weight", (768, 2), (1024, 2), 0),
        ("shared_experts.up_proj.weight_scale_inv", (6, 2), (8, 2), 0),
        ("shared_experts.down_proj.weight", (2, 768), (2, 1024), 1),
        ("shared_experts.down_proj.weight_scale_inv", (2, 6), (2, 8), 1),
    ],
)
def test_ling_pad_block_fp8_shared_expert_checkpoint_tensor(
    name: str,
    shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
    dim: int,
):
    quant_config = _quant_config(block_size=[128, 128])
    loaded_weight = torch.ones(shape)

    padded = bailing_moe_v3._pad_block_fp8_mlp_checkpoint_tensor(
        quant_config,
        name,
        loaded_weight,
        intermediate_size=768,
        padded_intermediate_size=1024,
    )

    assert padded.shape == expected_shape
    assert torch.equal(padded.narrow(dim, 0, shape[dim]), loaded_weight)
    assert (
        torch.count_nonzero(
            padded.narrow(dim, shape[dim], expected_shape[dim] - shape[dim])
        )
        == 0
    )


@pytest.mark.parametrize(
    ("suffix", "shard_size", "padded_shard_size"),
    [("weight", 768, 1024), ("weight_scale_inv", 6, 8)],
)
def test_ling_pad_fused_gate_up_checkpoint_tensor_between_logical_shards(
    suffix: str,
    shard_size: int,
    padded_shard_size: int,
):
    quant_config = _quant_config(block_size=[128, 128])
    loaded_weight = torch.cat(
        [torch.ones(shard_size, 2), torch.full((shard_size, 2), 2.0)]
    )

    padded = bailing_moe_v3._pad_block_fp8_mlp_checkpoint_tensor(
        quant_config,
        f"shared_experts.gate_up_proj.{suffix}",
        loaded_weight,
        intermediate_size=768,
        padded_intermediate_size=1024,
    )

    assert padded.shape == (2 * padded_shard_size, 2)
    assert torch.equal(padded[:shard_size], loaded_weight[:shard_size])
    assert torch.count_nonzero(padded[shard_size:padded_shard_size]) == 0
    assert torch.equal(
        padded[padded_shard_size : padded_shard_size + shard_size],
        loaded_weight[shard_size:],
    )
    assert torch.count_nonzero(padded[padded_shard_size + shard_size :]) == 0


def test_ling_shared_expert_checkpoint_padding_handles_mtp_layer_name(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        bailing_moe_v3,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    quant_config = _quant_config(block_size=[128, 128])

    class _FakeBailingConfig:
        moe_shared_expert_intermediate_size = 768
        num_shared_experts = 1

    loaded_scale = torch.ones(6, 2)
    padded = bailing_moe_v3._maybe_pad_block_fp8_shared_expert_checkpoint_tensor(
        quant_config,
        _FakeBailingConfig(),  # type: ignore[arg-type]
        "model.layers.42.mlp.shared_experts.gate_proj.weight_scale_inv",
        loaded_scale,
    )

    assert padded.shape == (8, 2)
    assert torch.equal(padded[:6], loaded_scale)
    assert torch.count_nonzero(padded[6:]) == 0


def test_ling_shared_expert_padding_keeps_down_bias_unchanged():
    quant_config = _quant_config(block_size=[128, 128])
    loaded_bias = torch.ones(2)

    padded = bailing_moe_v3._pad_block_fp8_mlp_checkpoint_tensor(
        quant_config,
        "shared_experts.down_proj.bias",
        loaded_bias,
        intermediate_size=768,
        padded_intermediate_size=1024,
    )

    assert padded is loaded_bias


def test_ling_shared_expert_padding_rejects_wrong_checkpoint_shape():
    quant_config = _quant_config(block_size=[128, 128])

    with pytest.raises(ValueError, match="expected each logical intermediate"):
        bailing_moe_v3._pad_block_fp8_mlp_checkpoint_tensor(
            quant_config,
            "shared_experts.gate_proj.weight",
            torch.ones(640, 2),
            intermediate_size=768,
            padded_intermediate_size=1024,
        )


def test_ling_shared_expert_padding_accepts_already_padded_checkpoint():
    quant_config = _quant_config(block_size=[128, 128])
    loaded_weight = torch.ones(1024, 2)

    padded = bailing_moe_v3._pad_block_fp8_mlp_checkpoint_tensor(
        quant_config,
        "shared_experts.gate_proj.weight",
        loaded_weight,
        intermediate_size=768,
        padded_intermediate_size=1024,
    )

    assert padded is loaded_weight
