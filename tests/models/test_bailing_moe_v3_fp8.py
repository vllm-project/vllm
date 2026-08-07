# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import fp8 as fp8_quantization
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models import bailing_moe_v3, bailing_moe_v3_mtp


def _block_fp8_config(
    ignored_layers: list[str] | None = None,
) -> Fp8Config:
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        ignored_layers=ignored_layers,
    )
    bailing_moe_v3._configure_ling_fp8_quant_config(quant_config)
    return quant_config


class _StubLinear(LinearBase):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)


@pytest.mark.parametrize(
    "model_class",
    [
        bailing_moe_v3.BailingMoeV3ForCausalLM,
        bailing_moe_v3_mtp.BailingMoeV3MTPModel,
    ],
)
def test_ling_block_fp8_maps_checkpoint_exclusions(model_class):
    assert (
        model_class.hf_to_vllm_mapper._map_name(
            "model.layers.5.attention.g_proj.weight"
        )
        == "model.layers.5.self_attn.g_proj.weight"
    )

    quant_config = Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["b_proj", "5.attention.g_proj"],
        }
    )
    configure_quant_config(quant_config, model_class)

    assert quant_config.ignored_layers == ["b_proj", "5.self_attn.g_proj"]
    assert quant_config.ignored_layers_match_mode == "exact"

    bailing_moe_v3._configure_ling_fp8_quant_config(quant_config)
    assert quant_config.ignored_layers_match_mode == "suffix"


def test_ling_block_fp8_dispatches_excluded_linear(
    monkeypatch: pytest.MonkeyPatch,
):
    quant_config = _block_fp8_config(ignored_layers=["b_proj"])

    class _StubFp8LinearMethod:
        pass

    fp8_method = _StubFp8LinearMethod()
    linear = _StubLinear()
    # Keep this dispatch test independent of platform-specific FP8 kernels.
    monkeypatch.setattr(
        fp8_quantization,
        "Fp8LinearMethod",
        lambda _: fp8_method,
    )
    assert isinstance(
        quant_config.get_quant_method(linear, "model.layers.0.self_attn.b_proj"),
        UnquantizedLinearMethod,
    )
    assert (
        quant_config.get_quant_method(linear, "model.layers.0.self_attn.q_b_proj")
        is fp8_method
    )


@pytest.mark.parametrize(
    "quant_config",
    [
        pytest.param(None, id="bf16"),
        pytest.param(Fp8Config(), id="online-fp8"),
        pytest.param(
            Fp8Config(is_checkpoint_fp8_serialized=True),
            id="serialized-per-tensor-fp8",
        ),
    ],
)
def test_ling_non_block_fp8_paths_unchanged(quant_config: Fp8Config | None):
    bailing_moe_v3._configure_ling_fp8_quant_config(quant_config)
    if quant_config is not None:
        assert quant_config.ignored_layers_match_mode == "exact"

    assert not bailing_moe_v3._is_fp8_module_excluded(
        quant_config, "model.layers.0.self_attn.b_proj"
    )
    assert (
        bailing_moe_v3._get_block_fp8_mlp_padded_intermediate_size(
            quant_config, 768, "model.layers.2.mlp.shared_experts"
        )
        == 768
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
    quant_config = _block_fp8_config()
    prefix = "model.layers.2.mlp.shared_experts"

    assert (
        bailing_moe_v3._get_block_fp8_mlp_padded_intermediate_size(
            quant_config, 768, prefix
        )
        == expected
    )


@pytest.mark.parametrize(
    ("ignored_layers", "expected"),
    [
        (["gate_up_proj", "down_proj"], 768),
        (["gate_up_proj"], 1024),
        (["down_proj"], 1024),
    ],
)
def test_ling_shared_expert_padding_requires_an_fp8_projection(
    monkeypatch: pytest.MonkeyPatch,
    ignored_layers: list[str],
    expected: int,
):
    monkeypatch.setattr(
        bailing_moe_v3,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    prefix = "model.layers.2.mlp.shared_experts"
    quant_config = _block_fp8_config(ignored_layers=ignored_layers)
    assert (
        bailing_moe_v3._get_block_fp8_mlp_padded_intermediate_size(
            quant_config, 768, prefix
        )
        == expected
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
    quant_config = _block_fp8_config()
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
    quant_config = _block_fp8_config()
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
    quant_config = _block_fp8_config()

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


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("shared_experts.down_proj.bias", (2,)),
        ("shared_experts.gate_proj.weight", (1024, 2)),
    ],
)
def test_ling_shared_expert_padding_keeps_unchanged_tensors(
    name: str,
    shape: tuple[int, ...],
):
    quant_config = _block_fp8_config()
    loaded_weight = torch.ones(shape)

    padded = bailing_moe_v3._pad_block_fp8_mlp_checkpoint_tensor(
        quant_config,
        name,
        loaded_weight,
        intermediate_size=768,
        padded_intermediate_size=1024,
    )

    assert padded is loaded_weight


def test_ling_shared_expert_padding_rejects_wrong_checkpoint_shape():
    quant_config = _block_fp8_config()

    with pytest.raises(ValueError, match="expected each logical intermediate"):
        bailing_moe_v3._pad_block_fp8_mlp_checkpoint_tensor(
            quant_config,
            "shared_experts.gate_proj.weight",
            torch.ones(640, 2),
            intermediate_size=768,
            padded_intermediate_size=1024,
        )
