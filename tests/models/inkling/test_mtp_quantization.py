# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.models.inkling import InklingMTP
from vllm.models.inkling.configs import InklingModelConfig


def _checkpoint_projections(depth: int) -> list[str]:
    prefix = f"model.mtp.layers.{depth}"
    return [
        f"{prefix}.input_proj",
        *[
            f"{prefix}.transformer_block.attn.{name}"
            for name in ("wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud")
        ],
        f"{prefix}.transformer_block.mlp.w13_dn",
        f"{prefix}.transformer_block.mlp.w2_md",
    ]


@pytest.mark.parametrize("mode", ["bf16", "class", "names", "ignored"])
def test_mtp_builds_quantized_or_excluded_linears(
    mode, dist_init, default_vllm_config
) -> None:
    """The saved config controls every projection, including fused attention."""
    paths = _checkpoint_projections(0) + _checkpoint_projections(1)
    quant_config = None
    if mode != "bf16":
        quant_config = CompressedTensorsConfig.from_config(
            {
                "format": "float-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": paths if mode == "names" else ["Linear"],
                        "weights": {
                            "num_bits": 8,
                            "type": "float",
                            "strategy": "tensor",
                            "symmetric": True,
                            "dynamic": False,
                        },
                    }
                },
                "ignore": paths if mode == "ignored" else [],
            }
        )
        configure_quant_config(quant_config, InklingMTP)

    config = InklingModelConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        num_nextn_predict_layers=2,
        local_layer_ids=[1],
    )
    vllm_config = default_vllm_config
    vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16, max_model_len=128, head_dtype=None
    )
    vllm_config.quant_config = quant_config
    vllm_config.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=config), num_speculative_tokens=2
    )
    model = InklingMTP(vllm_config=vllm_config)
    linears = [module for module in model.modules() if isinstance(module, LinearBase)]
    assert len(linears) == 10
    expected = (
        CompressedTensorsLinearMethod
        if mode in ("class", "names")
        else UnquantizedLinearMethod
    )
    assert all(isinstance(layer.quant_method, expected) for layer in linears)


@pytest.mark.parametrize("exclude_all", [False, True])
def test_mtp_honors_modelopt_exclusions(exclude_all) -> None:
    """Config mapping is shared across quantization backends."""
    quant_config = ModelOptNvFp4Config.from_config(
        {
            "quantization": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "kv_cache_quant_algo": None,
                "exclude_modules": (
                    ["model.mtp.*"] if exclude_all else _checkpoint_projections(0)
                ),
            }
        }
    )
    configure_quant_config(quant_config, InklingMTP)
    for name in (
        "input_proj",
        "transformer_block.attn.qkvr",
        "transformer_block.attn.wo_ud",
        "transformer_block.mlp.gate_up_proj",
        "transformer_block.mlp.down_proj",
    ):
        assert quant_config.is_layer_excluded(f"model.layers.0.{name}")
        assert quant_config.is_layer_excluded(f"model.layers.1.{name}") == exclude_all


@pytest.mark.parametrize(
    ("source", "target"), [("w13_dn", "gate_up_proj"), ("w2_md", "down_proj")]
)
def test_mtp_loads_mlp_activation_scales(source, target) -> None:
    """Projection renames apply to scale parameters as well as weights."""
    model = InklingMTP.__new__(InklingMTP)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.num_mtp_layers = 1
    model.model.chain_norm = None
    model.model.layers = nn.ModuleDict({"0": nn.Module()})
    block = model.model.layers["0"].transformer_block = nn.Module()
    block.mlp = nn.Module()
    projection = nn.Module()
    projection.input_scale = nn.Parameter(torch.empty(1))
    block.mlp.add_module(target, projection)
    with pytest.raises(ValueError, match="missing required parameters"):
        model.load_weights([])
    loaded = model.load_weights(
        [
            (
                f"model.mtp.layers.0.transformer_block.mlp.{source}.input_scale",
                torch.tensor([0.5]),
            )
        ]
    )
    assert loaded == {f"model.layers.0.transformer_block.mlp.{target}.input_scale"}
    torch.testing.assert_close(projection.input_scale, torch.tensor([0.5]))
