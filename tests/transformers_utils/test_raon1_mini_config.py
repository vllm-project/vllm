# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Release and legacy configs must select the same 1-based RDA/MLA layers."""

import pytest

from vllm.transformers_utils.configs.raon1_mini import Raon1MiniConfig


@pytest.mark.parametrize("release_names", [False, True])
def test_attention_mask_and_config_round_trip(release_names, tmp_path):
    attention = {
        "num_heads": 40,
        "head_dim": 128,
        "rda_layers": [1, 2, 3],
        "gate_lower_bound": -5.0,
    }
    if release_names:
        attention.update(mla_layers=[4], conv_kernel_size=4, full_rank_gate=True)
        config = Raon1MiniConfig(num_hidden_layers=4, rda_config=attention)
    else:
        attention.update(
            full_attn_layers=[4], short_conv_kernel_size=4, use_full_rank_gate=True
        )
        config = Raon1MiniConfig(num_hidden_layers=4, linear_attn_config=attention)
    config.save_pretrained(tmp_path)
    restored = Raon1MiniConfig.from_pretrained(tmp_path)
    assert [restored.is_rda_layer(i) for i in range(4)] == [True, True, True, False]
    assert restored.linear_attn_config["short_conv_kernel_size"] == 4
    assert restored.linear_attn_config["use_full_rank_gate"] is True


def test_reject_conflicting_mhc_alias():
    with pytest.raises(ValueError, match="Conflicting"):
        Raon1MiniConfig(raon_mhc_streams=4, mhc_streams=8)


@pytest.mark.parametrize("moe_name", ["moe", "block_sparse_moe"])
def test_mxfp8_keeps_router_and_shared_experts_unquantized(moe_name):
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config

    config = Raon1MiniConfig(
        quantization_config={
            "quant_method": "mxfp8",
            "ignored_layers": [
                f"model.layers.*.{moe_name}.gate*",
                f"model.layers.*.{moe_name}.shared_experts*",
                f"model.layers.*.{moe_name}.experts$",
            ],
        }
    )
    quant = ModelOptMxFp8Config.from_config(config.quantization_config)
    assert quant.is_layer_excluded("model.layers.1.moe.gate")
    assert quant.is_layer_excluded("model.layers.1.moe.shared_experts.gate_up_proj")
    assert not quant.is_layer_excluded("model.layers.1.moe.experts")
