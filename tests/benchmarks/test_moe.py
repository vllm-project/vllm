# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import Glm5NextConfig, Glm5NextTextConfig

from benchmarks.kernels.benchmark_moe import (
    get_model_params,
    get_weight_block_size_safety,
)


@pytest.mark.parametrize("multimodal", [True, False], ids=["nested", "text-only"])
def test_glm5next_moe_model_params(multimodal):
    """Tune routed experts using the MoE width, not the dense MLP width."""
    config = Glm5NextTextConfig(
        architectures=["Glm5NextForCausalLM"],
        n_routed_experts=288,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        intermediate_size=12288,
        hidden_size=4096,
    )
    if multimodal:
        config = Glm5NextConfig(
            architectures=["Glm5NextForConditionalGeneration"], text_config=config
        )

    assert get_model_params(config) == (288, 8, 2048, 4096)


def test_glm5next_outer_fp8_weight_block_size():
    """Reading text-model parameters must retain outer FP8 block metadata."""
    config = Glm5NextConfig(
        architectures=["Glm5NextForConditionalGeneration"],
        text_config={},
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    )

    get_model_params(config)

    assert get_weight_block_size_safety(config) == [128, 128]
