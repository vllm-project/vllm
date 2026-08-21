# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    _compute_g1_scale_c,
)


@pytest.mark.parametrize(
    ("is_act_and_mul", "activation", "fold_gate_scale"),
    [
        pytest.param(True, MoEActivation.SITU, False, id="situ"),
        pytest.param(True, MoEActivation.SILU, True, id="swiglu"),
        pytest.param(False, MoEActivation.RELU2_NO_MUL, False, id="non-gated"),
    ],
)
def test_compute_g1_scale_c(
    is_act_and_mul: bool,
    activation: MoEActivation,
    fold_gate_scale: bool,
):
    moe_config = SimpleNamespace(
        is_act_and_mul=is_act_and_mul,
        activation=activation,
    )
    quant_config = SimpleNamespace(
        g1_alphas=torch.tensor([2.0, 3.0]),
        a2_gscale=torch.tensor([5.0, 7.0]),
    )

    expected = quant_config.a2_gscale
    if fold_gate_scale:
        expected = quant_config.g1_alphas * expected

    torch.testing.assert_close(_compute_g1_scale_c(moe_config, quant_config), expected)
