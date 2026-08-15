# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    _compute_g1_scale_c,
)


def test_situ_g1_scale_does_not_fold_gate_scale():
    moe_config = SimpleNamespace(
        is_act_and_mul=True,
        activation=MoEActivation.SITU,
    )
    quant_config = SimpleNamespace(
        g1_alphas=torch.tensor([2.0, 3.0]),
        a2_gscale=torch.tensor([5.0, 7.0]),
    )

    torch.testing.assert_close(
        _compute_g1_scale_c(moe_config, quant_config),
        quant_config.a2_gscale,
    )


def test_swiglu_g1_scale_still_folds_gate_scale():
    moe_config = SimpleNamespace(
        is_act_and_mul=True,
        activation=MoEActivation.SILU,
    )
    quant_config = SimpleNamespace(
        g1_alphas=torch.tensor([2.0, 3.0]),
        a2_gscale=torch.tensor([5.0, 7.0]),
    )

    torch.testing.assert_close(
        _compute_g1_scale_c(moe_config, quant_config),
        quant_config.g1_alphas * quant_config.a2_gscale,
    )
