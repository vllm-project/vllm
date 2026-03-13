# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor


def test_load_fused_expert_weights_missing_param_does_not_raise():
    loaded_weight = torch.empty((1, 1))
    assert (
        Qwen3_5MultiTokenPredictor.load_fused_expert_weights(
            None, "layers.0.mlp.experts.w2_weight", {}, loaded_weight, "w2", 1
        )
        is False
    )
    assert (
        Qwen3_5Model.load_fused_expert_weights(
            None, "layers.0.mlp.experts.w2_weight", {}, loaded_weight, "w2", 1
        )
        is False
    )


def test_load_fused_expert_weights_returns_true_if_any_local_expert_loaded():
    class DummyParam:
        weight_loader: object

    param = DummyParam()

    def weight_loader(
        param,
        curr_expert_weight,
        name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        return expert_id == 0

    param.weight_loader = weight_loader

    params_dict = {"layers.0.mlp.experts.w2_weight": param}
    loaded_weight = torch.empty((2, 1))
    assert (
        Qwen3_5MultiTokenPredictor.load_fused_expert_weights(
            None,
            "layers.0.mlp.experts.w2_weight",
            params_dict,
            loaded_weight,
            "w2",
            2,
        )
        is True
    )
