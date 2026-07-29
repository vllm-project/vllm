# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.iquest_moe_v13 import (
    get_layer_sliding_window_size,
)
from vllm.model_executor.models.iquest_moe_v13_mtp import IquestMoeV13MTP


class _RecordingExpertParameter:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, str, str, int]] = []

    def weight_loader(
        self,
        param: "_RecordingExpertParameter",
        loaded_weight: torch.Tensor,
        weight_name: str,
        *,
        shard_id: str,
        expert_id: int,
    ) -> None:
        assert param is self
        self.calls.append((loaded_weight, weight_name, shard_id, expert_id))


def test_mtp_layers_skip_backbone_hybrid_attention_schedule() -> None:
    hybrid_config = {
        "first_layers_types": ["full_attention"],
        "last_layers_types": ["full_attention"] * 3,
        "hybrid_layers_types_block": [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
        ],
        "num_hybrid_layers_types_block": 21,
        "sliding_window_size": 4096,
    }

    assert get_layer_sliding_window_size(**hybrid_config, layer_idx=0) is None
    assert get_layer_sliding_window_size(**hybrid_config, layer_idx=2) == 4096
    assert get_layer_sliding_window_size(**hybrid_config, layer_idx=87) is None
    assert (
        get_layer_sliding_window_size(**hybrid_config, layer_idx=88, is_mtp_layer=True)
        is None
    )
    assert (
        get_layer_sliding_window_size(**hybrid_config, layer_idx=89, is_mtp_layer=True)
        is None
    )


def test_load_weights_accepts_streamed_per_expert_mtp_weights() -> None:
    layer_prefix = "model.layers.24.mtp_model_layer.mlp.experts"
    w13_param = _RecordingExpertParameter()
    w2_param = _RecordingExpertParameter()
    params = {
        f"{layer_prefix}.w13_weight": w13_param,
        f"{layer_prefix}.w2_weight": w2_param,
    }
    model = SimpleNamespace(
        config=SimpleNamespace(num_experts=8),
        mtp_start_layer_idx=24,
        named_parameters=lambda: params.items(),
        named_modules=lambda: (),
    )

    gate_weight = torch.tensor([1.0])
    up_weight = torch.tensor([2.0])
    down_weight = torch.tensor([3.0])
    loaded = IquestMoeV13MTP.load_weights(
        model,
        [
            (
                "mtp_layers.0.mtp_model_layer.mlp.experts.3.gate_proj.weight",
                gate_weight,
            ),
            (
                "mtp_layers.0.mtp_model_layer.mlp.experts.3.up_proj.weight",
                up_weight,
            ),
            (
                "mtp_layers.0.mtp_model_layer.mlp.experts.3.down_proj.weight",
                down_weight,
            ),
        ],
    )

    assert loaded == {
        f"{layer_prefix}.w13_weight",
        f"{layer_prefix}.w2_weight",
    }
    assert w13_param.calls == [
        (gate_weight, f"{layer_prefix}.w13_weight", "w1", 3),
        (up_weight, f"{layer_prefix}.w13_weight", "w3", 3),
    ]
    assert w2_param.calls == [
        (down_weight, f"{layer_prefix}.w2_weight", "w2", 3),
    ]
