# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v4.nvidia.model as deepseek_v4_model
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4Model


class _FakeDeepseekV4Model:
    config = SimpleNamespace(num_attention_heads=8)

    def named_parameters(self):
        return []

    def get_expert_mapping(self):
        return [("experts.routed_experts.w13_", "experts.0.w1.", 0, "w1")]


@pytest.fixture(autouse=True)
def _single_tp_no_pp_missing(monkeypatch):
    monkeypatch.setattr(
        deepseek_v4_model, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(deepseek_v4_model, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        deepseek_v4_model, "is_pp_missing_parameter", lambda name, model: False
    )


def test_deepseek_v4_load_weights_skips_missing_expert_scale():
    fake_model = _FakeDeepseekV4Model()

    loaded = DeepseekV4Model.load_weights(
        fake_model,
        [
            (
                "layers.0.ffn.experts.0.w1.weight_scale",
                torch.ones(1, dtype=torch.float32),
            )
        ],
    )

    assert loaded == set()


def test_deepseek_v4_load_weights_keeps_missing_expert_weight_strict():
    fake_model = _FakeDeepseekV4Model()

    with pytest.raises(
        KeyError,
        match="layers.0.ffn.experts.routed_experts.w13_weight",
    ):
        DeepseekV4Model.load_weights(
            fake_model,
            [("layers.0.ffn.experts.0.w1.weight", torch.ones(1))],
        )
