# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.amd import dspark as dspark_module
from vllm.models.deepseek_v4.amd.dspark import DSparkDeepseekV4ForCausalLM


def _make_uninitialized_model(confidence_head):
    model = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    object.__setattr__(
        model,
        "model",
        SimpleNamespace(confidence_head=confidence_head),
    )
    return model


def _prepare_loader_model(model, named_parameters):
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(
            n_routed_experts=1,
            expert_dtype="fp4",
            num_attention_heads=1,
        ),
    )
    object.__setattr__(model, "named_parameters", lambda: named_parameters)


def _disable_distributed_loader_paths(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        dspark_module,
        "fused_moe_make_expert_params_mapping",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        dspark_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(dspark_module, "get_tensor_model_parallel_rank", lambda: 0)


@pytest.mark.cpu_test
def test_deepseek_v4_rocm_dspark_maps_enabled_confidence_head():
    model = _make_uninitialized_model(object())

    assert (
        model._remap_dspark_name("mtp.2.confidence_head.proj.weight")
        == "model.confidence_head.proj.weight"
    )


@pytest.mark.cpu_test
def test_deepseek_v4_rocm_dspark_skips_disabled_confidence_head():
    model = _make_uninitialized_model(None)

    assert model._remap_dspark_name("mtp.2.confidence_head.proj.weight") is None


@pytest.mark.cpu_test
def test_deepseek_v4_rocm_dspark_disables_unloaded_confidence_head(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_uninitialized_model(object())
    _prepare_loader_model(model, [])
    _disable_distributed_loader_paths(monkeypatch)

    assert model.load_weights([]) == set()
    assert model.model.confidence_head is None


@pytest.mark.cpu_test
def test_deepseek_v4_rocm_dspark_loads_complete_confidence_head(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeParameter:
        loaded_weight = None

        def weight_loader(self, param, loaded_weight):
            assert param is self
            self.loaded_weight = loaded_weight

    confidence_head = object()
    parameter = FakeParameter()
    model = _make_uninitialized_model(confidence_head)
    _prepare_loader_model(
        model,
        [("model.confidence_head.proj.weight", parameter)],
    )
    _disable_distributed_loader_paths(monkeypatch)
    loaded_weight = torch.tensor([[1.0, 2.0]])

    loaded = model.load_weights([("mtp.2.confidence_head.proj.weight", loaded_weight)])

    assert loaded == {"model.confidence_head.proj.weight"}
    assert parameter.loaded_weight is loaded_weight
    assert model.model.confidence_head is confidence_head


@pytest.mark.cpu_test
def test_deepseek_v4_rocm_dspark_confidence_is_probability():
    class ConfidenceHead:
        def __call__(self, head_hidden, markov_embed):
            return (head_hidden[:, 0] + markov_embed[:, 0]).float()

    model = _make_uninitialized_model(ConfidenceHead())
    head_hidden = torch.tensor([[0.0], [1.0]], dtype=torch.bfloat16)
    markov_embed = torch.tensor([[0.0], [-2.0]], dtype=torch.bfloat16)

    confidence = model.compute_confidence(head_hidden, markov_embed)

    torch.testing.assert_close(
        confidence,
        torch.sigmoid(torch.tensor([0.0, -1.0])),
    )
    assert torch.all((confidence >= 0) & (confidence <= 1))
