# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.transformers_utils.configs.eagle import EAGLEConfig


class RecordingDraftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.load_calls: list[list[tuple[str, torch.Tensor]]] = []
        self.received_iterables: list[Iterable[tuple[str, torch.Tensor]]] = []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        self.received_iterables.append(weights)
        loaded = list(weights)
        self.load_calls.append(loaded)
        for name, tensor in loaded:
            param = self.get_parameter(name)
            param.data.copy_(tensor)


class ProposerWithGetModel:
    def __init__(self, model: nn.Module):
        self._model = model

    def get_model(self) -> nn.Module:
        return self._model


class ProposerWithModelAttr:
    def __init__(self, model: nn.Module):
        self.model = model


def make_worker():
    from vllm.v1.worker.gpu_worker import Worker

    worker = object.__new__(Worker)
    worker._sleep_saved_buffers = {}
    worker._sleep_saved_draft_params = {}
    worker.model_runner = SimpleNamespace(
        model=nn.Linear(2, 2, bias=False),
        post_kv_cache_wake_up=MagicMock(),
    )
    return worker


@pytest.mark.parametrize(
    ("runner_attr", "proposer_cls"),
    [
        ("drafter", ProposerWithGetModel),
        ("drafter", ProposerWithModelAttr),
        ("speculator", ProposerWithGetModel),
        ("speculator", ProposerWithModelAttr),
    ],
)
def test_worker_get_draft_model_resolves_v1_and_v2_proposers(
    runner_attr: str,
    proposer_cls: type[ProposerWithGetModel] | type[ProposerWithModelAttr],
):
    worker = make_worker()
    draft_model = RecordingDraftModel()
    setattr(worker.model_runner, runner_attr, proposer_cls(draft_model))

    assert worker.get_draft_model() is draft_model


def test_worker_get_draft_model_returns_none_without_proposer():
    worker = make_worker()

    assert worker.get_draft_model() is None


def test_update_speculative_model_weights_uses_draft_load_weights():
    worker = make_worker()
    draft_model = RecordingDraftModel()
    worker.model_runner.drafter = ProposerWithGetModel(draft_model)

    new_weight = torch.full((4,), 7.0)
    worker.update_speculative_model_weights([("weight", new_weight)])

    assert torch.equal(draft_model.weight, new_weight)
    assert len(draft_model.load_calls) == 1
    assert draft_model.load_calls[0][0][0] == "weight"
    assert torch.equal(draft_model.load_calls[0][0][1], new_weight)
    assert not isinstance(draft_model.received_iterables[0], list)


def test_update_speculative_model_weights_noops_without_draft_model():
    worker = make_worker()

    worker.update_speculative_model_weights([("weight", torch.ones(4))])


@patch("vllm.v1.worker.gpu_worker.get_mem_allocator_instance")
@patch("torch.cuda.mem_get_info", return_value=(1_000_000, 2_000_000))
def test_level_2_sleep_saves_draft_params_on_cpu(_, get_allocator):
    get_allocator.return_value.sleep = MagicMock()
    worker = make_worker()
    draft_model = RecordingDraftModel()
    worker.model_runner.drafter = ProposerWithGetModel(draft_model)

    worker.sleep(level=2)

    saved_weight = worker._sleep_saved_draft_params["weight"]
    assert saved_weight.device.type == "cpu"
    assert torch.equal(saved_weight, torch.arange(4, dtype=torch.float32))


@patch("vllm.v1.worker.gpu_worker.get_mem_allocator_instance")
def test_wake_up_restores_saved_draft_params_through_load_weights(get_allocator):
    get_allocator.return_value.wake_up = MagicMock()
    worker = make_worker()
    draft_model = RecordingDraftModel()
    worker.model_runner.drafter = ProposerWithGetModel(draft_model)
    saved_weight = torch.full((4,), 3.0)
    worker._sleep_saved_draft_params = {"weight": saved_weight}

    worker.wake_up()

    assert torch.equal(draft_model.weight, saved_weight)
    assert len(draft_model.load_calls) == 1
    assert draft_model.load_calls[0][0][0] == "weight"
    assert torch.equal(draft_model.load_calls[0][0][1], saved_weight)
    assert worker._sleep_saved_draft_params == {}


def test_dflash_config_derives_aux_layer_ids_from_top_level_target_ids():
    config = EAGLEConfig(
        model=PretrainedConfig(
            architectures=["LlamaForCausalLM"],
            vocab_size=128,
        ),
        method="dflash",
        target_layer_ids=[2, 4],
    )

    assert config.eagle_aux_hidden_state_layer_ids == [3, 5]
    assert config.dflash_config["target_layer_ids"] == [2, 4]


def test_dflash_config_rejects_conflicting_layer_id_aliases():
    with pytest.raises(ValueError, match="target_layer_ids conflict"):
        EAGLEConfig(
            model=PretrainedConfig(
                architectures=["LlamaForCausalLM"],
                vocab_size=128,
            ),
            method="dflash",
            target_layer_ids=[2, 4],
            dflash_config={"target_layer_ids": [3, 5]},
        )
