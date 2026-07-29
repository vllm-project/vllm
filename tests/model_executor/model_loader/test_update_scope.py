# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.lora.update import (
    TensorLoRAUpdateSession,
    config_digest,
    validate_complete_lora_weights,
)
from vllm.model_executor.model_loader.reload.baseline import (
    WeightUpdateBaselineEvent,
    WeightUpdateBaselineGroup,
    WeightUpdateBaselineReport,
    aggregate_weight_update_baselines,
    get_weight_update_baseline,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    finalize_load_recording,
    get_layer_completion_findings,
    get_layerwise_info,
    initialize_layerwise_reload,
    record_load_consumption,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.scope import (
    LoRAAdapterScope,
    PartialBaseWeightScope,
    normalize_update_scope,
)
from vllm.model_executor.model_loader.reload.source import observe_weight_sources
from vllm.model_executor.model_loader.reload.types import LoadManifestScope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

pytestmark = pytest.mark.skip_global_cleanup


class _Cfg:
    dtype = torch.float32


class _WeightLayer(nn.Module):
    def __init__(self, names: tuple[str, ...]) -> None:
        super().__init__()
        for name in names:
            parameter = nn.Parameter(torch.zeros(2), requires_grad=False)
            parameter.weight_loader = default_weight_loader
            self.register_parameter(name, parameter)


class _ScopedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = _WeightLayer(("weight",))
        self.second = _WeightLayer(("weight",))

    def load_weights(self, weights):
        loaded = set()
        for source_name, weight in weights:
            module_name, parameter_name = source_name.split(".")
            parameter = getattr(getattr(self, module_name), parameter_name)
            parameter.weight_loader(parameter, weight)
            loaded.add(source_name)
        return loaded


def _record_baseline(model: nn.Module, names: tuple[str, ...]) -> None:
    record_metadata_for_reloading(model)
    record_load_consumption(model)
    model.load_weights(
        observe_weight_sources(
            (name, torch.ones(2)) for name in names
        )
    )
    finalize_load_recording(model)


def test_partial_scope_initializes_only_selected_layers() -> None:
    model = _ScopedModel()
    _record_baseline(model, ("first.weight", "second.weight"))
    second_before = model.second.weight

    scope = PartialBaseWeightScope(("first.weight",))
    initialize_layerwise_reload(model, scope)

    assert get_layerwise_info(model.first).can_load()
    assert not get_layerwise_info(model.second).can_load()
    assert model.second.weight is second_before

    model.load_weights(
        observe_weight_sources(
            [("first.weight", torch.full((2,), 3.0))]
        )
    )
    finalize_layerwise_reload(model, _Cfg())

    assert torch.equal(model.first.weight, torch.full((2,), 3.0))
    assert model.second.weight is second_before
    assert get_layer_completion_findings() == []


def test_partial_scope_rejects_split_processing_unit_before_mutation() -> None:
    model = nn.Module()
    model.layer = _WeightLayer(("left", "right"))

    def load(weights):
        for source_name, weight in weights:
            parameter = getattr(model.layer, source_name.rsplit(".", 1)[-1])
            parameter.weight_loader(parameter, weight)

    record_metadata_for_reloading(model)
    record_load_consumption(model)
    load(
        observe_weight_sources(
            [
                ("layer.left", torch.ones(2)),
                ("layer.right", torch.ones(2)),
            ]
        )
    )
    finalize_load_recording(model)
    left_before = model.layer.left

    with pytest.raises(ValueError, match="splits a layerwise processing unit"):
        initialize_layerwise_reload(
            model,
            PartialBaseWeightScope(("layer.left",)),
        )

    assert model.layer.left is left_before
    assert not model.layer.left.is_meta


def test_weight_update_baseline_exposes_atomic_source_groups() -> None:
    model = nn.Module()
    model.layer = _WeightLayer(("left", "right"))

    def load(weights):
        for source_name, weight in weights:
            parameter = getattr(model.layer, source_name.rsplit(".", 1)[-1])
            parameter.weight_loader(parameter, weight)

    record_metadata_for_reloading(model)
    record_load_consumption(model)
    load(
        observe_weight_sources(
            [
                ("checkpoint.left", torch.ones(2)),
                ("checkpoint.right", torch.ones(2)),
            ]
        )
    )
    finalize_load_recording(model)

    report = get_weight_update_baseline(model)
    assert report.state == "exact"
    assert len(report.groups) == 1
    assert report.groups[0].module_name == "layer"
    assert report.groups[0].source_names == (
        "checkpoint.left",
        "checkpoint.right",
    )
    assert {event.target_name for event in report.groups[0].events} == {
        "layer.left",
        "layer.right",
    }


def test_weight_update_baseline_merges_cross_rank_closure_constraints() -> None:
    def report(*groups: tuple[str, ...]) -> WeightUpdateBaselineReport:
        return WeightUpdateBaselineReport(
            scope=LoadManifestScope(),
            state="exact",
            groups=tuple(
                WeightUpdateBaselineGroup(
                    module_name=f"layer.{index}",
                    events=tuple(
                        WeightUpdateBaselineEvent(name, name, ()) for name in names
                    ),
                )
                for index, names in enumerate(groups)
            ),
        )

    baseline = aggregate_weight_update_baselines(
        [
            report(("q", "k"), ("mlp",)),
            report(("k", "v"), ("expert.3",)),
        ]
    )

    assert baseline["ready"] is True
    assert baseline["atomic_source_groups"] == [
        ["expert.3"],
        ["k", "q", "v"],
        ["mlp"],
    ]
    assert baseline["scope_template"] == {
        "kind": "base_checkpoint",
        "mode": "partial",
        "source_names": [],
    }
    assert baseline["atomic_update_scopes"][1]["source_names"] == [
        "k",
        "q",
        "v",
    ]


def test_weight_update_baseline_reports_dummy_target_baseline() -> None:
    model = _ScopedModel()
    record_metadata_for_reloading(model)
    from vllm.model_executor.model_loader.reload.layerwise import (
        record_dummy_load_manifest,
    )

    record_dummy_load_manifest(model)
    report = get_weight_update_baseline(model)
    baseline = aggregate_weight_update_baselines([report])

    assert report.state == "provisional"
    assert baseline["ready"] is False
    assert "first.weight" in report.provisional_target_names
    assert "complete real base-weight update" in baseline["reason"]


def test_scope_normalization_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        normalize_update_scope(
            {
                "kind": "base_checkpoint",
                "mode": "partial",
                "source_names": ["a", "a"],
            }
        )


def test_tensor_lora_scope_requires_complete_manifest() -> None:
    peft_config = {
        "r": 2,
        "lora_alpha": 2,
        "target_modules": ["q_proj"],
    }
    names = (
        "base_model.model.q_proj.lora_A.weight",
        "base_model.model.q_proj.lora_B.weight",
    )
    scope = LoRAAdapterScope(
        adapter_id=7,
        adapter_name="policy",
        tensor_names=names,
        config_digest=config_digest(peft_config),
    )
    session = TensorLoRAUpdateSession(scope, peft_config)
    session.add_tensors({names[0]: torch.ones(2, 4)})

    with pytest.raises(ValueError, match="manifest mismatch"):
        session.finish()

    session.add_tensors({names[1]: torch.ones(4, 2)})
    request = session.finish()
    assert set(request.lora_tensors) == set(names)


def test_lora_replacement_rejects_incomplete_ab_pair() -> None:
    class Weights:
        lora_a = torch.ones(2, 4)
        lora_b = None

    with pytest.raises(ValueError, match="incomplete A/B"):
        validate_complete_lora_weights({"model.q_proj": Weights()})


def test_lora_remove_scope_rejects_replacement_manifest() -> None:
    with pytest.raises(ValueError, match="cannot declare replacement data"):
        LoRAAdapterScope(
            adapter_id=7,
            adapter_name="policy",
            operation="remove",
            tensor_names=("q_proj.lora_A.weight",),
        )
