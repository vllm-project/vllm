# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming per-unit expert reload: coverage, staging and commit timing."""

import weakref
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload.modelwise import (
    ModelwiseReloader,
    record_modelwise_reload_metadata,
)
from vllm.model_executor.model_loader.reload.sharding import (
    OBSERVED_SHARDS_ATTR,
    ReloadAwareWeightLoader,
    capture_rank_sharding,
    install_sharding_recorders,
    uninstall_sharding_recorders,
)
from vllm.model_executor.model_loader.reload.source import observe_weight_sources
from vllm.model_executor.model_loader.reload.units import (
    ReloadUnit,
    ShardCoverageTracker,
    StagingSpec,
    install_trackers,
    uninstall_trackers,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading

HIDDEN = 3
INTERMEDIATE = 2


class _FakeExperts(torch.nn.Module):
    """Minimal stand-in for RoutedExperts' expert weight loading contract."""

    def __init__(self, num_experts: int = 2, num_local_experts: int | None = None):
        super().__init__()
        self.local_num_experts = num_local_experts or num_experts
        self.intermediate_size_per_partition = INTERMEDIATE
        local = self.local_num_experts
        self.register_parameter(
            "w13_weight",
            torch.nn.Parameter(torch.zeros(local, 2 * INTERMEDIATE, HIDDEN)),
        )
        # Serving schema: one scale per expert, as post-load processing leaves
        # it. The checkpoint still carries one per half.
        self.register_parameter(
            "w13_weight_scale", torch.nn.Parameter(torch.ones(local))
        )
        self.register_parameter(
            "w2_weight", torch.nn.Parameter(torch.zeros(local, HIDDEN, INTERMEDIATE))
        )
        self.register_parameter(
            "w2_weight_scale", torch.nn.Parameter(torch.ones(local))
        )
        for param in self._parameters.values():
            param.weight_loader = self.weight_loader

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return expert_id if expert_id < self.local_num_experts else -1

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local = self._map_global_expert_id_to_local_expert_id(expert_id)
        if local < 0:
            return False if return_success else None

        if "scale" in weight_name:
            if shard_id in ("w1", "w3"):
                param.data[local][0 if shard_id == "w1" else 1] = loaded_weight.reshape(
                    ()
                )
            else:
                param.data[local] = loaded_weight.reshape(())
        elif shard_id == "w2":
            param.data[local].copy_(loaded_weight)
        else:
            start = 0 if shard_id == "w1" else INTERMEDIATE
            param.data[local].narrow(0, start, INTERMEDIATE).copy_(loaded_weight)
        return True if return_success else None


def _shard(value: float) -> torch.Tensor:
    return torch.full((INTERMEDIATE, HIDDEN), value)


def _w13_unit(layer: _FakeExperts, expert: int, commits: list) -> ReloadUnit:
    def commit(pieces):
        weight = pieces[("w13_weight", expert)]
        scale = pieces[("w13_weight_scale", expert)]
        # Stand-in for the FP8 requantization: publish the staged halves on
        # their shared max scale.
        layer.w13_weight.data[expert].copy_(weight)
        layer.w13_weight_scale.data[expert] = scale.max()
        commits.append(("w13", expert))

    return ReloadUnit(
        name=f"w13[{expert}]",
        keys=frozenset(
            {
                ("w13_weight", expert, "w1"),
                ("w13_weight", expert, "w3"),
                ("w13_weight_scale", expert, "w1"),
                ("w13_weight_scale", expert, "w3"),
            }
        ),
        commit=commit,
        staged={
            "w13_weight": StagingSpec((2 * INTERMEDIATE, HIDDEN), torch.float32),
            "w13_weight_scale": StagingSpec((2,), torch.float32),
        },
    )


def _load(layer, param_name, shard_id, expert_id, value):
    install_sharding_recorders(layer)
    param = getattr(layer, param_name)
    if "scale" in param_name:
        weight = torch.tensor(value)
    elif shard_id == "w2":
        weight = torch.full((HIDDEN, INTERMEDIATE), value)
    else:
        weight = _shard(value)
    return param.weight_loader(
        param=param,
        loaded_weight=weight,
        weight_name=param_name,
        shard_id=shard_id,
        expert_id=expert_id,
        return_success=True,
    )


def test_unit_commits_only_when_its_last_shard_arrives():
    layer = _FakeExperts()
    commits: list = []
    tracker = ShardCoverageTracker(layer, [_w13_unit(layer, 0, commits)])
    install_trackers([("layer", layer, tracker)])

    _load(layer, "w13_weight", "w1", 0, 5.0)
    _load(layer, "w13_weight", "w3", 0, 7.0)
    _load(layer, "w13_weight_scale", "w1", 0, 2.0)
    # Three of four shards: nothing committed, runtime untouched, staged.
    assert commits == []
    assert torch.equal(layer.w13_weight.data[0], torch.zeros(2 * INTERMEDIATE, HIDDEN))
    assert tracker.staged_bytes > 0

    _load(layer, "w13_weight_scale", "w3", 0, 4.0)

    assert commits == [("w13", 0)]
    assert torch.equal(layer.w13_weight.data[0][:INTERMEDIATE], _shard(5.0))
    assert torch.equal(layer.w13_weight.data[0][INTERMEDIATE:], _shard(7.0))
    assert layer.w13_weight_scale.data[0] == 4.0
    # The staging buffer is released as soon as the unit commits.
    assert tracker.staged_bytes == 0

    uninstall_trackers([("layer", layer, tracker)])


def test_unit_survives_shard_reordering_and_chunk_boundaries():
    layer = _FakeExperts()
    commits: list = []
    units = [_w13_unit(layer, e, commits) for e in range(2)]
    tracker = ShardCoverageTracker(layer, units)
    install_trackers([("layer", layer, tracker)])

    # Interleave two experts and put the scales before the weights.
    _load(layer, "w13_weight_scale", "w3", 1, 9.0)
    _load(layer, "w13_weight", "w1", 0, 1.0)
    _load(layer, "w13_weight_scale", "w1", 1, 3.0)
    _load(layer, "w13_weight", "w3", 1, 8.0)
    assert commits == []

    _load(layer, "w13_weight", "w1", 1, 6.0)
    assert commits == [("w13", 1)]

    _load(layer, "w13_weight", "w3", 0, 2.0)
    _load(layer, "w13_weight_scale", "w1", 0, 1.5)
    _load(layer, "w13_weight_scale", "w3", 0, 0.5)
    assert commits == [("w13", 1), ("w13", 0)]
    assert layer.w13_weight_scale.data[1] == 9.0
    assert layer.w13_weight_scale.data[0] == 1.5

    uninstall_trackers([("layer", layer, tracker)])


def test_repeated_send_rearms_the_unit():
    layer = _FakeExperts()
    commits: list = []
    tracker = ShardCoverageTracker(layer, [_w13_unit(layer, 0, commits)])
    install_trackers([("layer", layer, tracker)])

    for shard, value in (("w1", 1.0), ("w3", 2.0)):
        _load(layer, "w13_weight", shard, 0, value)
        _load(layer, "w13_weight_scale", shard, 0, value)
    assert commits == [("w13", 0)]

    # A second full round commits again rather than mixing rounds.
    _load(layer, "w13_weight", "w1", 0, 10.0)
    assert commits == [("w13", 0)]
    _load(layer, "w13_weight", "w3", 0, 11.0)
    _load(layer, "w13_weight_scale", "w1", 0, 12.0)
    _load(layer, "w13_weight_scale", "w3", 0, 13.0)
    assert commits == [("w13", 0), ("w13", 0)]
    assert torch.equal(layer.w13_weight.data[0][:INTERMEDIATE], _shard(10.0))
    assert layer.w13_weight_scale.data[0] == 13.0

    tracker.finish()


def test_partially_covered_unit_fails_closed_at_finish():
    layer = _FakeExperts()
    commits: list = []
    tracker = ShardCoverageTracker(layer, [_w13_unit(layer, 0, commits)])
    install_trackers([("layer", layer, tracker)])

    _load(layer, "w13_weight", "w1", 0, 1.0)
    _load(layer, "w13_weight_scale", "w1", 0, 1.0)

    with pytest.raises(ValueError, match="partial weight update"):
        tracker.finish()
    assert commits == []
    # Staging is dropped even when the transaction fails.
    assert tracker.staged_bytes == 0


def test_untouched_units_are_not_reported_as_incomplete():
    layer = _FakeExperts()
    commits: list = []
    units = [_w13_unit(layer, e, commits) for e in range(2)]
    tracker = ShardCoverageTracker(layer, units)
    install_trackers([("layer", layer, tracker)])

    for shard, value in (("w1", 1.0), ("w3", 2.0)):
        _load(layer, "w13_weight", shard, 0, value)
        _load(layer, "w13_weight_scale", shard, 0, value)

    tracker.finish()
    assert commits == [("w13", 0)]


def test_deferred_unit_commits_at_finish():
    layer = _FakeExperts()
    seen: list = []
    unit = ReloadUnit(
        name="input_scale",
        keys=frozenset({("w2_weight_scale", 0, "w2"), ("w2_weight_scale", 1, "w2")}),
        commit=lambda pieces: seen.append(sorted(pieces)),
        staged={"w2_weight_scale": StagingSpec((), torch.float32)},
        deferred=True,
    )
    tracker = ShardCoverageTracker(layer, [unit])
    install_trackers([("layer", layer, tracker)])

    _load(layer, "w2_weight_scale", "w2", 0, 1.0)
    _load(layer, "w2_weight_scale", "w2", 1, 2.0)
    assert seen == []

    tracker.finish()
    assert seen == [[("w2_weight_scale", 0), ("w2_weight_scale", 1)]]


def test_shards_of_remote_experts_are_ignored():
    layer = _FakeExperts(num_local_experts=1)
    commits: list = []
    tracker = ShardCoverageTracker(layer, [_w13_unit(layer, 0, commits)])
    install_trackers([("layer", layer, tracker)])

    assert _load(layer, "w13_weight", "w1", 5, 1.0) is False
    assert tracker.observed == set()

    tracker.finish()
    assert commits == []


def test_transport_tensor_is_not_retained_after_loading():
    layer = _FakeExperts()
    commits: list = []
    tracker = ShardCoverageTracker(layer, [_w13_unit(layer, 0, commits)])
    install_trackers([("layer", layer, tracker)])

    transport = _shard(3.0)
    ref = weakref.ref(transport)
    layer.w13_weight.weight_loader(
        param=layer.w13_weight,
        loaded_weight=transport,
        weight_name="w13_weight",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )
    del transport
    assert ref() is None

    tracker.release()


def test_loader_records_shard_keys_without_a_session():
    layer = _FakeExperts()

    _load(layer, "w13_weight", "w1", 0, 1.0)
    _load(layer, "w2_weight", "w2", 1, 1.0)
    _load(layer, "w13_weight", "w1", 9, 1.0)

    assert getattr(layer, OBSERVED_SHARDS_ATTR) == {
        ("w13_weight", 0, "w1"),
        ("w2_weight", 1, "w2"),
    }


def test_loader_preserves_moe_loading_marker_and_unwrapping():
    layer = _FakeExperts()
    install_sharding_recorders(layer)
    loader = layer.w13_weight.weight_loader
    assert isinstance(loader, ReloadAwareWeightLoader)
    assert loader.__wrapped__ == layer.weight_loader
    assert hasattr(loader, "__name__")


class _StreamingMethod(QuantizeMethodBase):
    """Quant method used to verify manifest-driven expert streaming."""

    def __init__(self, layer: _FakeExperts, commits: list):
        self.layer = layer
        self.commits = commits
        self.pwal_calls = 0

    def create_weights(self, layer, *args, **kwargs) -> None:
        raise NotImplementedError

    def apply(self, layer, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def process_weights_after_loading(self, layer) -> None:
        self.pwal_calls += 1


class _StreamingModel(torch.nn.Module):
    def __init__(self, commits: list):
        super().__init__()
        self.experts = _FakeExperts()
        self.experts.w13_weight_scale = torch.nn.Parameter(torch.ones(2, 2))
        self.experts.w13_weight_scale.weight_loader = self.experts.weight_loader
        self.experts.quant_method = _StreamingMethod(self.experts, commits)
        self.dense = torch.nn.Module()
        self.dense.weight = torch.nn.Parameter(torch.zeros(2))

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            if name.startswith("experts."):
                _, param_name, shard_id, expert_id = name.split(":")[0].split(".")
                param = getattr(self.experts, param_name)
                param.weight_loader(
                    param=param,
                    loaded_weight=value,
                    weight_name=param_name,
                    shard_id=shard_id,
                    expert_id=int(expert_id),
                    return_success=True,
                )
            else:
                self.get_parameter(name).data.copy_(value)
            loaded.add(name)
        return loaded


def _record_manifest(model, weights):
    """Capture the exact source-to-shard manifest for a test model."""
    install_sharding_recorders(model)
    try:
        with capture_rank_sharding(model, reset=True):
            model.load_weights(observe_weight_sources(weights))
    finally:
        uninstall_sharding_recorders(model)


def _expert_weights():
    for expert in range(2):
        for shard in ("w1", "w3"):
            yield f"experts.w13_weight.{shard}.{expert}", _shard(float(expert + 1))
            yield (
                f"experts.w13_weight_scale.{shard}.{expert}",
                torch.tensor(float(expert + 1)),
            )


def test_modelwise_reload_streams_experts_from_the_manifest():
    commits: list = []
    model = _StreamingModel(commits)
    record_modelwise_reload_metadata(model)
    initial = [*_expert_weights(), ("dense.weight", torch.tensor([0.0, 0.0]))]
    _record_manifest(model, initial)

    runtime_weight = model.experts.w13_weight
    runtime_ptr = runtime_weight.untyped_storage().data_ptr()
    method = model.experts.quant_method
    process_weights_after_loading(model, Mock(), torch.device("cpu"))
    assert method.pwal_calls == 1

    loaded = ModelwiseReloader(model, Mock(), torch.device("cpu")).reload(
        [*_expert_weights(), ("dense.weight", torch.tensor([1.0, 2.0]))]
    )

    assert commits == []
    assert method.pwal_calls == 2
    assert loaded is not None
    assert model.experts.w13_weight is runtime_weight
    assert model.experts.w13_weight.untyped_storage().data_ptr() == runtime_ptr
    assert torch.equal(model.experts.w13_weight.data[1][:INTERMEDIATE], _shard(2.0))
    assert torch.equal(model.dense.weight.data, torch.tensor([1.0, 2.0]))


def test_modelwise_reload_allows_a_partial_expert_update():
    commits: list = []
    model = _StreamingModel(commits)
    record_modelwise_reload_metadata(model)
    _record_manifest(model, list(_expert_weights()))
    process_weights_after_loading(model, Mock(), torch.device("cpu"))
    runtime = model.experts.w13_weight.detach().clone()

    partial = [
        ("experts.w13_weight.w1.0", _shard(1.0)),
        ("experts.w13_weight_scale.w1.0", torch.tensor(1.0)),
    ]
    loaded = ModelwiseReloader(model, Mock(), torch.device("cpu")).reload(partial)

    # The incomplete expert module is discarded and restored in runtime schema.
    assert loaded == {name for name, _ in partial}
    assert model.experts.w13_weight.shape == (2, 2 * INTERMEDIATE, HIDDEN)
    assert torch.equal(model.experts.w13_weight, runtime)
    assert getattr(model.experts, "_reload_shard_tracker", None) is None


def test_manifest_drives_expert_reload_without_a_second_tracker():
    """The rank manifest alone drives complete expert-module reload."""
    model = _StreamingModel([])
    initial = list(_expert_weights())
    record_modelwise_reload_metadata(model)
    _record_manifest(model, initial)
    runtime = model.experts.w13_weight.detach().clone()

    updated = list(_expert_weights())
    updated[0] = (updated[0][0], _shard(9.0))
    loaded = ModelwiseReloader(model, Mock(), torch.device("cpu")).reload(updated)

    assert loaded is not None
    assert torch.equal(model.experts.w13_weight[0][:INTERMEDIATE], _shard(9.0))
    assert not torch.equal(model.experts.w13_weight, runtime)
    assert getattr(model.experts, "_reload_shard_tracker", None) is None
