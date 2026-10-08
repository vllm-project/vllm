# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed production settings shared by both sharded-RDT endpoints."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import ray
import torch

import vllm.distributed.weight_transfer.sharded_rdt_engine as engine_mod
import vllm.distributed.weight_transfer.sharded_rdt_trainer as trainer_mod
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    check_ray_rdt_version,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    ShardedRDTWeightTransferEngine,
    ShardedRDTWeightTransferInitInfo,
)
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
    ShardedRDTTrainerWeightTransferEngine,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("staging_device", "cpu"),
        ("staging_device", ""),
        ("stages_in_flight", True),
        ("stages_in_flight", 1.5),
        ("stages_in_flight", 0),
        ("num_rdt_buffers", False),
        ("num_rdt_buffers", "2"),
        ("num_rdt_buffers", -1),
        ("buffer_presize_gb", True),
        ("buffer_presize_gb", "1"),
        ("buffer_presize_gb", float("nan")),
        ("buffer_presize_gb", float("inf")),
        ("buffer_presize_gb", -0.1),
        ("num_consumers", True),
        ("num_consumers", 1.0),
        ("num_consumers", 0),
        ("num_consumers", -1),
    ],
)
def test_trainer_settings_reject_malformed_values_by_field(field, value):
    kwargs = {"num_consumers": 1, field: value}
    with pytest.raises(ValueError, match=field):
        ShardedRDTTrainerInitInfo(rank=0, **kwargs)


@pytest.mark.parametrize("staging_device", ["gpu", "host"])
def test_trainer_accepts_typed_staging_modes(staging_device):
    info = ShardedRDTTrainerInitInfo(
        rank=0,
        num_consumers=1,
        staging_device=staging_device,
        stages_in_flight=3,
        num_rdt_buffers=4,
        buffer_presize_gb=0.5,
    )
    assert info.staging_device == staging_device


def _worker_init_info(trainer_info):
    engine = ShardedRDTTrainerWeightTransferEngine.__new__(
        ShardedRDTTrainerWeightTransferEngine
    )
    engine._init_info = trainer_info
    engine._groups = [["w"]]
    engine._meta = [SimpleNamespace(name="w", dtype=torch.bfloat16, shape=(4,))]
    engine._owner_sets = [[0]]
    engine._name_owner_class = [0]
    worker = engine._build_worker_init_info(["server-0"])
    return ShardedRDTWeightTransferInitInfo(**asdict(worker))


def test_settings_reach_workers_unchanged():
    trainer = ShardedRDTTrainerInitInfo(
        rank=0,
        num_consumers=7,
        staging_device="host",
        stages_in_flight=3,
        num_rdt_buffers=4,
        buffer_presize_gb=0.5,
    )
    worker = _worker_init_info(trainer)
    assert worker.num_consumers == 7
    assert worker.staging_device == "host"
    assert worker.stages_in_flight == 3
    assert worker.num_rdt_buffers == 4
    assert worker.buffer_presize_gb == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("staging_device", "cpu"),
        ("stages_in_flight", True),
        ("stages_in_flight", 0),
        ("num_rdt_buffers", 1.5),
        ("num_rdt_buffers", -1),
        ("buffer_presize_gb", float("-inf")),
        ("buffer_presize_gb", -0.1),
        ("num_consumers", False),
        ("num_consumers", 1.0),
        ("num_consumers", -1),
    ],
)
def test_worker_rejects_malformed_values_before_pool_setup(field, value):
    engine = ShardedRDTWeightTransferEngine.__new__(ShardedRDTWeightTransferEngine)
    engine._configure_ring = Mock()
    info = ShardedRDTWeightTransferInitInfo(**{field: value})

    with pytest.raises(ValueError, match=field):
        engine.init_transfer_engine(info)

    engine._configure_ring.assert_not_called()


@pytest.mark.parametrize("num_consumers", [0, 1, 8])
def test_worker_accepts_compatible_consumer_counts(num_consumers):
    info = ShardedRDTWeightTransferInitInfo(num_consumers=num_consumers)
    engine_mod.validate_sharded_rdt_settings(
        staging_device=info.staging_device,
        stages_in_flight=info.stages_in_flight,
        num_rdt_buffers=info.num_rdt_buffers,
        buffer_presize_gb=info.buffer_presize_gb,
        num_consumers=info.num_consumers,
        allow_zero_consumers=True,
    )


@pytest.mark.parametrize(
    ("staging_device", "installed", "passes"),
    [
        ("gpu", "2.55.0", False),
        ("gpu", "2.56.0", True),
        ("host", "2.57.0", False),
        ("host", "2.58.0", True),
    ],
)
def test_ray_floor_depends_on_staging_mode(
    monkeypatch, staging_device, installed, passes
):
    monkeypatch.setattr("importlib.metadata.version", lambda _name: installed)
    if passes:
        check_ray_rdt_version(staging_device)
    else:
        with pytest.raises(ValueError, match=staging_device):
            check_ray_rdt_version(staging_device)


def test_worker_checks_host_ray_floor_before_pool_setup(monkeypatch):
    checked = []
    monkeypatch.setattr(engine_mod, "check_ray_rdt_version", checked.append)
    engine = ShardedRDTWeightTransferEngine.__new__(ShardedRDTWeightTransferEngine)
    engine.parallel_config = SimpleNamespace(enable_eplb=False)
    engine._configure_ring = Mock()

    with pytest.raises(NotImplementedError, match="staging_device"):
        engine.init_transfer_engine(
            ShardedRDTWeightTransferInitInfo(staging_device="host")
        )

    assert checked == ["host"]
    engine._configure_ring.assert_not_called()


def test_trainer_checks_selected_ray_floor_before_remote(monkeypatch):
    events = []
    monkeypatch.setattr(
        trainer_mod, "check_ray_rdt_version", lambda mode: events.append(mode)
    )
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "node"),
    )
    monkeypatch.setattr(ray, "get_gpu_ids", lambda: [])

    def remote(*_args, **_kwargs):
        events.append("remote")
        raise RuntimeError("stop at ray.remote")

    monkeypatch.setattr(ray, "remote", remote)
    engine = ShardedRDTTrainerWeightTransferEngine.__new__(
        ShardedRDTTrainerWeightTransferEngine
    )
    engine._init_info = ShardedRDTTrainerInitInfo(
        rank=0, num_consumers=1, staging_device="host"
    )

    with pytest.raises(RuntimeError, match="stop at ray.remote"):
        engine._spawn_server(["w"])

    assert events == ["host", "remote"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("staging_device", "host"), ("stages_in_flight", 2)],
)
def test_unbuilt_execution_paths_are_refused_by_field(field, value):
    info = ShardedRDTWeightTransferInitInfo(**{field: value})
    with pytest.raises(NotImplementedError, match=field):
        ShardedRDTWeightTransferEngine._refuse_unbuilt_settings(info)
