# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration contracts shared by sharded-RDT trainers and workers."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.weight_transfer.sharded_rdt_common import (
    check_ray_rdt_version,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
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
        ("stages_in_flight", 0),
        ("stages_in_flight", True),
        ("num_rdt_buffers", -1),
        ("buffer_presize_gb", float("nan")),
        ("num_consumers", 0),
    ],
)
def test_trainer_rejects_invalid_settings(field, value):
    with pytest.raises(ValueError, match=field):
        ShardedRDTTrainerInitInfo(rank=0, **{"num_consumers": 1, field: value})


def test_settings_reach_workers_unchanged():
    trainer = ShardedRDTTrainerInitInfo(
        rank=0,
        num_consumers=7,
        staging_device="host",
        stages_in_flight=3,
        num_rdt_buffers=4,
        buffer_presize_gb=0.5,
    )
    engine = ShardedRDTTrainerWeightTransferEngine.__new__(
        ShardedRDTTrainerWeightTransferEngine
    )
    engine._init_info = trainer
    engine._groups = [["w"]]
    engine._meta = [SimpleNamespace(name="w", dtype=torch.bfloat16, shape=(4,))]
    engine._owner_sets = [[0]]
    engine._name_owner_class = [0]

    worker = ShardedRDTWeightTransferInitInfo(
        **asdict(engine._build_worker_init_info(["server-0"]))
    )

    assert (
        worker.num_consumers,
        worker.staging_device,
        worker.stages_in_flight,
        worker.num_rdt_buffers,
        worker.buffer_presize_gb,
    ) == (7, "host", 3, 4, 0.5)


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
