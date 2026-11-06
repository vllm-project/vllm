# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.v1.executor.ray_utils import _check_dp_backend_supports_cluster


def _pc(dp_size: int, dp_backend: str, world_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        data_parallel_size=dp_size,
        data_parallel_backend=dp_backend,
        world_size=world_size,
    )


def test_mp_dp_multinode_rejected():
    pc = _pc(dp_size=4, dp_backend="mp", world_size=2)
    with pytest.raises(ValueError, match="data-parallel-backend=ray"):
        _check_dp_backend_supports_cluster(
            pc, driver_node_device_count=2, device_str="GPU"
        )


def test_mp_dp_singlenode_allowed():
    pc = _pc(dp_size=4, dp_backend="mp", world_size=2)
    _check_dp_backend_supports_cluster(pc, driver_node_device_count=8, device_str="GPU")


def test_ray_dp_backend_allowed():
    pc = _pc(dp_size=4, dp_backend="ray", world_size=2)
    _check_dp_backend_supports_cluster(pc, driver_node_device_count=2, device_str="GPU")


def test_no_dp_allowed():
    pc = _pc(dp_size=1, dp_backend="mp", world_size=2)
    _check_dp_backend_supports_cluster(pc, driver_node_device_count=1, device_str="GPU")
