# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

from vllm.config.parallel import ParallelConfig


def test_reserve_snapshot_ports():
    parallel_config = SimpleNamespace(
        _snapshot_data_parallel_port_list=None,
    )

    with patch(
        "vllm.config.parallel.get_open_ports_list", return_value=[2000, 2001]
    ) as get_ports:
        ParallelConfig.reserve_snapshot_ports(parallel_config)

    get_ports.assert_called_once_with(2)
    assert parallel_config._snapshot_data_parallel_port_list == [2000, 2001]


def test_restore_uses_snapshot_port_queue():
    parallel_config = SimpleNamespace(
        _snapshot_data_parallel_port_list=[2000, 2001],
        _data_parallel_master_port_list=[1000],
        data_parallel_master_port=999,
    )

    with patch("vllm.snapshot.utils.is_restore", return_value=True):
        port = ParallelConfig.get_next_dp_init_port(parallel_config)

    assert port == 2001
    assert parallel_config._snapshot_data_parallel_port_list == [2000]
    assert parallel_config._data_parallel_master_port_list == [1000]


def test_restore_without_snapshot_ports_uses_normal_queue():
    parallel_config = SimpleNamespace(
        _snapshot_data_parallel_port_list=None,
        _data_parallel_master_port_list=[1000],
        data_parallel_master_port=999,
    )

    with patch("vllm.snapshot.utils.is_restore", return_value=True):
        port = ParallelConfig.get_next_dp_init_port(parallel_config)

    assert port == 1000
