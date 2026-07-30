# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.parallel import ParallelConfig
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("CPU-only test", allow_module_level=True)

from vllm.v1.worker.cpu_worker import _get_cpushm_dist_ident  # noqa: E402


@pytest.mark.parametrize(
    ("dp_ports", "expected"),
    [
        ([26001, 26001], True),
        ([26001, 26002], False),
    ],
    ids=["same-dp-rendezvous", "different-dp-rendezvous"],
)
def test_get_cpushm_dist_ident_uses_dp_rendezvous(dp_ports, expected):
    idents = [
        _get_cpushm_dist_ident(
            ParallelConfig(
                tensor_parallel_size=1,
                data_parallel_size=2,
                data_parallel_master_ip="127.0.0.1",
                _data_parallel_master_port_list=[dp_port],
            ),
            f"tcp://127.0.0.1:{11001 + rank}",
        )
        for rank, dp_port in enumerate(dp_ports)
    ]

    assert (idents[0] == idents[1]) is expected


def test_get_cpushm_dist_ident_uses_file_store_basename():
    parallel_config = ParallelConfig()

    assert (
        _get_cpushm_dist_ident(parallel_config, "file:///tmp/vllm_dist_ident")
        == "vllm_dist_ident"
    )
