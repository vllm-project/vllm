# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngramConfig.verify_host_memory: reject pinned Engram tables that cannot fit.

The sizes mirror DeepSeek-V4.1-Flash, whose two Engram layers hold ~189 GiB of
fp8 rows plus scales. Split over TP8 each rank pins ~24 GiB, which looks fine
against a 55 GiB container limit until all eight ranks do it at once.
"""

from types import SimpleNamespace

import psutil
import pytest

import vllm.utils.cpu_resource_utils as cpu_resource_utils
from vllm.config.engram import EngramConfig

GIB = 1 << 30
DSV41_FLASH_TABLE_BYTES = 202_758_032_400


@pytest.fixture
def host(monkeypatch):
    """Fake the node's memory limits and record the soft-warning handoff."""
    state = SimpleNamespace(physical=1024 * GIB, cgroup=None, warned=[])
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(total=state.physical)
    )
    monkeypatch.setattr(
        cpu_resource_utils, "get_cgroup_memory_limit", lambda: state.cgroup
    )
    monkeypatch.setattr(
        cpu_resource_utils,
        "check_cgroup_memory_available",
        lambda required, name: state.warned.append(required),
    )
    return state


def parallel(dp_size: int = 1, dp_local: int = 1):
    return SimpleNamespace(
        data_parallel_size=dp_size, data_parallel_size_local=dp_local
    )


def offloaded(**kwargs) -> EngramConfig:
    return EngramConfig(cpu_offload=True, dp_shared_memory=False, **kwargs)


def test_rejects_whole_table_even_when_one_rank_shard_fits(host):
    host.cgroup = 55 * GIB
    assert host.cgroup > DSV41_FLASH_TABLE_BYTES // 8
    with pytest.raises(ValueError, match="container memory limit") as error:
        offloaded().verify_host_memory(DSV41_FLASH_TABLE_BYTES, parallel())
    assert '"cpu_offload": false' in str(error.value)


def test_rejects_against_physical_ram_without_cgroup(host):
    host.physical = 128 * GIB
    with pytest.raises(ValueError, match="physical RAM"):
        offloaded().verify_host_memory(DSV41_FLASH_TABLE_BYTES, parallel())


def test_tighter_of_cgroup_and_physical_wins(host):
    host.physical, host.cgroup = 128 * GIB, 512 * GIB
    with pytest.raises(ValueError, match="physical RAM"):
        offloaded().verify_host_memory(DSV41_FLASH_TABLE_BYTES, parallel())


def test_fitting_table_is_handed_to_the_soft_check(host):
    host.cgroup = 256 * GIB
    offloaded().verify_host_memory(DSV41_FLASH_TABLE_BYTES, parallel())
    assert host.warned == [DSV41_FLASH_TABLE_BYTES]


def test_each_local_dp_replica_pins_a_full_table(host):
    host.cgroup = 256 * GIB
    with pytest.raises(ValueError):
        offloaded().verify_host_memory(
            DSV41_FLASH_TABLE_BYTES, parallel(dp_size=2, dp_local=2)
        )


def test_embedding_across_dp_pins_only_the_local_share(host):
    host.cgroup = 128 * GIB
    offloaded(embedding_across_dp=True).verify_host_memory(
        DSV41_FLASH_TABLE_BYTES, parallel(dp_size=4, dp_local=2)
    )
    assert host.warned == [DSV41_FLASH_TABLE_BYTES // 2]


@pytest.mark.parametrize(
    "config",
    [
        EngramConfig(cpu_offload=False),
        EngramConfig(cpu_offload=True, dp_shared_memory=True),
    ],
    ids=["gpu_resident", "dev_shm_backed"],
)
def test_nothing_to_check_when_tables_are_not_pinned_per_rank(host, config):
    host.physical = 1 * GIB
    config.verify_host_memory(DSV41_FLASH_TABLE_BYTES, parallel())
    assert host.warned == []
