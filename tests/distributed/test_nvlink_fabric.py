# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

import vllm.v1.attention.ops.cp_common as cp_common
from vllm.distributed.device_communicators import nvlink_fabric


@pytest.fixture(autouse=True)
def clear_topology_cache():
    nvlink_fabric.get_symmetric_memory_topology.cache_clear()
    yield
    nvlink_fabric.get_symmetric_memory_topology.cache_clear()


def _nvidia_smi_xml(
    state: str = "Completed",
    status: str = "Success",
    cluster_uuid: str = "bd77041e-6cd1-4b2a-81ed-f13c81d31e11",
    clique_id: str = "32766",
) -> str:
    return f"""\
<nvidia_smi_log>
  <gpu>
    <performance_state>P0</performance_state>
    <fabric>
      <state>{state}</state>
      <status>{status}</status>
      <cliqueId>{clique_id}</cliqueId>
      <clusterUuid>{cluster_uuid}</clusterUuid>
    </fabric>
  </gpu>
</nvidia_smi_log>
"""


def test_parse_completed_fabric_key() -> None:
    assert nvlink_fabric._parse_nvidia_smi_fabric_key(_nvidia_smi_xml()) == (
        "bd77041e6cd14b2a81edf13c81d31e11",
        32766,
    )


def test_nvidia_smi_uses_current_visible_device(monkeypatch) -> None:
    monkeypatch.setattr(
        nvlink_fabric.torch.accelerator, "current_device_index", lambda: 2
    )
    physical_id = MagicMock(return_value=5)
    monkeypatch.setattr(
        nvlink_fabric.current_platform,
        "visible_device_id_to_physical_device_id",
        physical_id,
    )

    assert nvlink_fabric._local_gpu_id_for_nvidia_smi() == "5"
    physical_id.assert_called_once_with(2)


@pytest.mark.parametrize(
    "output",
    [
        _nvidia_smi_xml(state="Not Started"),
        _nvidia_smi_xml(status="Failure"),
        _nvidia_smi_xml(status="N/A"),
        _nvidia_smi_xml(cluster_uuid="00000000-0000-0000-0000-000000000000"),
        _nvidia_smi_xml(cluster_uuid="bd77041e6cd14b2a81edf13c81d31e11-extra"),
        _nvidia_smi_xml(clique_id="-1"),
        "GPU 00000000:06:00.0",
    ],
)
def test_parse_rejects_unusable_fabric(output: str) -> None:
    assert nvlink_fabric._parse_nvidia_smi_fabric_key(output) is None


@pytest.mark.parametrize(
    ("gathered_keys", "expected"),
    [
        ([("fabric-a", 32766), ("fabric-a", 32766)], True),
        ([("fabric-a", 32766), ("fabric-b", 32766)], False),
        ([("fabric-a", 32766), None], False),
    ],
)
def test_cross_node_nvlink_requires_one_shared_fabric(
    monkeypatch,
    gathered_keys: list[tuple[str, int] | None],
    expected: bool,
) -> None:
    process_group = object()
    monkeypatch.setattr(nvlink_fabric.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(nvlink_fabric.dist, "is_available", lambda: True)
    monkeypatch.setattr(nvlink_fabric.dist, "is_initialized", lambda: True)

    def get_world_size(*, group):
        assert group is process_group
        return len(gathered_keys)

    monkeypatch.setattr(
        nvlink_fabric.dist,
        "get_world_size",
        get_world_size,
    )

    def all_gather_object(output, _local_key, *, group):
        assert group is process_group
        output[:] = gathered_keys

    monkeypatch.setattr(nvlink_fabric.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        nvlink_fabric,
        "_get_local_nvlink_fabric_key",
        lambda: gathered_keys[0],
    )

    assert nvlink_fabric.has_cross_node_nvlink(process_group) is expected


@pytest.mark.parametrize(
    ("same_node", "cross_node_nvlink", "expected"),
    [
        (True, False, nvlink_fabric.SymmetricMemoryTopology.INTRA_NODE),
        (False, True, nvlink_fabric.SymmetricMemoryTopology.CROSS_NODE_NVLINK),
        (False, False, nvlink_fabric.SymmetricMemoryTopology.UNSUPPORTED),
    ],
)
def test_symmetric_memory_topology(
    monkeypatch,
    same_node: bool,
    cross_node_nvlink: bool,
    expected: nvlink_fabric.SymmetricMemoryTopology,
) -> None:
    process_group = object()
    fabric_probe = MagicMock(return_value=cross_node_nvlink)
    monkeypatch.setattr(
        nvlink_fabric,
        "in_the_same_node_as",
        lambda _group, source_rank: [same_node],
    )
    monkeypatch.setattr(nvlink_fabric, "has_cross_node_nvlink", fabric_probe)

    assert nvlink_fabric.get_symmetric_memory_topology(process_group) is expected
    if same_node:
        fabric_probe.assert_not_called()
    else:
        fabric_probe.assert_called_once_with(process_group)


@pytest.mark.parametrize("use_direct", [None, True])
def test_multinode_without_nvlink_disables_direct_cp(
    monkeypatch, use_direct: bool | None
) -> None:
    group = MagicMock(cpu_group=object())
    symmetric_memory = MagicMock()

    cp_common._symm_mem_spans_group.cache_clear()
    monkeypatch.setattr(cp_common, "symm_mem_available", True)
    monkeypatch.setattr(cp_common.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        cp_common,
        "get_symmetric_memory_topology",
        lambda _group: nvlink_fabric.SymmetricMemoryTopology.UNSUPPORTED,
    )
    monkeypatch.setattr(cp_common, "symm_mem", symmetric_memory)

    assert not cp_common.direct_cp_enabled(group, torch.bfloat16, use_direct)
    symmetric_memory.empty.assert_not_called()
