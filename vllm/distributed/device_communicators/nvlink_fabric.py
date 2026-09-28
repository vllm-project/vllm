# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-node NVLink fabric detection via ``nvidia-smi``.

The Python NVML fabric API can return an all-zero structure on GB300 systems,
while ``nvidia-smi -q -x`` reports the fabric state and identifiers correctly.
"""

import functools
import subprocess
import xml.etree.ElementTree as ElementTree
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = init_logger(__name__)

_NULL_CLUSTER_UUID_HEX = "0" * 32
_HEX_CHARS = frozenset("0123456789abcdefABCDEF")
_NvlinkFabricKey = tuple[str, int]


class SymmetricMemoryTopology(Enum):
    INTRA_NODE = auto()
    CROSS_NODE_NVLINK = auto()
    UNSUPPORTED = auto()


def _local_gpu_id_for_nvidia_smi() -> str:
    visible_device_id = torch.accelerator.current_device_index()
    return str(
        current_platform.visible_device_id_to_physical_device_id(visible_device_id)
    )


def _parse_nvidia_smi_fabric_key(output: str) -> _NvlinkFabricKey | None:
    """Parse one healthy Fabric record from ``nvidia-smi -q -x``."""
    try:
        fabric = ElementTree.fromstring(output).find("./gpu/fabric")
    except ElementTree.ParseError:
        return None
    if fabric is None:
        return None

    state = fabric.findtext("state")
    status = fabric.findtext("status")
    cluster_uuid = fabric.findtext("clusterUuid")
    clique_id = fabric.findtext("cliqueId")

    if state is None or state.strip().lower() != "completed":
        return None
    if status is None or status.strip().lower() != "success":
        return None
    if cluster_uuid is None or clique_id is None:
        return None

    uuid_hex = cluster_uuid.strip().replace("-", "").lower()
    if len(uuid_hex) != 32 or any(char not in _HEX_CHARS for char in uuid_hex):
        return None
    if uuid_hex == _NULL_CLUSTER_UUID_HEX:
        return None

    try:
        clique = int(clique_id.strip(), 0)
    except ValueError:
        return None
    if clique < 0:
        return None

    return (uuid_hex, clique)


def _get_local_nvlink_fabric_key() -> _NvlinkFabricKey | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x", "-i", _local_gpu_id_for_nvidia_smi()],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as error:
        logger.debug("nvidia-smi fabric detection failed: %s", error)
        return None

    if result.returncode != 0:
        logger.debug("nvidia-smi failed: %s", result.stderr.strip())
        return None

    return _parse_nvidia_smi_fabric_key(result.stdout)


def has_cross_node_nvlink(group: "ProcessGroup") -> bool:
    """Return whether all ranks report the same valid NVLink fabric.

    Detection fails closed when fabric information is unavailable or incomplete.
    The caller is responsible for establishing that the group crosses nodes.
    """
    if (
        not current_platform.is_cuda()
        or not dist.is_available()
        or not dist.is_initialized()
    ):
        return False

    world_size = dist.get_world_size(group=group)
    if world_size <= 1:
        return False

    fabric_keys: list[_NvlinkFabricKey | None] = [None] * world_size
    dist.all_gather_object(
        fabric_keys,
        _get_local_nvlink_fabric_key(),
        group=group,
    )
    valid_keys = [key for key in fabric_keys if key is not None]
    if len(valid_keys) != world_size:
        return False

    has_shared_fabric = len(set(valid_keys)) == 1
    if has_shared_fabric:
        logger.info_once(
            "Detected multi-node NVLink fabric: %s",
            valid_keys[0],
            scope="local",
        )
    return has_shared_fabric


@functools.cache
def get_symmetric_memory_topology(
    group: "ProcessGroup",
) -> SymmetricMemoryTopology:
    """Classify whether a CPU process group can safely rendezvous.

    Same-node symmetric-memory rendezvous can reject unsupported GPU
    connectivity synchronously. Cross-node rendezvous is safe to attempt only
    when every rank reports the same healthy NVLink fabric.
    """
    if all(in_the_same_node_as(group, source_rank=0)):
        return SymmetricMemoryTopology.INTRA_NODE
    if has_cross_node_nvlink(group):
        return SymmetricMemoryTopology.CROSS_NODE_NVLINK
    return SymmetricMemoryTopology.UNSUPPORTED
