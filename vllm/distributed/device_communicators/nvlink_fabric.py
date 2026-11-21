# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-node NVLink fabric detection via NVML."""

import ctypes
import functools
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import import_pynvml

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = init_logger(__name__)
pynvml = import_pynvml()

_NULL_CLUSTER_UUID = bytes(pynvml.NVML_GPU_FABRIC_UUID_LEN)
_NvlinkFabricKey = tuple[str, int]


class SymmetricMemoryTopology(Enum):
    INTRA_NODE = auto()
    CROSS_NODE_NVLINK = auto()
    UNSUPPORTED = auto()


def _local_physical_device_id() -> int:
    visible_device_id = torch.accelerator.current_device_index()
    return current_platform.visible_device_id_to_physical_device_id(visible_device_id)


def _fabric_uuid_bytes(fabric: ctypes.Structure) -> bytes:
    offset = type(fabric).clusterUuid.offset
    return ctypes.string_at(
        ctypes.addressof(fabric) + offset,
        pynvml.NVML_GPU_FABRIC_UUID_LEN,
    )


def _get_local_nvlink_fabric_key() -> _NvlinkFabricKey | None:
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(_local_physical_device_id())
            fabric = pynvml.c_nvmlGpuFabricInfoV_t()
            pynvml.nvmlDeviceGetGpuFabricInfoV(handle, ctypes.byref(fabric))
        finally:
            pynvml.nvmlShutdown()
    except Exception as error:
        logger.debug("NVML fabric detection failed: %s", error)
        return None

    if (
        fabric.state != pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
        or fabric.status != pynvml.NVML_SUCCESS
    ):
        return None

    cluster_uuid = _fabric_uuid_bytes(fabric)
    if cluster_uuid == _NULL_CLUSTER_UUID:
        return None
    return (cluster_uuid.hex(), int(fabric.cliqueId))


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
