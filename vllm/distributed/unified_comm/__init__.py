# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified Communication Abstraction Layer for vllm-hust.

Four-layer architecture:
  Layer 1: CommBackend     - device-level backend abstraction
                             (hides NCCL / HCCL / MetaX CCL differences)
  Layer 2: CollectiveOps   - device-agnostic collective primitives
                             (AllReduce / AllGather / Broadcast / ...)
  Layer 3: TransferPlane   - unified transfer abstraction for
                             KV-cache / EC / weight movement
  Layer 4: CommStrategy    - selects algorithm and link based on
                             topology, message size, and pattern

Integration:
  UnifiedCommAdapter       - bridges this layer into vLLM's
                             ``GroupCoordinator``. Activated only when
                             the environment variable
                             ``UNIFIED_COMM_ENABLED=1`` is set; the
                             default code path is unaffected otherwise.
"""

from vllm.distributed.unified_comm.adapter import (
    UnifiedCommAdapter,
    is_unified_comm_enabled,
)
from vllm.distributed.unified_comm.backend import (
    CommBackend,
    CommBackendRegistry,
    CommConfig,
    CommGroupInfo,
    ReduceOp,
    get_available_backend,
    get_backend,
    register_backend,
)
from vllm.distributed.unified_comm.collective import (
    CollectiveGroup,
    CollectiveOps,
)
from vllm.distributed.unified_comm.strategy import (
    CommStrategy,
    ConfigDrivenStrategy,
    DefaultStrategy,
    TopologyInfo,
)
from vllm.distributed.unified_comm.transfer_plane import (
    TransferPlane,
    TransferPlaneRegistry,
)
from vllm.distributed.unified_comm.types import (
    TransferProtocol,
    TransferType,
)

__all__ = [
    # Integration Adapter
    "UnifiedCommAdapter",
    "is_unified_comm_enabled",
    # Layer 1: Device Backend
    "CommBackend",
    "CommBackendRegistry",
    "CommConfig",
    "CommGroupInfo",
    "ReduceOp",
    "register_backend",
    "get_backend",
    "get_available_backend",
    # Layer 2: Collective Ops
    "CollectiveOps",
    "CollectiveGroup",
    # Layer 3: Transfer Plane
    "TransferPlane",
    "TransferPlaneRegistry",
    "TransferProtocol",
    "TransferType",
    # Layer 4: Strategy
    "CommStrategy",
    "DefaultStrategy",
    "ConfigDrivenStrategy",
    "TopologyInfo",
]
