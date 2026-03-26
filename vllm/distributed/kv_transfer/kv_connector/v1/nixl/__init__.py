# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL KV connector package.

This package implements the NIXL (Nvidia Inter-node eXchange Library) connector
for KV cache transfer between vLLM instances in disaggregated inference setups.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    RemoteMeta,
    ReqId,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
    NixlConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
    NixlPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    GET_META_MSG,
    NIXL_CONNECTOR_VERSION,
    NixlWrapper,
    nixl_agent_config,
    nixlXferTelemetry,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)

__all__ = [
    # Main connector class
    "NixlConnector",
    # Scheduler and Worker
    "NixlConnectorScheduler",
    "NixlConnectorWorker",
    # Metadata classes
    "NixlAgentMetadata",
    "NixlConnectorMetadata",
    "NixlHandshakePayload",
    "RemoteMeta",
    "ReqMeta",
    "ReqId",
    # Stats and metrics
    "NixlKVConnectorStats",
    "NixlPromMetrics",
    # Utils and constants
    "NIXL_CONNECTOR_VERSION",
    "GET_META_MSG",
    "NixlWrapper",
    "nixlXferTelemetry",
    "nixl_agent_config",
    "zmq_ctx",
]
