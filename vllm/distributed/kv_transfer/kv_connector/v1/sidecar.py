# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public contract for KV-connector block sidecars."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass(frozen=True)
class KVConnectorSidecarConfig:
    """Connector storage layout exposed to block-aligned sidecars."""

    num_connector_blocks: int
    blocks_per_connector_block: int


@dataclass(frozen=True)
class KVConnectorSidecarBlockMap:
    """One normalized block mapping for connector sidecar data."""

    gpu_block_ids: np.ndarray
    connector_block_ids: np.ndarray
    connector_block_offsets: np.ndarray


@dataclass(frozen=True)
class KVConnectorSidecarTransferPlan:
    """Normalized load/store mappings for one connector step."""

    load: KVConnectorSidecarBlockMap | None = None
    store: KVConnectorSidecarBlockMap | None = None


class SupportsKVConnectorSidecar(ABC):
    """Capability implemented by connectors that expose block sidecars."""

    @abstractmethod
    def get_block_sidecar_config(self) -> KVConnectorSidecarConfig | None:
        """Return the connector layout, or ``None`` if unsupported."""
        raise NotImplementedError

    @abstractmethod
    def get_block_sidecar_transfers(
        self,
        connector_metadata: KVConnectorMetadata,
        kv_group_id: int,
    ) -> KVConnectorSidecarTransferPlan:
        """Normalize this step's internal KV jobs for sidecar consumers."""
        raise NotImplementedError
