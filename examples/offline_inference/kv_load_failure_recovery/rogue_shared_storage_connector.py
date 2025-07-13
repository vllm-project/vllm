# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import logging
from typing import Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (
    SharedStorageConnector,
    SharedStorageConnectorMetadata,
)

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class RogueSharedStorageConnector(SharedStorageConnector):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self.invalid_block_ids: set = None

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(connector_metadata, SharedStorageConnectorMetadata)
        index, failed_request = next(
            (
                (i, x)
                for i, x in enumerate(connector_metadata.requests)
                if not x.is_store
            ),
            (None, None),
        )
        if index is not None:
            del connector_metadata.requests[index]
            self.invalid_block_ids = set(
                (
                    failed_request.slot_mapping[:: self._block_size] // self._block_size
                ).tolist()
            )
            logger.info(
                "Simulating failure to load all KV blocks for the "
                "first load request. Total blocks: %d",
                len(self.invalid_block_ids),
            )
        super().bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        self.invalid_block_ids: set = None
        super().clear_connector_metadata()

    def get_block_ids_with_load_errors(self) -> Optional[set[int]]:
        return self.invalid_block_ids
