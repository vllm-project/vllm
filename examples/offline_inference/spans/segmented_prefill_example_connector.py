# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""
Example connector for segmented prefill with KV offloading.

Gap creation is now handled by SpanAwareGapPolicy at the scheduler level.
This connector only handles KV loading/storing operations.

To use this connector with gap policy:
    scheduler_config = SchedulerConfig(
        gap_policy_name="span_aware",
        gap_policy_config={
            "gap_length": 32,
            "span_marker_token_id": 10,
            "min_external_tokens": 32,
        }
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="SegmentedPrefillOffloadConnector",
    )
"""
import logging
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class SegmentedPrefillOffloadConnector(OffloadingConnector):
    """
    Example connector for segmented prefill with KV offloading.
    
    Gap creation is now handled by SpanAwareGapPolicy at the scheduler level.
    This connector only handles KV loading/storing operations.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config
        )
        logger.info(
            "SegmentedPrefillOffloadConnector initialized. "
            "Gap creation is handled by scheduler's GapPolicy."
        )

    def get_external_cache_hints(
        self,
        request: "Request",
    ) -> list[tuple[int, int]]:
        """
        Report any external cache issues.
        
        For this example connector, we don't have cache corruption issues,
        so we return an empty list. Real connectors might report:
        - Partially loaded blocks
        - Corrupted cache entries
        - Unavailable remote data
        """
        return []
