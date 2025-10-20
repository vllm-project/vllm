# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DecodeBenchConnector: A KV Connector for decode instance performance testing.

This connector emulates a prefill-decode disaggregated setting by filling
the KV cache with dummy values, allowing measurement of decoder performance
under larger input sequence lengths (ISL) in resource-limited environments.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class DecodeBenchConnectorMetadata(KVConnectorMetadata):
    """Metadata for DecodeBenchConnector.

    Contains information about which requests need their KV cache filled
    with dummy values for benchmarking purposes.
    """

    # request_id -> (block_ids, num_tokens_to_fill)
    reqs_to_fill: dict[str, tuple[list[int], int]]


class DecodeBenchConnector(KVConnectorBase_V1):
    """
    A KV Connector for decode instance performance testing.

    This connector fills the KV cache with dummy (non-zero) values to
    emulate a prefill-decode disaggregated setting, enabling performance
    testing of the decoder with larger input sequence lengths.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)

        self.connector_scheduler: DecodeBenchConnectorScheduler | None = None
        self.connector_worker: DecodeBenchConnectorWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = DecodeBenchConnectorScheduler(vllm_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = DecodeBenchConnectorWorker(vllm_config)

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, DecodeBenchConnectorMetadata)
        self.connector_worker.start_fill_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # All operations are synchronous, so nothing to wait for
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        # This connector doesn't save KV cache (benchmarking only)
        pass

    def wait_for_save(self):
        # This connector doesn't save KV cache (benchmarking only)
        pass

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None


class DecodeBenchConnectorScheduler:
    """Scheduler-side implementation for DecodeBenchConnector."""

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        # Track which requests have already been filled
        self._filled_requests: set[str] = set()

        # Track pending fills for the current scheduler step
        # request_id -> (block_ids, num_tokens_to_fill)
        self._pending_fills: dict[str, tuple[list[int], int]] = {}

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        For new requests, return the number of tokens that should be filled
        with dummy KV cache values.

        Returns:
            (num_tokens_to_fill, is_async)
            - num_tokens_to_fill: total tokens in the request minus 1
              (we fill everything except the last token for decode)
            - is_async: False (synchronous filling)
        """
        req_id = request.request_id

        # Only fill once per request on first scheduling
        if req_id in self._filled_requests or num_computed_tokens > 0:
            return 0, False

        # Fill all tokens except the last one (which will be decoded)
        # This simulates having processed a long prefill
        num_tokens_to_fill = max(0, request.num_tokens - 1)

        if num_tokens_to_fill == 0:
            return 0, False

        logger.debug(
            "DecodeBenchConnector: Request %s will fill %d tokens in KV cache",
            req_id,
            num_tokens_to_fill,
        )

        # Return False for synchronous operation - the fill is fast enough
        # that async overhead isn't worth it
        return num_tokens_to_fill, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Called after blocks are allocated. Store the block IDs so we can
        fill them with dummy values.
        """
        req_id = request.request_id

        if num_external_tokens == 0:
            return

        # Get the block IDs that were allocated
        block_groups = blocks.get_block_ids()
        block_ids = block_groups[0]  # Get first group (for single-tensor KV)

        # Calculate how many blocks we need to fill
        # num_external_tokens are the tokens we said we'd provide
        num_blocks_to_fill = (
            num_external_tokens + self.block_size - 1
        ) // self.block_size

        # Store the blocks to fill
        self._pending_fills[req_id] = (
            block_ids[:num_blocks_to_fill],
            num_external_tokens,
        )
        self._filled_requests.add(req_id)

        logger.debug(
            "DecodeBenchConnector: Allocated %d blocks for request %s",
            num_blocks_to_fill,
            req_id,
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        """
        Build metadata containing information about which blocks to fill
        with dummy KV values.
        """
        meta = DecodeBenchConnectorMetadata(reqs_to_fill=self._pending_fills.copy())

        # Clear pending fills after building metadata
        self._pending_fills.clear()

        return meta

    def request_finished(self, request: "Request"):
        """
        Called when a request has finished. Clean up any state.
        """
        self._filled_requests.discard(request.request_id)


class DecodeBenchConnectorWorker:
    """Worker-side implementation for DecodeBenchConnector."""

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        # Will be populated via register_kv_caches
        self.kv_caches: dict[str, torch.Tensor] | None = None

        # Cache for pre-filled dummy block to avoid repeated allocation
        self._dummy_block_cache: torch.Tensor | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Store references to the KV cache tensors."""
        self.kv_caches = kv_caches
        logger.debug(
            "DecodeBenchConnector: Registered %d KV cache layers", len(kv_caches)
        )

    def start_fill_kv(self, metadata: DecodeBenchConnectorMetadata):
        """
        Fill the allocated KV cache blocks with dummy (non-zero) values.

        This simulates having a populated KV cache from a prefill phase,
        allowing decode performance testing with larger context sizes.
        """
        if not metadata.reqs_to_fill:
            return

        assert self.kv_caches is not None, "KV caches must be registered before filling"

        for req_id, (block_ids, num_tokens) in metadata.reqs_to_fill.items():
            self._fill_blocks(block_ids, num_tokens)
            logger.debug(
                "DecodeBenchConnector: Filled %d blocks (%d tokens) for request %s",
                len(block_ids),
                num_tokens,
                req_id,
            )

    def _fill_blocks(self, block_ids: list[int], num_tokens: int):
        """
        Fill specified blocks with dummy non-zero values.

        Args:
            block_ids: List of block IDs to fill
            num_tokens: Total number of tokens to fill across these blocks
        """
        if not block_ids:
            return

        # Fill each layer's KV cache with constant value
        assert self.kv_caches is not None
        for layer_name, kv_cache in self.kv_caches.items():
            # Create dummy block cache once per device/dtype
            if self._dummy_block_cache is None:
                block_shape = kv_cache.shape[1:]
                self._dummy_block_cache = torch.full(
                    block_shape, 0.015, dtype=kv_cache.dtype, device=kv_cache.device
                )

            # Convert block_ids to tensor on device
            block_ids_tensor = torch.tensor(
                block_ids, dtype=torch.long, device=kv_cache.device
            )

            # Filter invalid block IDs
            valid_mask = block_ids_tensor < kv_cache.shape[0]
            valid_block_ids = block_ids_tensor[valid_mask]

            if len(valid_block_ids) > 0:
                # Batch fill operation
                kv_cache[valid_block_ids] = self._dummy_block_cache

        logger.debug(
            "DecodeBenchConnector: Filled %d blocks with dummy values", len(block_ids)
        )
