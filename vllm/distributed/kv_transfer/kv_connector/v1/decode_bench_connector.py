# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DecodeBenchConnector: A KV Connector for decode instance performance testing.

This connector emulates a prefill-decode disaggregated setting by filling
the KV cache with dummy values, allowing measurement of decoder performance
under larger input sequence lengths (ISL) in resource-limited environments.

Usage:
    To use this connector for benchmarking, configure it in the kv_transfer_config:

    Example:
        vllm serve <model> --kv-transfer-config '{
            "kv_connector": "DecodeBenchConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "fill_mean": 0.015,
                "fill_std": 0.0
            }
        }'

    Then run your benchmark with desired input/output lengths:
        vllm bench serve --base-url http://127.0.0.1:8000 --model <model> \\
            --dataset-name random --random-input-len 40000 \\
            --random-output-len 100 --max-concurrency 10

    Configuration options (via kv_connector_extra_config):
        - fill_mean (float): Mean value for random normal fill (default: 0.015)
        - fill_std (float): Standard deviation for random fill (default: 0.0)
          Set to 0 for constant values, >0 for random sampling
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_utils import (
    dcp_world_size_for_kv_cache_spec,
    resolve_dcp_kv_block_size,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec, iter_layer_specs

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class DecodeBenchConnectorMetadata(KVConnectorMetadata):
    """Metadata for DecodeBenchConnector.

    Contains information about which requests need their KV cache filled
    with dummy values for benchmarking purposes.
    """

    # request_id -> (block_ids_per_group, num_tokens_to_fill)
    # block_ids_per_group is a tuple of lists, one per KV cache group
    # One group: ([1, 2, 3],)
    # Multiple groups: ([1, 2], [5, 6])
    reqs_to_fill: dict[str, tuple[tuple[list[int], ...], int]]


class DecodeBenchConnector(KVConnectorBase_V1, SupportsHMA):
    """
    A KV Connector for decode instance performance testing.

    This connector fills the KV cache with dummy values to emulate a
    prefill-decode disaggregated setting, enabling performance testing of the
    decoder with larger input sequence lengths.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        self.connector_scheduler: DecodeBenchConnectorScheduler | None = None
        self.connector_worker: DecodeBenchConnectorWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = DecodeBenchConnectorScheduler(
                vllm_config, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = DecodeBenchConnectorWorker(
                vllm_config, kv_cache_config
            )

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
        attn_metadata: AttentionMetadata,
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

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        # HMA-enabled path: same cleanup as the single-group variant since
        # this connector owns no external state per block.
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None


class DecodeBenchConnectorScheduler:
    """Scheduler-side implementation for DecodeBenchConnector."""

    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.kv_cache_groups = kv_cache_config.kv_cache_groups
        self.group_block_sizes = tuple(
            resolve_dcp_kv_block_size(
                group.kv_cache_spec,
                dcp_world_size_for_kv_cache_spec(
                    group.kv_cache_spec,
                    dcp_world_size,
                ),
            )
            for group in self.kv_cache_groups
        )

        # Track which requests have already been filled
        self._filled_requests: set[str] = set()

        # Track pending fills for the current scheduler step
        # request_id -> (block_ids_per_group, num_tokens_to_fill)
        # Note: _pending_fills doesn't need explicit cleanup - it's cleared
        # after build_connector_meta() is called in the same scheduler step
        self._pending_fills: dict[str, tuple[tuple[list[int], ...], int]] = {}

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
            - num_tokens_to_fill: number of uncomputed tokens minus 1
                (we fill everything except the last token for decode)
            - is_async: False (synchronous filling)
        """
        req_id = request.request_id

        # Only fill once per request on first scheduling
        if req_id in self._filled_requests:
            return 0, False

        # Calculate how many tokens we need to fill
        # Fill all uncomputed tokens except the last one (which will be decoded)
        # This simulates having processed a long prefill
        num_uncomputed_tokens = request.num_tokens - num_computed_tokens
        num_tokens_to_fill = max(0, num_uncomputed_tokens - 1)

        if num_tokens_to_fill == 0:
            return 0, False

        # Return False for synchronous operation - the fill is fast enough
        # that async overhead isn't worth it
        return num_tokens_to_fill, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Called after blocks are allocated. Store the block IDs so we can
        fill them with dummy values.

        Supports both single- and multi-group KV cache configurations.
        """
        req_id = request.request_id

        if num_external_tokens == 0:
            return

        total_computed_tokens = request.num_tokens - 1
        num_local_computed_tokens = total_computed_tokens - num_external_tokens
        block_ids_per_group_list: list[list[int]] = []
        for group_idx, (
            group,
            group_blocks,
            group_block_size,
        ) in enumerate(
            zip(
                self.kv_cache_groups,
                blocks.blocks,
                self.group_block_sizes,
                strict=True,
            )
        ):
            is_circular_buffer = all(
                isinstance(spec, CircularBufferSpec)
                for spec in iter_layer_specs(group.kv_cache_spec)
            )
            if not is_circular_buffer:
                num_computed_blocks = cdiv(total_computed_tokens, group_block_size)
                external_block_start = num_local_computed_tokens // group_block_size
                assert (
                    0
                    <= external_block_start
                    <= num_computed_blocks
                    <= len(group_blocks)
                ), (
                    "DecodeBenchConnector block range exceeds allocated blocks: "
                    f"request={req_id}, group={group_idx}, "
                    f"range=[{external_block_start}, {num_computed_blocks}), "
                    f"allocated={len(group_blocks)}"
                )
                selected_blocks = group_blocks[external_block_start:num_computed_blocks]
            else:
                selected_blocks = group_blocks

            block_ids = [
                block.block_id for block in selected_blocks if not block.is_null
            ]
            if not block_ids:
                logger.warning(
                    "DecodeBenchConnector: No blocks selected for KV cache group "
                    "%d with %d external tokens for request %s",
                    group_idx,
                    num_external_tokens,
                    req_id,
                )
            block_ids_per_group_list.append(block_ids)
        block_ids_per_group = tuple(block_ids_per_group_list)

        # Store the blocks to fill for all group. _pending_fills doesn't need cleanup
        # as it's cleared after build_connector_meta
        self._pending_fills[req_id] = (
            block_ids_per_group,
            num_external_tokens,
        )
        self._filled_requests.add(req_id)

        block_counts = tuple(len(group) for group in block_ids_per_group)
        logger.debug(
            "DecodeBenchConnector: Selected %d total blocks across %d KV cache "
            "groups for request %s (per-group counts: %s)",
            sum(block_counts),
            len(blocks.blocks),
            req_id,
            ", ".join(map(str, block_counts)),
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

    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        # Get fill parameters from extra config
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.fill_mean = kv_transfer_config.get_from_extra_config("fill_mean", 0.015)
        self.fill_std = kv_transfer_config.get_from_extra_config("fill_std", 0.0)

        # Will be populated via register_kv_caches
        self.kv_caches: dict[str, torch.Tensor] | None = None

        # Mapping from KV cache group index to list of layer names in that group
        self.group_to_layers = {
            group_idx: list(group.layer_names)
            for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
        }
        self._zero_fill_group_ids = {
            group_idx
            for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
            if any(
                isinstance(spec, CircularBufferSpec)
                for spec in iter_layer_specs(group.kv_cache_spec)
            )
        }

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Store references to the KV cache tensors."""
        self.kv_caches = kv_caches

        logger.debug(
            "DecodeBenchConnector: Registered %d KV cache layers",
            len(kv_caches),
        )

    def start_fill_kv(self, metadata: DecodeBenchConnectorMetadata):
        """
        Fill the allocated KV cache blocks with dummy values.

        This simulates having a populated KV cache from a prefill phase,
        allowing decode performance testing with larger context sizes.

        Supports both single- and multi-group KV cache configurations.
        """
        if not metadata.reqs_to_fill:
            return

        assert self.kv_caches is not None, "KV caches must be registered before filling"

        for req_id, (block_ids_per_group, num_tokens) in metadata.reqs_to_fill.items():
            # Fill blocks for each KV cache group
            for group_idx, block_ids in enumerate(block_ids_per_group):
                self._fill_blocks(group_idx, block_ids, num_tokens)

            block_counts = tuple(len(group) for group in block_ids_per_group)
            logger.debug(
                "DecodeBenchConnector: Filled %d total blocks (%d tokens) across "
                "%d groups for request %s (per-group counts: %s)",
                sum(block_counts),
                num_tokens,
                len(block_ids_per_group),
                req_id,
                ", ".join(map(str, block_counts)),
            )

    def _fill_blocks(self, group_idx: int, block_ids: list[int], num_tokens: int):
        """
        Fill specified blocks with dummy values for a specific KV cache group.

        Args:
            group_idx: The KV cache group index to fill
            block_ids: List of block IDs to fill in this group
            num_tokens: Total number of tokens to fill across these blocks
        """
        if not block_ids:
            return

        assert self.kv_caches is not None

        # Circular buffers may pack non-floating metadata alongside their
        # floating-point state, so arbitrary fill values are not representation-safe.
        fill_mean, fill_std = (
            (0.0, 0.0)
            if group_idx in self._zero_fill_group_ids
            else (self.fill_mean, self.fill_std)
        )

        # Get the layers that belong to this group
        layer_names = self.group_to_layers.get(group_idx, [])

        # Fill only the layers in this group
        for layer_name in layer_names:
            if layer_name not in self.kv_caches:
                logger.warning(
                    "DecodeBenchConnector: Layer %s not found in KV caches", layer_name
                )
                continue

            kv_cache = self.kv_caches[layer_name]

            # Attention layers store KV as a single block-indexed tensor whose
            # first dim is num_blocks; fill the requested block rows. Hybrid /
            # linear-attention layers (e.g. Mamba, Kimi Delta Attention) store
            # their state as a list/tuple of tensors that are NOT block-indexed
            # — each tensor is a single state buffer with no num_blocks
            # dimension — so fill each tensor in its entirety with the same
            # dummy values.
            if isinstance(kv_cache, torch.Tensor):
                self._fill_block_tensor(kv_cache, block_ids, fill_mean, fill_std)
            elif isinstance(kv_cache, (list, tuple)) and all(
                isinstance(t, torch.Tensor) for t in kv_cache
            ):
                for state_tensor in kv_cache:
                    self._fill_state_tensor(state_tensor, fill_mean, fill_std)
            else:
                logger.warning_once(
                    "DecodeBenchConnector: skipping fill for layer %s whose KV "
                    "cache is %s, not a tensor or a list/tuple of tensors.",
                    layer_name,
                    type(kv_cache).__name__,
                )
                continue

        logger.debug(
            "DecodeBenchConnector: Filled %d blocks in group %d with %s values "
            "(mean=%.3f, std=%.3f)",
            len(block_ids),
            group_idx,
            "random" if fill_std > 0 else "constant",
            fill_mean,
            fill_std,
        )

    def _fill_block_tensor(
        self,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        fill_mean: float,
        fill_std: float,
    ):
        """Fill the requested block rows of a block-indexed KV cache tensor.

        Args:
            kv_cache: A KV cache tensor whose first dim is num_blocks.
            block_ids: Block IDs to fill. IDs that are out of range for this
                tensor's first dim are ignored.
            fill_mean: Mean value for the fill.
            fill_std: Standard deviation for the fill.
        """
        # Convert block_ids to tensor on device
        block_ids_tensor = torch.tensor(
            block_ids, dtype=torch.long, device=kv_cache.device
        )

        # Filter invalid block IDs
        valid_mask = block_ids_tensor < kv_cache.shape[0]
        valid_block_ids = block_ids_tensor[valid_mask]

        if len(valid_block_ids) == 0:
            return

        # Create fill values - either constant or random
        block_shape = kv_cache.shape[1:]
        if fill_std > 0:
            # Random normal sampling
            fill_values = torch.normal(
                mean=fill_mean,
                std=fill_std,
                size=(len(valid_block_ids),) + block_shape,
                dtype=kv_cache.dtype,
                device=kv_cache.device,
            )
        else:
            # Constant fill value
            fill_values = torch.full(
                (len(valid_block_ids),) + block_shape,
                fill_mean,
                dtype=kv_cache.dtype,
                device=kv_cache.device,
            )

        # Batch fill operation
        kv_cache[valid_block_ids] = fill_values

    def _fill_state_tensor(
        self, kv_cache: torch.Tensor, fill_mean: float, fill_std: float
    ):
        """Fill an entire non-block-indexed state tensor with dummy values.

        Hybrid / linear-attention layers (e.g. Mamba, Kimi Delta Attention)
        store their per-layer state as tensors with no num_blocks dimension,
        so the whole tensor is filled with the same constant or random values
        used for block fills, rather than selected block rows.

        Args:
            kv_cache: A state tensor to fill in its entirety.
            fill_mean: Mean value for the fill.
            fill_std: Standard deviation for the fill.
        """
        if fill_std > 0:
            kv_cache.normal_(mean=fill_mean, std=fill_std)
        else:
            kv_cache.fill_(fill_mean)
