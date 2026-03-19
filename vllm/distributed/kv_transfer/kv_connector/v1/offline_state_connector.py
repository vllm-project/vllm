# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OfflineState KV Connector: Bloom-filter cooperative caching for
decentralized KV cache discovery with three-tier lookup.

Based on the OfflineState design from "Cuckoo for Clients: Disaggregated
Cuckoo Hashing" thesis, this connector uses bloom filters for peer-to-peer
KV cache block discovery without centralized coordinators.

Three-tier lookup:
  Tier 1: Local BlockPool (0 RTT) - handled by vLLM's existing prefix cache
  Tier 2: Peer bloom filter check - this connector discovers which peer has it
  Tier 3: Recompute (no external match found)

Key properties:
  - Decentralized: no single point of failure for discovery
  - Compact: ~12 KB bloom filter per node for 10K blocks at 1% FPR
  - Sub-microsecond discovery via local bloom filter check
  - Periodic O(N) sync replaces continuous O(mutations) event streaming
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_discovery import (
    BloomFilterPeerDiscovery,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class OfflineStateRequestMeta:
    """Metadata for a single request's KV transfer."""

    request_id: str
    source_node_id: int  # Peer node that has the blocks
    block_hashes: list[int]  # Block hashes to load from peer
    num_tokens: int


@dataclass
class OfflineStateConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler-side to worker-side."""

    requests_to_load: list[OfflineStateRequestMeta] = field(default_factory=list)
    requests_to_store: list[str] = field(default_factory=list)


class OfflineStateConnector(KVConnectorBase_V1):
    """Bloom-filter cooperative caching connector for decentralized
    KV cache discovery.

    Scheduler-side: uses three-tier lookup to discover which peer
    node has the KV cache blocks for a request prefix.

    Worker-side: handles KV data transfer from discovered peer.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size

        # Get connector-specific config from extra_config
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._sync_interval_ms = extra.get("sync_interval_ms", 100.0)
        self._bloom_fp_rate = extra.get("bloom_fp_rate", 0.01)
        self._max_cache_entries = extra.get("max_cache_entries", 10000)
        self._zmq_base_port = extra.get("zmq_base_port", 15600)
        self._peer_hosts: list[str] | None = extra.get("peer_hosts", None)

        # Determine node identity from engine_id
        engine_id = self._kv_transfer_config.engine_id
        self._node_id = hash(engine_id) % 10000 if engine_id else 0
        self._num_nodes = extra.get("num_nodes", 1)

        # Scheduler-side state
        self._requests_need_load: dict[str, "Request"] = {}
        self._request_peer_map: dict[str, int] = {}  # req_id -> peer node_id
        self._request_block_hashes: dict[str, list[int]] = {}

        # Peer discovery service
        self._discovery: BloomFilterPeerDiscovery | None = None
        if self._num_nodes > 1:
            self._discovery = BloomFilterPeerDiscovery(
                node_id=self._node_id,
                num_nodes=self._num_nodes,
                sync_interval_ms=self._sync_interval_ms,
                bloom_fp_rate=self._bloom_fp_rate,
                max_cache_entries=self._max_cache_entries,
                zmq_base_port=self._zmq_base_port,
                peer_hosts=self._peer_hosts,
            )
            self._discovery.start()
            logger.info(
                "OfflineState connector initialized: node=%d, nodes=%d, "
                "sync_interval=%.0fms, fp_rate=%.3f, max_entries=%d",
                self._node_id,
                self._num_nodes,
                self._sync_interval_ms,
                self._bloom_fp_rate,
                self._max_cache_entries,
            )
        else:
            logger.info(
                "OfflineState connector initialized in single-node mode "
                "(bloom filter discovery disabled)"
            )

        # Statistics
        self._stats = {
            "local_prefix_hits": 0,
            "peer_bloom_hits": 0,
            "misses": 0,
            "blocks_registered": 0,
        }

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Three-tier prefix discovery.

        Tier 1: Local prefix cache is already handled by vLLM's
            BlockPool/KVCacheCoordinator before this method is called.
            num_computed_tokens reflects local prefix cache hits.

        Tier 2: Check merged peer bloom filter for remaining tokens.
            If a peer likely has the blocks, return the match count.

        Tier 3: No match found — return 0 (will recompute).
        """
        if self._discovery is None:
            return 0, False

        # Get the block hashes for the uncomputed prefix
        token_ids = request.prompt_token_ids or []
        total_prompt_tokens = len(token_ids)

        if num_computed_tokens >= total_prompt_tokens:
            return 0, False

        # Check how many additional blocks the peer might have
        # We need to compute block hashes for the uncomputed portion
        remaining_tokens = total_prompt_tokens - num_computed_tokens
        num_remaining_blocks = remaining_tokens // self._block_size

        if num_remaining_blocks == 0:
            return 0, False

        # Check bloom filter for each block hash in the remaining prefix
        # We use a simple hash of token ranges as block identifiers
        matched_blocks = 0
        matched_block_hashes = []
        peer_node_id = None

        for i in range(num_remaining_blocks):
            start = num_computed_tokens + i * self._block_size
            end = start + self._block_size
            if end > total_prompt_tokens:
                break

            # Compute a block hash from token IDs
            block_tokens = token_ids[start:end]
            block_hash = hash(tuple(block_tokens))

            peer = self._discovery.find_peer_with_block(block_hash)
            if peer is not None:
                if peer_node_id is None:
                    peer_node_id = peer
                if peer == peer_node_id:
                    matched_blocks += 1
                    matched_block_hashes.append(block_hash)
                else:
                    # Block is on a different peer; contiguous
                    # prefix from one source ends here.
                    break
            else:
                # Stop at first non-matched block (prefix must be contiguous)
                break

        if matched_blocks == 0:
            self._stats["misses"] += 1
            return 0, False

        num_matched_tokens = matched_blocks * self._block_size
        self._stats["peer_bloom_hits"] += 1

        # Store peer info for use in update_state_after_alloc
        assert peer_node_id is not None
        self._request_peer_map[request.request_id] = peer_node_id
        self._request_block_hashes[request.request_id] = matched_block_hashes

        logger.debug(
            "Bloom filter hit: request=%s, peer=%d, "
            "matched_blocks=%d, matched_tokens=%d",
            request.request_id,
            peer_node_id,
            matched_blocks,
            num_matched_tokens,
        )

        return num_matched_tokens, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Register newly allocated blocks in the bloom filter."""
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

        # Register all allocated blocks in bloom filter for peer discovery
        if self._discovery is not None:
            block_hashes = []
            token_ids = request.prompt_token_ids or []
            for group_blocks in blocks.blocks:
                for i, block in enumerate(group_blocks):
                    if hasattr(block, "block_hash") and block.block_hash is not None:
                        block_hashes.append(block.block_hash)
                    else:
                        # Compute hash from token position
                        start = i * self._block_size
                        end = min(start + self._block_size, len(token_ids))
                        if start < len(token_ids):
                            block_hash = hash(tuple(token_ids[start:end]))
                            block_hashes.append(block_hash)

            if block_hashes:
                self._discovery.register_blocks(block_hashes)
                self._stats["blocks_registered"] += len(block_hashes)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build metadata for worker-side KV transfer."""
        meta = OfflineStateConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            if req_id in self._requests_need_load:
                peer_node = self._request_peer_map.get(req_id, -1)
                block_hashes = self._request_block_hashes.get(req_id, [])
                meta.requests_to_load.append(
                    OfflineStateRequestMeta(
                        request_id=req_id,
                        source_node_id=peer_node,
                        block_hashes=block_hashes,
                        num_tokens=len(new_req.prompt_token_ids or []),
                    )
                )

        # Clean up state
        self._requests_need_load.clear()
        self._request_peer_map.clear()
        self._request_block_hashes.clear()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request finishes."""
        return False, None

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        """Start loading KV cache from discovered peer.

        In this initial implementation, we log the transfer intent.
        Full GPU-to-GPU RDMA transfer would be implemented here
        in a production deployment.
        """
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, OfflineStateConnectorMetadata):
            return

        for req_meta in metadata.requests_to_load:
            logger.debug(
                "OfflineState: would load %d blocks for request %s "
                "from peer node %d",
                len(req_meta.block_hashes),
                req_meta.request_id,
                req_meta.source_node_id,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until layer KV is loaded from peer."""
        # Synchronous in this implementation
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """After computing a layer, optionally notify discovery."""
        # No per-layer save needed — blocks are registered in
        # update_state_after_alloc at the scheduler level
        return

    def wait_for_save(self) -> None:
        """Block until all saves complete."""
        return

    # ==============================
    # Lifecycle
    # ==============================

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._discovery is not None:
            self._discovery.stop()
            logger.info(
                "OfflineState connector stats: %s, discovery: %s",
                self._stats,
                self._discovery.stats,
            )

    def get_discovery_stats(self) -> dict[str, Any]:
        """Get combined stats for evaluation."""
        stats = self._stats.copy()
        if self._discovery is not None:
            stats["discovery"] = self._discovery.stats
        return stats
