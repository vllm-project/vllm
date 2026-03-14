# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Decentralized peer discovery via periodic bloom filter exchange.

Each vLLM instance maintains:
- A local bloom filter tracking block hashes in its own KV cache
- Per-peer bloom filters received from other instances
- A merged bloom filter (bitwise OR of all peer filters)

The sync protocol runs periodically in a background thread:
1. Rebuild local bloom filter from current cache keys
2. Send local bloom to all peers via ZMQ
3. Receive peer blooms, compute merged filter

This enables three-tier KV block lookup:
  Tier 1: Local BlockPool (0 RTT)
  Tier 2: Peer bloom filter check (~1 us same-rack)
  Tier 3: Recompute or remote storage (cross-rack latency)
"""

import threading
import time
from collections import OrderedDict

import msgspec
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.bloom_filter import (
    BloomFilter,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class BloomFilterMessage(
    msgspec.Struct,
    array_like=True,
    gc=False,
):
    """Wire format for bloom filter exchange between peers."""

    node_id: int
    bloom_data: bytes
    cache_entry_count: int
    timestamp: float


class PeerCacheEntry:
    """Metadata about a peer's KV cache contents."""

    __slots__ = ("node_id", "bloom", "last_updated", "cache_entry_count")

    def __init__(self, node_id: int, bloom: BloomFilter):
        self.node_id = node_id
        self.bloom = bloom
        self.last_updated = 0.0
        self.cache_entry_count = 0


class BloomFilterPeerDiscovery:
    """Decentralized peer discovery for KV cache blocks.

    Maintains local and peer bloom filters for efficient cross-node
    block discovery without centralized coordinators.
    """

    def __init__(
        self,
        node_id: int,
        num_nodes: int,
        sync_interval_ms: float = 100.0,
        bloom_fp_rate: float = 0.01,
        max_cache_entries: int = 10000,
        zmq_base_port: int = 15600,
    ):
        self._node_id = node_id
        self._num_nodes = num_nodes
        self._sync_interval_s = sync_interval_ms / 1000.0
        self._bloom_fp_rate = bloom_fp_rate
        self._max_cache_entries = max_cache_entries
        self._zmq_base_port = zmq_base_port

        # Local cache state (block_hash -> node_id for self)
        # Using OrderedDict as LRU cache for block hash metadata
        self._local_cache: OrderedDict[int, int] = OrderedDict()

        # Bloom filters
        self._local_bloom = BloomFilter(
            expected_items=max_cache_entries,
            fp_rate=bloom_fp_rate,
            auto_scale_clients=num_nodes,
        )
        self._peer_blooms: dict[int, PeerCacheEntry] = {}
        self._merged_bloom = self._local_bloom.copy()

        # Thread safety
        self._lock = threading.Lock()
        self._merged_bloom_lock = threading.Lock()

        # Statistics
        self._stats = {
            "local_hits": 0,
            "peer_hits": 0,
            "misses": 0,
            "false_positives": 0,
            "syncs_completed": 0,
        }

        # ZMQ setup (lazy init in start())
        self._zmq_ctx: zmq.Context | None = None
        self._pub_socket: zmq.Socket | None = None
        self._sub_socket: zmq.Socket | None = None
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(BloomFilterMessage)

        # Background sync thread
        self._running = False
        self._sync_thread: threading.Thread | None = None

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    def register_block(self, block_hash: int) -> None:
        """Called when a block is cached locally."""
        with self._lock:
            if block_hash in self._local_cache:
                self._local_cache.move_to_end(block_hash)
                return

            # Evict LRU if at capacity
            if len(self._local_cache) >= self._max_cache_entries:
                self._local_cache.popitem(last=False)

            self._local_cache[block_hash] = self._node_id
            self._local_bloom.add(block_hash)

    def unregister_block(self, block_hash: int) -> None:
        """Called when a block is evicted locally."""
        with self._lock:
            self._local_cache.pop(block_hash, None)
            # Note: can't remove from bloom filter; handled at next rebuild

    def register_blocks(self, block_hashes: list[int]) -> None:
        """Batch register multiple blocks."""
        with self._lock:
            for bh in block_hashes:
                if bh in self._local_cache:
                    self._local_cache.move_to_end(bh)
                    continue
                if len(self._local_cache) >= self._max_cache_entries:
                    self._local_cache.popitem(last=False)
                self._local_cache[bh] = self._node_id
                self._local_bloom.add(bh)

    def find_peer_with_block(self, block_hash: int) -> int | None:
        """Three-tier lookup for a KV cache block.

        Returns:
            node_id of peer that likely has the block, or None.
            Tier 1 (local) is not checked here — that's the caller's
            responsibility via BlockPool.
        """
        # Tier 2: Check merged peer bloom filter
        with self._merged_bloom_lock:
            if self._merged_bloom.contains(block_hash):
                self._stats["peer_hits"] += 1
                # Return a peer node_id hint (we don't know which
                # specific peer; the merged bloom is an OR of all)
                # The caller should try peers or use additional info
                return self._find_specific_peer(block_hash)

        self._stats["misses"] += 1
        return None

    def _find_specific_peer(self, block_hash: int) -> int | None:
        """Try to find which specific peer has the block."""
        for node_id, entry in self._peer_blooms.items():
            if entry.bloom.contains(block_hash):
                return node_id
        # Merged bloom said yes but individual blooms say no
        # This can happen due to race conditions; treat as miss
        self._stats["false_positives"] += 1
        return None

    def has_local_block(self, block_hash: int) -> bool:
        """Check if a block is in our local cache."""
        with self._lock:
            return block_hash in self._local_cache

    def get_local_cache_size(self) -> int:
        """Get number of blocks tracked locally."""
        with self._lock:
            return len(self._local_cache)

    def _sync_bloom_filters(self) -> None:
        """Periodic sync: rebuild local bloom, exchange with peers, merge."""
        # Step 1: Rebuild local bloom from current cache keys
        with self._lock:
            cache_keys = list(self._local_cache.keys())

        self._local_bloom.rebuild_from_keys(cache_keys)

        # Step 2: Publish local bloom to peers
        if self._pub_socket is not None:
            msg = BloomFilterMessage(
                node_id=self._node_id,
                bloom_data=self._local_bloom.to_bytes(),
                cache_entry_count=len(cache_keys),
                timestamp=time.time(),
            )
            try:
                self._pub_socket.send(self._encoder.encode(msg), zmq.NOBLOCK)
            except zmq.Again:
                logger.debug("Bloom filter publish dropped (HWM reached)")

        # Step 3: Receive peer blooms
        if self._sub_socket is not None:
            while True:
                try:
                    data = self._sub_socket.recv(zmq.NOBLOCK)
                    peer_msg = self._decoder.decode(data)
                    if peer_msg.node_id == self._node_id:
                        continue  # Skip our own messages
                    peer_bloom = BloomFilter.from_bytes(peer_msg.bloom_data)
                    if peer_msg.node_id not in self._peer_blooms:
                        self._peer_blooms[peer_msg.node_id] = PeerCacheEntry(
                            peer_msg.node_id, peer_bloom
                        )
                    else:
                        entry = self._peer_blooms[peer_msg.node_id]
                        entry.bloom = peer_bloom
                        entry.last_updated = peer_msg.timestamp
                        entry.cache_entry_count = peer_msg.cache_entry_count
                except zmq.Again:
                    break  # No more messages

        # Step 4: Merge all peer blooms
        all_blooms = [self._local_bloom]
        for entry in self._peer_blooms.values():
            all_blooms.append(entry.bloom)

        merged = BloomFilter.bitwise_or(all_blooms)
        with self._merged_bloom_lock:
            self._merged_bloom = merged

        self._stats["syncs_completed"] += 1

    def _sync_loop(self) -> None:
        """Background sync thread."""
        while self._running:
            try:
                self._sync_bloom_filters()
            except Exception:
                logger.exception("Error in bloom filter sync")
            time.sleep(self._sync_interval_s)

    def start(self) -> None:
        """Start the background sync thread and ZMQ sockets."""
        if self._running:
            return

        self._running = True

        # Initialize ZMQ
        if self._num_nodes > 1:
            self._zmq_ctx = zmq.Context.instance()

            # PUB socket: bind to our port
            self._pub_socket = self._zmq_ctx.socket(zmq.PUB)
            pub_port = self._zmq_base_port + self._node_id
            self._pub_socket.bind(f"tcp://*:{pub_port}")

            # SUB socket: connect to all peers
            self._sub_socket = self._zmq_ctx.socket(zmq.SUB)
            self._sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._sub_socket.setsockopt(zmq.RCVHWM, 100)
            for i in range(self._num_nodes):
                if i == self._node_id:
                    continue
                peer_port = self._zmq_base_port + i
                self._sub_socket.connect(f"tcp://localhost:{peer_port}")

            logger.info(
                "BloomFilterPeerDiscovery node %d: PUB on port %d, "
                "SUB to %d peers",
                self._node_id,
                pub_port,
                self._num_nodes - 1,
            )

        # Start sync thread
        self._sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name=f"bloom-sync-{self._node_id}",
        )
        self._sync_thread.start()

    def stop(self) -> None:
        """Stop the background sync thread and clean up."""
        self._running = False
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=2.0)
            self._sync_thread = None

        if self._pub_socket is not None:
            self._pub_socket.close(linger=0)
            self._pub_socket = None
        if self._sub_socket is not None:
            self._sub_socket.close(linger=0)
            self._sub_socket = None

    def force_sync(self) -> None:
        """Force an immediate bloom filter sync (useful for testing)."""
        self._sync_bloom_filters()
