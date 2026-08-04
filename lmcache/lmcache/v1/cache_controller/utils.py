# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, NamedTuple, Optional, Set
import threading

# Third Party
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.locks import FastLockWithTimeout, RWLockWithTimeout
from lmcache.v1.cache_controller.message import BatchedKVOperationMsg, WorkerInfo
from lmcache.v1.utils.cache_utils import TTLListCache

logger = init_logger(__name__)


class KVChunkInfo(NamedTuple):
    """
    Represents the location information of a KV chunk in the cluster.
    This class is immutable and can be used as a dictionary key.
    """

    instance_id: str
    worker_id: int
    location: str


class FullSyncState(Enum):
    """State of full sync for a worker"""

    IDLE = "idle"  # Not in sync
    SYNCING = "syncing"  # Sync in progress
    COMPLETED = "completed"  # Sync completed
    FAILED = "failed"  # Sync failed/timeout


@dataclass
class WorkerSyncInfo:
    """Information about a worker's sync state"""

    sync_id: str
    state: FullSyncState
    start_time: float
    expected_total_keys: int
    expected_batch_count: int
    received_batches: Set[int] = field(default_factory=set)
    received_keys_count: int = 0
    last_activity_time: float = 0.0

    def __post_init__(self):
        if self.last_activity_time == 0.0:
            self.last_activity_time = self.start_time


@dataclass
class WorkerNode:
    """
    Represents a single worker with all its associated metadata.
    """

    worker_id: int
    ip: str
    port: int
    peer_init_url: Optional[str]
    socket: Optional[zmq.asyncio.Socket]
    registration_time: float
    last_heartbeat_time: float
    # Guarded by _lock
    seq_tracker: dict[str, int] = field(default_factory=dict)  # location -> seq_num
    kv_store: dict[str, set[int]] = field(
        default_factory=dict
    )  # location -> set[chunk_hash]
    sync_info: Optional[WorkerSyncInfo] = None  # Full sync state

    def __post_init__(self):
        # Fast lock with timeout for WorkerNode operations
        self._lock = FastLockWithTimeout()

    def handle_batched_kv_operations(
        self,
        msg: BatchedKVOperationMsg,
        on_seq_num_out_of_order: Optional[Callable[[], None]] = None,
        is_full_sync: bool = False,
    ) -> bool:
        """
        Handle batched KV operations with single lock acquisition.
        Logs warning and calls callback if sequence out of order is detected.

        Args:
            msg: The batched KV operation message.
            on_seq_num_out_of_order: Callback when sequence out of order detected.
            is_full_sync: If True, this is a full sync batch operation.
                         During full sync, sequence check is skipped.
                         When worker is syncing, non-full-sync operations are rejected.

        Returns:
            True if operations were processed, False if rejected
            (e.g., during syncing when is_full_sync=False).
        """
        seq_warning: Optional[tuple[int, int, int]] = None
        with self._lock:
            # During syncing, reject non-full-sync operations
            if (
                self.sync_info is not None
                and self.sync_info.state == FullSyncState.SYNCING
                and not is_full_sync
            ):
                return False

            location = msg.location

            if location not in self.kv_store:
                self.kv_store[location] = set()

            for op in msg.operations:
                # Apply operation first, then check sequence
                # (sequence check is skipped during full sync)
                if op.op_type.value == "admit":
                    self.kv_store[location].add(op.key)
                elif op.op_type.value == "evict":
                    self.kv_store[location].discard(op.key)
                else:
                    logger.error("Unknown op_type: %s", op.op_type)

                # Skip sequence check during full sync
                if is_full_sync:
                    continue

                # Sequence check
                last_seq_num = self.seq_tracker.get(location)
                if last_seq_num is not None:
                    expected_seq = last_seq_num + 1
                    if op.seq_num != expected_seq and seq_warning is None:
                        seq_warning = (
                            expected_seq,
                            op.seq_num,
                            op.seq_num - expected_seq,
                        )
                self.seq_tracker[location] = op.seq_num

            # Clean up empty set
            if not self.kv_store[location]:
                self.kv_store.pop(location, None)

        # Log warning and call callback outside lock
        if seq_warning is not None:
            if on_seq_num_out_of_order is not None:
                on_seq_num_out_of_order()
            logger.warning(
                "KV batch sequence out of order detected: "
                "key=%s, expected_seq=%s, actual_seq=%s, gap=%s",
                (msg.instance_id, msg.worker_id, msg.location),
                seq_warning[0],
                seq_warning[1],
                seq_warning[2],
            )
        return True

    def has_kv(self, location: str, key: int) -> bool:
        """Check if a KV chunk exists in this worker."""
        with self._lock:
            return location in self.kv_store and key in self.kv_store[location]

    def get_kv_keys(self, location: str) -> set[int]:
        """Get all keys for a location."""
        with self._lock:
            keys = self.kv_store.get(location)
            if keys is None:
                return set()
            # Return a shallow copy for thread safety
            return set(keys)

    def clear_kv_store(self) -> None:
        """Clear all KV data for this worker."""
        with self._lock:
            self.kv_store.clear()
            self.seq_tracker.clear()

    def get_kv_count(self) -> int:
        """Get total count of KV chunks."""
        with self._lock:
            return sum(len(keys) for keys in self.kv_store.values())

    def update_seq_num(self, location: str, seq_num: int) -> None:
        """Update sequence number for a location."""
        with self._lock:
            self.seq_tracker[location] = seq_num

    def get_seq_num(self, location: str) -> Optional[int]:
        """Get sequence number for a location."""
        with self._lock:
            return self.seq_tracker.get(location)

    def find_key(
        self, key: int
    ) -> Optional[tuple[KVChunkInfo, Optional[str], set[int]]]:
        """
        Find a key in this worker's kv_store.
        Returns: (KVChunkInfo, peer_init_url, keys) if found, None otherwise.
        KVChunkInfo will have instance_id="" since WorkerNode doesn't know
        its instance.
        """
        with self._lock:
            for location, keys in self.kv_store.items():
                if key in keys:
                    # WorkerNode doesn't know its instance_id, so we leave it empty
                    # The caller should fill in the instance_id
                    return (
                        KVChunkInfo("", self.worker_id, location),
                        self.peer_init_url,
                        keys,
                    )
        return None

    def find_key_simple(self, key: int) -> Optional[KVChunkInfo]:
        """
        Find a key in this worker's kv_store, returning only KVChunkInfo.
        KVChunkInfo will have instance_id="" since WorkerNode doesn't know
        its instance.
        """
        with self._lock:
            for location, keys in self.kv_store.items():
                if key in keys:
                    return KVChunkInfo("", self.worker_id, location)
        return None

    def to_worker_info(self, instance_id: str) -> WorkerInfo:
        """Convert to WorkerInfo for backward compatibility."""
        # No need to lock here
        return WorkerInfo(
            instance_id=instance_id,
            worker_id=self.worker_id,
            ip=self.ip,
            port=self.port,
            peer_init_url=self.peer_init_url,
            registration_time=self.registration_time,
            last_heartbeat_time=self.last_heartbeat_time,
        )


@dataclass
class InstanceNode:
    """
    Represents an instance with all its workers.
    Tree structure: InstanceNode -> WorkerNode
    Each InstanceNode has its own lock for thread-safe worker operations.
    """

    instance_id: str
    # Guarded by _rwlock
    workers: dict[int, WorkerNode] = field(
        default_factory=dict
    )  # worker_id -> WorkerNode

    def __post_init__(self):
        # RW lock for protecting workers dict access
        self._rwlock = RWLockWithTimeout()

    def add_worker(self, worker_node: WorkerNode) -> None:
        """Add a worker to this instance."""
        with self._rwlock.write_lock(timeout=10):
            self.workers[worker_node.worker_id] = worker_node

    def remove_worker(self, worker_id: int) -> Optional[WorkerNode]:
        """Remove and return a worker from this instance."""
        with self._rwlock.write_lock(timeout=10):
            return self.workers.pop(worker_id, None)

    def get_worker(self, worker_id: int) -> Optional[WorkerNode]:
        """Get a worker by worker_id. Uses optimistic locking (lock-free read)."""
        # Optimistic read: workers dict rarely changes
        return self.workers.get(worker_id)

    def get_worker_ids(self) -> list[int]:
        """Get sorted list of worker IDs."""
        # Snapshot keys to avoid RuntimeError if dict changes during iteration
        return sorted(list(self.workers.keys()))

    def has_workers(self) -> bool:
        """Check if instance has any workers."""
        # Optimistic read: workers dict rarely changes
        return len(self.workers) > 0

    def get_all_worker_infos(self) -> list[WorkerInfo]:
        """Get WorkerInfo for all workers in this instance."""
        # Snapshot values to avoid RuntimeError if dict changes during iteration
        workers_snapshot = list(self.workers.values())
        return [worker.to_worker_info(self.instance_id) for worker in workers_snapshot]

    def find_key(
        self, key: int
    ) -> Optional[tuple[KVChunkInfo, Optional[str], set[int]]]:
        """
        Find a key in any worker within this instance.
        Returns: (KVChunkInfo, peer_init_url, keys) if found, None otherwise.
        """
        # Snapshot workers to avoid RuntimeError if dict changes during iteration
        workers_snapshot = list(self.workers.items())
        for worker_id, worker_node in workers_snapshot:
            if result := worker_node.find_key(key):
                # Fill in the instance_id in KVChunkInfo
                kv_info, peer_init_url, keys = result
                return (
                    KVChunkInfo(self.instance_id, worker_id, kv_info.location),
                    peer_init_url,
                    keys,
                )
        return None

    def find_key_simple(self, key: int) -> Optional[KVChunkInfo]:
        """
        Find a key in any worker within this instance, returning only KVChunkInfo.
        """
        # Snapshot workers to avoid RuntimeError if dict changes during iteration
        workers_snapshot = list(self.workers.items())
        for worker_id, worker_node in workers_snapshot:
            if kv_info := worker_node.find_key_simple(key):
                return KVChunkInfo(self.instance_id, worker_id, kv_info.location)
        return None

    def has_worker_with_ip(self, ip: str) -> bool:
        """
        Check if any worker in this instance has the specified IP address.
        """
        # Snapshot workers to avoid RuntimeError if dict changes during iteration
        return any(worker.ip == ip for worker in list(self.workers.values()))

    def get_total_kv_count(self) -> int:
        """
        Get total count of KV chunks across all workers in this instance.
        """
        # Snapshot workers to avoid RuntimeError if dict changes during iteration
        return sum(worker.get_kv_count() for worker in list(self.workers.values()))


class RegistryTree:
    """
    Central registry managing the tree structure of instances and workers.
    Structure: instance_id -> InstanceNode -> WorkerNode

    Lock hierarchy (from coarse to fine):
    1. RegistryTree._rwlock: protects instances dict access
    2. InstanceNode._rwlock: protects workers dict access
    3. WorkerNode._lock: protects kv_store and seq_tracker access

    This fine-grained locking allows concurrent operations on different
    instances/workers, improving throughput significantly.
    """

    def __init__(self):
        # Guarded by _rwlock
        # instance_id -> InstanceNode
        self.instances: dict[str, InstanceNode] = {}
        # RW lock only for protecting instances dict access
        self._rwlock = RWLockWithTimeout()
        # Atomic counter for sequence discontinuity (protected by _counter_lock)
        self._seq_discontinuity_count = 0
        self._counter_lock = threading.Lock()
        # Cache for get_all_worker_infos (for Prometheus metrics)
        self._worker_info_cache: TTLListCache[WorkerInfo] = TTLListCache[WorkerInfo]()
        # Cache for get_all_worker_nodes (for FullSyncTracker and other use cases)
        self._worker_node_cache: TTLListCache[tuple[str, WorkerNode]] = TTLListCache[
            tuple[str, WorkerNode]
        ]()

    def get_seq_discontinuity_count(self) -> int:
        """Get the count of sequence discontinuities (thread-safe)."""
        # Lock-free read, no need for lock
        return self._seq_discontinuity_count

    def _incr_seq_discontinuity_count(self) -> None:
        """Increment the sequence discontinuity counter (thread-safe)."""
        with self._counter_lock:
            self._seq_discontinuity_count += 1

    def _get_or_create_instance(self, instance_id: str) -> InstanceNode:
        """Get or create an instance node. Internal use only."""
        # Optimistic read first: instances dict rarely changes
        instance_node = self.instances.get(instance_id)
        if instance_node is not None:
            return instance_node

        # Need to create, use write lock
        with self._rwlock.write_lock(timeout=10):
            # Double-check after acquiring write lock
            instance_node = self.instances.get(instance_id)
            if instance_node is None:
                instance_node = InstanceNode(instance_id=instance_id)
                self.instances[instance_id] = instance_node
            return instance_node

    def _get_instance(self, instance_id: str) -> Optional[InstanceNode]:
        """Get an instance node. Uses optimistic locking (lock-free read)."""
        # Optimistic read: instances dict rarely changes
        return self.instances.get(instance_id)

    def register_worker(
        self,
        instance_id: str,
        worker_id: int,
        ip: str,
        port: int,
        peer_init_url: Optional[str],
        socket: zmq.asyncio.Socket,
        registration_time: float,
    ) -> WorkerNode:
        """Register a new worker, creating instance if needed."""
        # Get or create instance (locks instances dict)
        instance_node = self._get_or_create_instance(instance_id)

        # Create worker node
        worker_node = WorkerNode(
            worker_id=worker_id,
            ip=ip,
            port=port,
            peer_init_url=peer_init_url,
            socket=socket,
            registration_time=registration_time,
            last_heartbeat_time=registration_time,
        )
        # Add worker (locks workers dict in instance_node)
        instance_node.add_worker(worker_node)
        return worker_node

    def deregister_worker(
        self, instance_id: str, worker_id: int
    ) -> Optional[WorkerNode]:
        """Deregister a worker and clean up empty instances."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return None

        # Remove worker (locks workers dict in instance_node)
        worker_node = instance_node.remove_worker(worker_id)

        # Clean up empty instance (need write lock on instances)
        if not instance_node.has_workers():
            # TODO(baoloongmao): Move timeout values to configuration
            with self._rwlock.write_lock(timeout=100):
                # Double-check after acquiring write lock
                # Use pop to avoid KeyError if another thread already removed it
                if not instance_node.has_workers() and instance_id in self.instances:
                    self.instances.pop(instance_id, None)

        return worker_node

    def get_worker(self, instance_id: str, worker_id: int) -> Optional[WorkerNode]:
        """Get a specific worker."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return None
        return instance_node.get_worker(worker_id)

    def get_instance(self, instance_id: str) -> Optional[InstanceNode]:
        """Get an instance by instance_id."""
        return self._get_instance(instance_id)

    def get_instance_by_ip(self, ip: str) -> Optional[InstanceNode]:
        """Get an instance by IP address. Returns first instance if multiple exist."""
        # Snapshot to avoid RuntimeError if dict changes during iteration
        for instance_node in list(self.instances.values()):
            if instance_node.has_worker_with_ip(ip):
                return instance_node
        return None

    def get_instances_by_ip(self, ip: str) -> list[InstanceNode]:
        """Get all instances by IP address."""
        # Snapshot to avoid RuntimeError if dict changes during iteration
        result = []
        for instance_node in list(self.instances.values()):
            if instance_node.has_worker_with_ip(ip):
                result.append(instance_node)
        return result

    def get_worker_ids(self, instance_id: str) -> list[int]:
        """Get sorted list of worker IDs for an instance."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return []
        return instance_node.get_worker_ids()

    def get_all_worker_infos_cached(
        self, timeout_seconds: Optional[float] = None
    ) -> list[WorkerInfo]:
        """Get WorkerInfo for all workers across all instances.

        Args:
            timeout_seconds: Cache timeout in seconds.
                - If None (default): use the cache's default timeout.
                - If 0: cache immediately expires, always get fresh data
                  and update cache.
                - If > 0: use cache with specified timeout.

        Returns:
            List of WorkerInfo, either from cache or fresh data
        """
        return self._worker_info_cache.get_cached(
            lambda: self._get_all_worker_infos_fresh(), timeout_override=timeout_seconds
        )

    def _get_all_worker_infos_fresh(self) -> list[WorkerInfo]:
        """Internal method to get fresh WorkerInfo without cache."""
        # Snapshot to avoid RuntimeError if dict changes during iteration
        result = []
        for instance_node in list(self.instances.values()):
            result.extend(instance_node.get_all_worker_infos())
        return result

    def get_all_worker_nodes_cached(
        self, timeout_seconds: Optional[float] = None
    ) -> list[tuple[str, WorkerNode]]:
        """Get all (instance_id, WorkerNode) tuples across all instances, with caching.

        Args:
            timeout_seconds: Cache timeout in seconds.
                - If None (default): use the cache's default timeout.
                - If 0: cache immediately expires, always get fresh data
                  and update cache.
                - If > 0: use cache with specified timeout.

        Returns:
            List of (instance_id, WorkerNode) tuples, either from cache or fresh data
        """
        return self._worker_node_cache.get_cached(
            lambda: self._get_all_worker_nodes_fresh(), timeout_override=timeout_seconds
        )

    def _get_all_worker_nodes_fresh(self) -> list[tuple[str, WorkerNode]]:
        """Internal method to get fresh (instance_id, WorkerNode) tuples
        without cache."""
        # Snapshot to avoid RuntimeError if dict changes during iteration
        result = []
        for instance_id, instance_node in self.instances.items():
            for worker_node in instance_node.workers.values():
                result.append((instance_id, worker_node))
        return result

    def update_heartbeat(
        self, instance_id: str, worker_id: int, timestamp: float
    ) -> bool:
        """Update worker heartbeat timestamp. Returns True if successful."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return False
        worker_node = instance_node.get_worker(worker_id)
        if worker_node is None:
            return False
        worker_node.last_heartbeat_time = timestamp
        return True

    def handle_batched_kv_operations(
        self, msg: BatchedKVOperationMsg, is_full_sync: bool = False
    ) -> bool:
        """
        Handle batched KV operations by forwarding to WorkerNode.

        Args:
            msg: The batched KV operation message.
            is_full_sync: If True, this is a full sync batch operation.

        Returns:
            True if operations were processed successfully.
            False if worker not found or operations were rejected.
        """
        instance_node = self._get_instance(msg.instance_id)
        if instance_node is None:
            return False
        worker_node = instance_node.get_worker(msg.worker_id)
        if worker_node is None:
            return False
        return worker_node.handle_batched_kv_operations(
            msg,
            on_seq_num_out_of_order=self._incr_seq_discontinuity_count,
            is_full_sync=is_full_sync,
        )

    def find_kv(
        self,
        key: int,
        exclude_instance_id: Optional[str] = None,
    ) -> Optional[KVChunkInfo]:
        """
        Find a KV chunk across all workers.

        Args:
            key: The KV chunk key to find.
            exclude_instance_id: Instance ID to exclude
            (all workers in this instance will be excluded).

        Returns: KVChunkInfo if found, None otherwise.
        """
        with self._rwlock.read_lock(timeout=1):
            for instance_id, instance_node in self.instances.items():
                # Exclude all workers in the specified instance
                if (
                    exclude_instance_id is not None
                    and instance_id == exclude_instance_id
                ):
                    continue
                result = instance_node.find_key_simple(key)
                if result is not None:
                    return result
            return None

    def find_kv_with_worker_info(
        self,
        key: int,
        exclude_instance_id: Optional[str] = None,
    ) -> Optional[tuple[KVChunkInfo, Optional[str], set[int]]]:
        """
        Find a KV chunk and return worker info in one lookup.
        Optimized for batched_p2p_lookup to avoid multiple lookups.

        Returns: (KVChunkInfo, peer_init_url, kv_keys) if found, None otherwise.
        """
        # Snapshot instances to avoid RuntimeError if dict changes during iteration
        for instance_id, instance_node in list(self.instances.items()):
            if exclude_instance_id is not None and instance_id == exclude_instance_id:
                continue
            result = instance_node.find_key(key)
            if result is not None:
                return result
        return None

    def get_total_kv_count(self) -> int:
        """Get total count of KV chunks across all workers."""
        # Snapshot to avoid RuntimeError if dict changes during iteration
        total = 0
        for instance_node in list(self.instances.values()):
            total += instance_node.get_total_kv_count()
        return total

    def get_worker_kv_keys(
        self, instance_id: str, worker_id: int, location: str
    ) -> set[int]:
        """Get all KV keys for a specific worker and location."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return set()
        worker_node = instance_node.get_worker(worker_id)
        if worker_node is None:
            return set()
        return worker_node.get_kv_keys(location)

    def clear_worker_kv(
        self, instance_id: str, worker_id: int, location: Optional[str] = None
    ) -> bool:
        """
        Clear all KV chunks for a specific worker and location.
        Returns True if successful, False if worker not found.
        """
        worker_node = self.get_worker(instance_id, worker_id)
        if worker_node is None:
            return False
        with worker_node._lock:
            if location is None:
                worker_node.kv_store.clear()
                worker_node.seq_tracker.clear()
                return True
            if location in worker_node.kv_store:
                del worker_node.kv_store[location]
            if location in worker_node.seq_tracker:
                del worker_node.seq_tracker[location]
        return True

    def clear_all_worker_kv(self, instance_id: str, worker_id: int) -> bool:
        """
        Clear all KV chunks for a worker across all locations.
        Returns True if successful, False if worker not found.
        """
        worker_node = self.get_worker(instance_id, worker_id)
        if worker_node is None:
            return False

        worker_node.clear_kv_store()
        return True
