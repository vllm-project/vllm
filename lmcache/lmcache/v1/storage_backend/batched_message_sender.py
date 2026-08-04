# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, List, Optional
import queue
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.observability import PrometheusLogger
from lmcache.v1.cache_controller.message import (
    BatchedKVOperationMsg,
    KVOpEvent,
    OpType,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class BatchedMessageSender:
    """
    Batched message sender for KVOperation.

    This class accumulates KV admit/evict messages and sends them in batches
    to reduce communication overhead. Messages are flushed when either:
    1. The batch size threshold is reached (default: 50 messages)
    2. The timeout period expires (default: 0.01 seconds)

    Each message is assigned a unique, monotonically increasing sequence number
    to enable the receiver to detect missing or out-of-order messages.

    Design rationale:
    - Uses a SINGLE queue for both admit and evict messages to maintain strict
      order consistency. This is critical because operations like
      admit(key) -> evict(key) -> admit(key) must be processed in exact order
      to avoid race conditions and state inconsistencies on the receiver side.

    Thread-safe: Uses locks to protect internal queue and sequence counter.

    Args:
        metadata: Metadata for the worker
        config: Configuration for the worker
        location: Location of the worker
        lmcache_worker: The worker to send messages to. If None, batching is disabled.
    """

    def __init__(
        self,
        metadata: LMCacheMetadata,
        config: LMCacheEngineConfig,
        location: str,
        lmcache_worker: "LMCacheWorker",
    ) -> None:
        self.batch_size = config.get_extra_config_value("kv_msg_batch_size", 50)
        self.batch_timeout = config.get_extra_config_value("kv_msg_batch_timeout", 0.01)
        self.lmcache_worker = lmcache_worker
        self.metadata = metadata
        self.config = config

        # Common fields shared by all operations in the batch
        self.instance_id = config.lmcache_instance_id
        self.worker_id = metadata.worker_id
        self.location = location

        # Use thread-safe queue for producer-consumer pattern
        self.message_queue: queue.Queue[KVOpEvent] = queue.Queue()
        self.sequence_number = 0
        self.sequence_lock = threading.Lock()

        # Condition variable for coordinating producer and consumer
        self.cv = threading.Condition()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self._start_background_thread()

        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Setup metrics for monitoring queue size."""
        prometheus_logger = PrometheusLogger.GetOrCreate(
            self.metadata,
            config=self.config,
        )
        prometheus_logger.kv_msg_queue_size.set_function(
            lambda: self.message_queue.qsize()
        )

    def _start_background_thread(self):
        """Start background thread for periodic flushing."""
        self.running = True
        self.thread = threading.Thread(
            target=self._consumer_loop, daemon=True, name="batched-msg-sender-thread"
        )
        self.thread.start()

    def _consumer_loop(self):
        """Consumer loop that drains queue and sends batched messages."""
        while self.running:
            with self.cv:
                # Wait for timeout or notification from producer
                self.cv.wait(timeout=self.batch_timeout)

                # Check if we have messages to process while holding the lock
                # This prevents race conditions but we'll release lock
                # before blocking operations
                if self.message_queue.empty():
                    continue

            # Drain the queue without holding the lock to avoid blocking producers
            # This improves performance during the actual message processing
            self._drain_and_send()

    def _get_next_sequence_number(self) -> int:
        """Get next sequence number for message tracking.

        Thread-safe: Uses dedicated lock for sequence number generation.
        """
        with self.sequence_lock:
            seq = self.sequence_number
            self.sequence_number += 1
            return seq

    def add_kv_op(
        self,
        op_type: OpType,
        key: int,
    ):
        """Add a KV operation to the batch queue.

        Producer method: Adds operation to queue and notifies consumer
        when batch size threshold is reached.

        Args:
            op_type: Operation type (ADMIT or EVICT)
            key: Chunk hash key
        """
        # Create operation without sequence number (will be assigned during drain)
        op = KVOpEvent(op_type=op_type, key=key, seq_num=-1)

        # Thread-safe queue put
        self.message_queue.put(op)

        # Notify consumer if batch size threshold is reached
        if self.message_queue.qsize() >= self.batch_size:
            with self.cv:
                self.cv.notify()

    def _drain_and_send(self):
        """Drain the queue and send all messages in a batch.

        This method is called by the consumer thread to collect all pending
        operations from the queue and send them as a single batched message.
        """
        ops_to_send: List[KVOpEvent] = []

        # Drain all messages from the queue using blocking get with timeout
        # This ensures we don't miss any messages due to race conditions
        while True:
            try:
                # Use a small timeout to avoid blocking indefinitely
                op = self.message_queue.get(timeout=0.001)
                # Assign sequence number at drain time to ensure strict ordering
                op.seq_num = self._get_next_sequence_number()
                ops_to_send.append(op)
            except queue.Empty:
                # Queue is empty, break the loop
                break

        if not ops_to_send:
            return

        try:
            # Ensure common fields are set
            assert self.instance_id is not None, "instance_id must be set"
            assert self.worker_id is not None, "worker_id must be set"
            assert self.location is not None, "location must be set"

            # Create batched message with common fields and lightweight operations
            # This reduces redundancy: common fields are sent once instead of N times
            batched_msg = BatchedKVOperationMsg(
                instance_id=self.instance_id,
                worker_id=self.worker_id,
                location=self.location,
                operations=ops_to_send,
            )
            self.lmcache_worker.put_msg(batched_msg)
        finally:
            # Mark all tasks as done regardless of success/failure
            # This ensures flush() doesn't hang if put_msg fails
            for _ in ops_to_send:
                self.message_queue.task_done()

    def flush(self):
        """Manually flush all pending messages.

        This method ensures all pending messages in the queue are processed
        before returning. It triggers the consumer thread and waits for the
        queue to be empty.
        """
        with self.cv:
            self.cv.notify()

        self.message_queue.join()

    def close(self):
        """Close the batched message sender and flush remaining messages."""
        self.flush()
        self.running = False

        # Wake up consumer thread to exit
        with self.cv:
            self.cv.notify()

        # Wait for thread to finish
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                logger.warning(
                    "Batched message sender thread did not terminate within timeout"
                )
