# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from typing import Any, Union
import queue
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.token_database import ChunkedTokenDatabase

logger = init_logger(__name__)


class RecordStrategy(ABC):
    """Base class for chunk recording strategies."""

    def __init__(self, chunk_size: int):
        """Initialize the recording strategy.

        Args:
            chunk_size: Size of each token chunk for processing
        """
        self.chunk_size = chunk_size
        self.total_chunks = 0
        self.unique_chunks_count = 0
        self.lock = threading.RLock()

        self._token_db = ChunkedTokenDatabase()
        self._token_db.chunk_size = chunk_size

    def _compute_chunk_hashes(self, token_ids: list[int]) -> list[int]:
        """Compute prefix hashes for all chunks using ChunkedTokenDatabase.

        Args:
            token_ids: List of token IDs to process

        Returns:
            List of hash values (integers) for each chunk.
        """
        chunk_hashes = []
        for _, _, hash_val in self._token_db.process_tokens(
            tokens=token_ids, make_key=False
        ):
            chunk_hashes.append(hash_val)
        return chunk_hashes

    def _compute_chunk_hashes_hex(self, token_ids: list[int]) -> list[str]:
        """Compute prefix hashes for all chunks and return as hex strings.

        Args:
            token_ids: List of token IDs to process

        Returns:
            List of hash values (hex strings) for each chunk.
        """
        chunk_hashes = []
        for hash_val in self._compute_chunk_hashes(token_ids):
            if hash_val < 0:
                hash_val = hash_val & ((1 << 64) - 1)
            chunk_hashes.append(hex(hash_val))
        return chunk_hashes

    @abstractmethod
    def preprocess(self, token_ids: list[int]) -> Any:
        """Preprocess token IDs before recording.

        This method is called to transform raw token IDs into a format suitable
        for the specific recording strategy. For example, it might compute hash
        positions for a bloom filter or convert hashes to hex strings for file storage.

        Args:
            token_ids: List of token IDs to preprocess

        Returns:
            Preprocessed data in strategy-specific format. The return type depends
            on the concrete strategy implementation.
        """
        pass

    @abstractmethod
    def record(self, preprocessed_data: Any, lookup_id: str) -> None:
        """Record the preprocessed chunk data.

        This method performs the actual recording operation using the preprocessed
        data. It should update internal statistics (total_chunks,
        unique_chunks_count) and perform strategy-specific recording (e.g., update
        bloom filter, write to file).

        This method must be thread-safe as it may be called from async worker
        threads.

        Args:
            preprocessed_data: Data returned from preprocess() method
            lookup_id: Unique identifier for this lookup operation
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset all statistics and internal state.

        This method should clear all recorded data and reset counters to their
        initial state. It should be safe to call at any time.
        """

    def get_statistics(self) -> dict:
        """Get current statistics.

        Returns:
            Dictionary containing statistics about recorded chunks, including:
            - total_chunks: Total number of chunks processed
            - unique_chunks: Number of unique chunks seen
            - duplicate_chunks: Number of duplicate chunks
            - reuse_rate: Ratio of duplicate to total chunks
        """
        with self.lock:
            dup = self.total_chunks - self.unique_chunks_count
            base_stats = {
                "total_chunks": self.total_chunks,
                "unique_chunks": self.unique_chunks_count,
                "duplicate_chunks": dup,
                "reuse_rate": dup / self.total_chunks if self.total_chunks > 0 else 0.0,
            }
            return base_stats

    def setup_metrics(self, prometheus_logger) -> None:
        """Setup Prometheus metrics for this strategy.

        Args:
            prometheus_logger: Prometheus logger instance to register metrics with
        """
        prometheus_logger.chunk_statistics_total_chunks.set_function(
            lambda: self.total_chunks
        )
        prometheus_logger.chunk_statistics_unique_chunks.set_function(
            lambda: self.unique_chunks_count
        )
        prometheus_logger.chunk_statistics_reuse_rate.set_function(
            lambda: (self.total_chunks - self.unique_chunks_count) / self.total_chunks
            if self.total_chunks > 0
            else 0.0
        )

    def close(self) -> None:  # noqa: B027
        """Clean up resources.

        This method is called when the strategy is no longer needed. Subclasses
        should override this to clean up any resources (e.g., close file handles).
        """
        pass


class AsyncRecorder:
    """Async processing infrastructure for RecordStrategy.

    This class wraps a RecordStrategy and provides asynchronous processing
    capabilities using a background worker thread and queue.
    """

    def __init__(
        self,
        strategy: RecordStrategy,
        queue_capacity: int = 100000,
        preprocess_in_caller: bool = False,
    ):
        """Initialize the async recorder.

        Args:
            strategy: The RecordStrategy instance to wrap
            queue_capacity: Maximum number of items in the async queue
            preprocess_in_caller: If True, preprocess in caller thread before queueing;
                                 if False, preprocess in worker thread
        """
        self.strategy = strategy
        self.queue_capacity = queue_capacity
        self.preprocess_in_caller = preprocess_in_caller

        self.async_queue: queue.Queue = queue.Queue(maxsize=queue_capacity)
        self.async_shutdown = False
        self.queue_full_blocks = 0
        self.queue_max_size = 0
        self.lock = threading.RLock()

        self.async_worker_thread = threading.Thread(
            target=self._async_worker,
            daemon=True,
            name=f"{strategy.__class__.__name__}AsyncWorker",
        )
        self.async_worker_thread.start()

    def _async_worker(self) -> None:
        """Background worker thread that processes queued items."""
        while not self.async_shutdown:
            try:
                item = self.async_queue.get(timeout=0.1)
                if item is None:
                    break
                data, lookup_id = item
                if self.preprocess_in_caller:
                    preprocessed_data = data
                else:
                    preprocessed_data = self.strategy.preprocess(data)
                self.strategy.record(preprocessed_data, lookup_id)
                self.async_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Async worker error: %s", e, exc_info=True)

        # Process remaining items
        while not self.async_queue.empty():
            try:
                item = self.async_queue.get_nowait()
                if item is not None:
                    data, lookup_id = item
                    if self.preprocess_in_caller:
                        preprocessed_data = data
                    else:
                        preprocessed_data = self.strategy.preprocess(data)
                    self.strategy.record(preprocessed_data, lookup_id)
                self.async_queue.task_done()
            except (queue.Empty, Exception):
                break

    def record_async(
        self, token_ids: Union[torch.Tensor, list[int]], lookup_id: str
    ) -> None:
        """Record token IDs asynchronously.

        Args:
            token_ids: Token IDs to record (tensor or list)
            lookup_id: Unique identifier for this lookup operation
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if self.preprocess_in_caller:
            data = self.strategy.preprocess(token_ids)
        else:
            data = token_ids

        self._queue_item((data, lookup_id))

    def _queue_item(self, item, timeout: float = 10.0) -> None:
        """Add item to async queue with timeout handling."""
        try:
            self.async_queue.put(item, block=True, timeout=timeout)
        except queue.Full:
            with self.lock:
                self.queue_full_blocks += 1
            self.async_queue.put(item, block=True)

    def get_statistics(self) -> dict:
        """Get statistics including async queue metrics.

        Returns:
            Dictionary containing strategy statistics plus async queue metrics
        """
        stats = self.strategy.get_statistics()
        with self.lock:
            queue_size = self.async_queue.qsize()
            self.queue_max_size = max(self.queue_max_size, queue_size)
            stats["async_queue"] = {
                "capacity": self.queue_capacity,
                "current_size": queue_size,
                "max_size_reached": self.queue_max_size,
                "full_blocks": self.queue_full_blocks,
                "utilization": queue_size / self.queue_capacity
                if self.queue_capacity > 0
                else 0.0,
            }
        return stats

    def wait_for_completion(self, timeout: float = 5.0) -> bool:
        """Wait for async queue to be processed.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if queue is empty, False if timeout occurred
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.async_queue.empty():
                time.sleep(0.01)
                if self.async_queue.empty():
                    return True
            time.sleep(0.01)
        return self.async_queue.empty()

    def reset(self) -> None:
        """Reset strategy and clear async queue."""
        self.wait_for_completion(timeout=5.0)
        with self.lock:
            self.strategy.reset()
            self.queue_full_blocks = 0
            self.queue_max_size = 0
            self._clear_queue()

    def _clear_queue(self) -> None:
        """Clear all items from async queue."""
        while not self.async_queue.empty():
            try:
                self.async_queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        """Shutdown async worker and clean up resources."""
        self.async_shutdown = True
        try:
            self.async_queue.put(None, block=False)
        except queue.Full:
            pass

        self.async_worker_thread.join(timeout=5.0)
        if self.async_worker_thread.is_alive():
            logger.warning("Async worker did not stop gracefully")

        self.strategy.close()
