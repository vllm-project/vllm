# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import threading
from typing import Callable

from vllm.logger import init_logger
from vllm.v1.offloading.abstract import LoadStoreSpec

# a single transfer spec (src_blocks_spec_list, dst_blocks_spec_list)
TransferSpec = tuple[list[LoadStoreSpec], list[LoadStoreSpec]]
# transfers are forwarded to workers by (src_medium, dst_medium)
TransferType = tuple[str, str]
# transfer result (job_id, is_success)
TransferResult = tuple[int, bool]

# a transfer execution function (src, dst) -> success
TransferFunction = Callable[[TransferSpec], bool]
# submission queue of transfers (job_id, (src_blocks, dst_blocks)))
SubmissionQueue = queue.Queue[tuple[int, TransferSpec]]
# completion queue of transfers (job_id, is_success)
CompletionQueue = queue.Queue[TransferResult]

logger = init_logger(__name__)


class OffloadingWorker:
    """
    Multithreaded offloading worker.

    This class runs in the worker, and operates using multiple spawned threads.
    It reads KV transfer requests from a dedicated submission queue.
    Transfers are executed using a configurable transfer function, and are
    published to a unified completion queue used by all workers.
    """

    def __init__(self,
                 completion_queue: CompletionQueue,
                 transfer_type: TransferType,
                 transfer_fn: TransferFunction,
                 num_threads: int = 1):
        # queue of pending transfers (job_id, src, dst)
        self.submission_queue = SubmissionQueue()
        self.completion_queue = completion_queue
        self.transfer_type = transfer_type
        self.transfer_fn = transfer_fn
        self.num_threads = num_threads
        self._shutdown_event = threading.Event()
        self._worker_threads: list[threading.Thread] = []

        for thread_idx in range(num_threads):
            t = threading.Thread(target=self.run,
                                 args=(thread_idx, ),
                                 name=f"{transfer_type}-worker-{thread_idx}")
            t.start()
            self._worker_threads.append(t)

        logger.info("Started %d worker threads for transfer type %r",
                    num_threads, transfer_type)

    def run(self, thread_idx: int):
        while True:
            job_id, transfer_spec = self.submission_queue.get()
            if self._shutdown_event.is_set():
                logger.info("Thread %d for transfer type %r finished",
                            thread_idx, self.transfer_type)
                return

            logger.debug("Executing %r transfer %d", self.transfer_type,
                         job_id)

            try:
                success = self.transfer_fn(transfer_spec)
            except Exception as e:
                logger.warning("Exception in %r transfer %d: %r",
                               self.transfer_type,
                               job_id,
                               e,
                               exc_info=True)
                success = False

            logger.debug("Result of %r transfer %d: %r", self.transfer_type,
                         job_id, success)
            self.completion_queue.put((job_id, success))

    def initiate_shutdown(self):
        self._shutdown_event.set()

        # Ensure thread not blocked by submission_queue.get()
        dummy_reference: list[LoadStoreSpec] = []
        dummy_transfer = (-1, (dummy_reference, dummy_reference))
        for _ in range(self.num_threads):
            self.submission_queue.put(dummy_transfer)

    def wait_for_shutdown(self):
        for t in self._worker_threads:
            t.join()


class OffloadingQueueManager:
    """
    OffloadingQueueManager class for managing asynchronous KV data transfers

    This class runs in the worker.
    It sends KV data transfer requests to worker queues, and allows
    collecting back completion statuses.

    The class provides the following primitives:
        register_worker() - registers a new worker (with own thread) to handle
            a specific transfer type
        transfer_async() - adds a new transfer request
            to one of the worker queues. Returns a job ID which can be used
            to track this transfer's completion.
        get_finished() - returns a list of newly finished job IDs.
    """

    def __init__(self):
        self.workers: dict[TransferType, OffloadingWorker] = {}
        self.completion_queue = CompletionQueue()

    def register_worker(self,
                        src_cls: type[LoadStoreSpec],
                        dst_cls: type[LoadStoreSpec],
                        transfer_fn: TransferFunction,
                        num_threads: int = 1):
        """
        Registers a new worker (with own threads).

        Args:
            src_cls: the source type of transfers handled by this worker.
            dst_cls: the destination type of transfers handled by this worker.
            transfer_fn: the function that will be called
                to execute a transfer.
            num_threads: the number of threads to spawn for executing
                this type of transfers.
        """
        transfer_type = (src_cls.medium(), dst_cls.medium())
        assert transfer_type not in self.workers
        self.workers[transfer_type] = OffloadingWorker(self.completion_queue,
                                                       transfer_type,
                                                       transfer_fn,
                                                       num_threads)

    def transfer_async(self, job_id: int, spec: TransferSpec):
        """
        Initiates an asynchronous transfer of KV data.

        Args:
            job_id: a unique ID that will be used when notifying back on
                transfer completion.
            spec: the (src, dst) spec of the KV data transfer.
                Assumes all sources are of the same medium,
                and the same for the destinations.
        """
        src, dst = spec
        assert src and dst

        transfer_type = (src[0].medium(), dst[0].medium())
        worker = self.workers.get(transfer_type)
        assert worker is not None

        worker.submission_queue.put((job_id, spec))
        logger.debug("Submitted %r transfer %d: %r", transfer_type, job_id,
                     spec)

    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of (job_id, success) of transfers.
        """
        finished = []
        while True:
            try:
                item = self.completion_queue.get_nowait()
                finished.append(item)
            except queue.Empty:
                break
        return finished

    def shutdown(self):
        """Shutdown, cleaning up spawned workers."""
        for worker in self.workers.values():
            worker.initiate_shutdown()
        for worker in self.workers.values():
            worker.wait_for_shutdown()

    def __del__(self):
        self.shutdown()
