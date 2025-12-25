# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import LoadStoreSpec

# a single transfer spec (src_blocks_spec, dst_blocks_spec)
TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]
# transfers are forwarded to workers by (src_medium, dst_medium)
TransferType = tuple[str, str]
# transfer result (job_id, success)
TransferResult = tuple[int, bool]

logger = init_logger(__name__)


@dataclass
class TransferStats:
    num_blocks: int
    time: float  # Can be start_time or duration
    transfer_type: TransferType


class OffloadingHandler(ABC):
    """
    OffloadingHandler class for managing asynchronous KV data transfers

    This class runs in the worker.
    It kicks off async KV data transfer requests, and allows
    collecting back completion statuses.

    The class provides the following primitives:
        transfer_async() - kicks off a new transfer job
        get_finished() - returns a list of newly finished job IDs.
    """

    @abstractmethod
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Initiates an asynchronous transfer of KV data.

        Args:
            job_id: a unique ID that will be used when notifying back on
                transfer completion.
            spec: the (src, dst) spec of the KV data transfer.

        Returns:
            True if transfer was submitted successfully.
        """
        pass

    @abstractmethod
    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of (job_id, success) of transfers.
        """
        pass


class OffloadingWorker:
    """
    OffloadingWorker class for managing asynchronous KV data transfers
    using multiple OffloadingHandlers

    This class runs in the worker.
    It kicks off async KV data transfer requests, by delegating
    to one of its registered OffloadingHandlers, based on the transfer type.

    The class provides the following primitives:
        register_handler() - registers a new handler to handle
            a specific transfer type
        transfer_async() - kicks off a new transfer job
            using one of the registered handlers.
        get_finished() - returns a list of newly finished job IDs
            from all handlers.
    """

    def __init__(self):
        self.handlers: set[OffloadingHandler] = set()
        self.transfer_type_to_handler: dict[TransferType, OffloadingHandler] = {}
        # _transfer_stats: job_id -> num_blocks, time, transfer type
        self._transfer_stats: dict[int, TransferStats] = {}

    def register_handler(
        self,
        src_cls: type[LoadStoreSpec],
        dst_cls: type[LoadStoreSpec],
        handler: OffloadingHandler,
    ) -> None:
        """
        Registers a new handler.

        Args:
            src_cls: the source type of transfers handled by this handler.
            dst_cls: the destination type of transfers handled by this handler.
            handler: the handler that will handle transfers.
        """
        transfer_type = (src_cls.medium(), dst_cls.medium())
        assert transfer_type not in self.transfer_type_to_handler
        self.handlers.add(handler)
        self.transfer_type_to_handler[transfer_type] = handler

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Initiates an asynchronous transfer of KV data.

        Args:
            job_id: a unique ID that will be used when notifying back on
                transfer completion.
            spec: the (src, dst) spec of the KV data transfer.

        Returns:
            True if transfer was submitted successfully.
        """
        src, dst = spec
        transfer_type = (src.medium(), dst.medium())
        handler = self.transfer_type_to_handler.get(transfer_type)
        assert handler is not None
        start_time = time.perf_counter()
        try:
            success = handler.transfer_async(job_id, spec)
        except Exception as e:
            logger.warning(
                "Exception in %r transfer %d: %r",
                transfer_type,
                job_id,
                e,
                exc_info=True,
            )
            return False

        if not success:
            logger.warning("Failed to submit %r transfer %d", transfer_type, job_id)
        else:
            logger.debug("Submitted %r transfer %d: %r", transfer_type, job_id, spec)
        num_blocks = src.num_blocks
        self._transfer_stats[job_id] = TransferStats(
            num_blocks, start_time, transfer_type
        )
        return success

    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of (job_id, success) of transfers.
        """
        finished = []
        finish_time = time.perf_counter()
        for handler in self.handlers:
            finished.extend(handler.get_finished())
        for job_id, success in finished:
            start_time = self._transfer_stats[job_id].time
            self._transfer_stats[job_id].time = finish_time - start_time
        return finished

    def get_stats(self, job_id: int) -> tuple[int, float, TransferType]:
        stats = self._transfer_stats.pop(job_id)
        num_blocks = stats.num_blocks
        transfer_time = stats.time
        transfer_type = stats.transfer_type
        return num_blocks, transfer_time, transfer_type
