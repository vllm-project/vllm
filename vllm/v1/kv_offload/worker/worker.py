# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import GroupTransfer, LoadStoreSpec

# transfers are forwarded to workers by (src_medium, dst_medium)
TransferType = tuple[str, str]


@dataclass
class TransferSpec:
    """Direction tagged per group transfer descriptor.

    groups: one GroupTransfer per KV cache group, positionally aligned with
        kv_cache_groups. Groups with no blocks to move have empty block_ids.
    is_store: True = GPU to offloaded medium, False = offloaded medium to GPU.
        (Task 7 #33689 will replace this flag with explicit submit_store /
        submit_load methods. is_store bridges the gap for till then.)
    """

    groups: Sequence[GroupTransfer]
    is_store: bool


logger = init_logger(__name__)


@dataclass
class TransferResult:
    job_id: int
    success: bool
    transfer_size: int | None = None  # Size in bytes
    transfer_time: float | None = None
    transfer_type: TransferType | None = None


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
            spec: per group TransferSpec describing which blocks to move
                and in which direction (spec.is_store: True = GPU to offloaded,
                False = offloaded to GPU).

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

    @abstractmethod
    def wait(self, job_ids: set[int]) -> None:
        """
        Wait for jobs to finish (blocking).
        Args:
            job_ids: The set of job IDs to wait for.
        """

    def shutdown(self) -> None:
        """Shutdown the handler and release any resources."""
        return


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
            spec: the per-group transfer spec (direction encoded in is_store).

        Returns:
            True if transfer was submitted successfully.
        """
        assert spec.groups, "TransferSpec must have at least one group"
        offload_medium = spec.groups[0].offload_spec.medium()
        if spec.is_store:
            transfer_type = ("GPU", offload_medium)
        else:
            transfer_type = (offload_medium, "GPU")
        handler = self.transfer_type_to_handler.get(transfer_type)
        assert handler is not None, (
            f"No handler registered for transfer type {transfer_type!r}"
        )
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
            logger.debug("Submitted %r transfer %d", transfer_type, job_id)
        return success

    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of TransferResults
        """
        finished = []
        for handler in self.handlers:
            finished.extend(handler.get_finished())
        return finished

    def wait(self, job_ids: set[int]) -> None:
        """
        Wait for jobs to finish (blocking).

        Args:
            job_ids: The set of job IDs to wait for.
        """
        for handler in self.handlers:
            handler.wait(job_ids)

    def shutdown(self) -> None:
        for handler in self.handlers:
            handler.shutdown()
