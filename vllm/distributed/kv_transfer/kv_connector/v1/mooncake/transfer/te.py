# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.backend import (
    BatchTransferResult,
    MemoryRegion,
    RangeRequest,
    TransferBackend,
    TransferEvent,
    TransferOp,
    TransferState,
)
from vllm.logger import init_logger

SESSION_METADATA = "P2PHANDSHAKE"
logger = init_logger(__name__)


class TransferEngineBackend(TransferBackend):
    """Wraps `mooncake.engine.TransferEngine` behind the backend interface.

    The Transfer Engine reports one status per batch, so every range of a
    submission shares the same outcome.
    """

    def __init__(
        self,
        *,
        engine_cls: Callable[[], Any],
        hostname: str,
        protocol: str,
        device_name: str,
    ) -> None:
        self._engine = engine_cls()
        ret_value = self._engine.initialize(
            hostname, SESSION_METADATA, protocol, device_name
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")
        self._rpc_port = self._engine.get_rpc_port()
        array_write = getattr(self._engine, "batch_transfer_sync_write_arrays", None)
        self._array_write = array_write if callable(array_write) else None
        if self._array_write is None:
            logger.info(
                "Mooncake Transfer Engine does not expose a callable "
                "batch_transfer_sync_write_arrays; falling back to "
                "batch_transfer_sync_write (list API). Install "
                "mooncake-transfer-engine v0.3.14 or newer to use the array API."
            )

    @property
    def rpc_port(self) -> int:
        return self._rpc_port

    @property
    def engine(self) -> Any:
        """The `mooncake.engine.TransferEngine` this backend drives."""
        return self._engine

    def register_regions(self, regions: Sequence[MemoryRegion]) -> None:
        if not regions:
            return
        ret_value = self._engine.batch_register_memory(
            [region.base_addr for region in regions],
            [region.length for region in regions],
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

    def unregister_regions(self, regions: Sequence[MemoryRegion]) -> None:
        if not regions:
            return
        ret_value = self._engine.batch_unregister_memory(
            [region.base_addr for region in regions]
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory unregistration failed.")

    def transfer(
        self, peer: str, op: TransferOp, ranges: Sequence[RangeRequest]
    ) -> list[TransferEvent]:
        if not ranges:
            return []
        src_ptrs = [request.local_addr for request in ranges]
        dst_ptrs = [request.remote_addr for request in ranges]
        lengths = [request.length for request in ranges]
        if op is TransferOp.WRITE:
            ret_value = self._engine.batch_transfer_sync_write(
                peer, src_ptrs, dst_ptrs, lengths
            )
        else:
            ret_value = self._engine.batch_transfer_sync_read(
                peer, src_ptrs, dst_ptrs, lengths
            )
        state = TransferState.COMPLETED if ret_value == 0 else TransferState.FAILED
        error = (
            None if ret_value == 0 else f"Mooncake transfer engine returned {ret_value}"
        )
        return [TransferEvent(state=state, error=error) for _ in ranges]

    def transfer_sync_arrays(
        self,
        peer: str,
        op: TransferOp,
        local_addrs: NDArray[np.uint64],
        remote_addrs: NDArray[np.uint64],
        lengths: NDArray[np.uint64],
    ) -> BatchTransferResult:
        for column in (local_addrs, remote_addrs, lengths):
            if (
                column.ndim != 1
                or column.dtype != np.uint64
                or not column.flags.c_contiguous
            ):
                raise ValueError(
                    "Transfer columns must be contiguous 1D native uint64."
                )
        num_ranges = len(local_addrs)
        if len(remote_addrs) != num_ranges or len(lengths) != num_ranges:
            raise ValueError("Transfer columns must have equal lengths.")
        if num_ranges == 0:
            return BatchTransferResult(0, TransferState.COMPLETED)

        if op is TransferOp.WRITE and self._array_write is not None:
            ret_value = self._array_write(peer, local_addrs, remote_addrs, lengths)
        else:
            src_ptrs = local_addrs.tolist()
            dst_ptrs = remote_addrs.tolist()
            sizes = lengths.tolist()
            if op is TransferOp.WRITE:
                ret_value = self._engine.batch_transfer_sync_write(
                    peer, src_ptrs, dst_ptrs, sizes
                )
            else:
                ret_value = self._engine.batch_transfer_sync_read(
                    peer, src_ptrs, dst_ptrs, sizes
                )
        if ret_value == 0:
            return BatchTransferResult(num_ranges, TransferState.COMPLETED)
        return BatchTransferResult(
            num_ranges,
            TransferState.FAILED,
            f"Mooncake transfer engine returned {ret_value}",
        )

    def shutdown(self) -> None:
        """Does nothing: the Transfer Engine exposes no teardown entry point.

        The interface keeps the method so a caller stops either backend the same
        way.
        """
