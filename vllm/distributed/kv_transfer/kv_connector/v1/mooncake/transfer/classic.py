# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from collections.abc import Callable, Sequence
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.engine import (
    MemoryRegion,
    RangeRequest,
    TransferBackend,
    TransferEvent,
    TransferOp,
    TransferState,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

SESSION_METADATA = "P2PHANDSHAKE"


class ClassicTransferBackend(TransferBackend):
    """Wraps `mooncake.engine.TransferEngine` behind the backend interface.

    The classic engine reports one status per batch, so every range of a
    submission shares the same outcome. A submission settles inside `submit`,
    so its events are returned by the next `poll` on the submitting thread.
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
        self._hostname = hostname
        self._rpc_port = self._engine.get_rpc_port()
        self._pending = threading.local()

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def hostname(self) -> str:
        return self._hostname

    @property
    def rpc_port(self) -> int:
        return self._rpc_port

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

    def submit(self, peer: str, op: TransferOp, ranges: Sequence[RangeRequest]) -> None:
        if not ranges:
            return
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
        events = [
            TransferEvent(
                range_id=request.range_id,
                state=state,
                transferred_bytes=request.length if error is None else None,
                error=error,
            )
            for request in ranges
        ]
        self._pending_events().extend(events)

    def poll(self) -> list[TransferEvent]:
        events: list[TransferEvent] = self._pending_events()
        self._pending.events = []
        return events

    def _pending_events(self) -> list[TransferEvent]:
        events: list[TransferEvent] | None = getattr(self._pending, "events", None)
        if events is None:
            events = []
            self._pending.events = events
        return events

    def shutdown(self) -> None:
        """The classic engine exposes no teardown entry point."""
