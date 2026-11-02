# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

SESSION_METADATA = "P2PHANDSHAKE"


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

    def shutdown(self) -> None:
        """The Transfer Engine exposes no teardown entry point."""
