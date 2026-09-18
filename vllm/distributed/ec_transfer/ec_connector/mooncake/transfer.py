# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake data-plane engine and memory-registration ownership.

The wrapper isolates optional-engine initialization, session addressing,
synchronous batched writes, and the lifetime of transient source registrations.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MOONCAKE_IMPORT_ERROR: ImportError | None = None
try:
    from mooncake.engine import TransferEngine
except ImportError as error:
    TransferEngine = None  # type: ignore[misc, assignment]
    _MOONCAKE_IMPORT_ERROR = error


def ensure_mooncake_available() -> None:
    if _MOONCAKE_IMPORT_ERROR is not None:
        raise ImportError(
            "Install mooncake-transfer-engine to use ECMooncakeConnector."
        ) from _MOONCAKE_IMPORT_ERROR


@dataclass
class _SourceRegistration:
    """Retain one transient source range while batches reference it."""

    tensor: torch.Tensor
    nbytes: int
    users: int = 1


class MooncakeTransfer:
    """Own a lazy Mooncake engine and transient memory registrations."""

    def __init__(self, hostname: str, protocol: str) -> None:
        self._hostname = hostname
        self._protocol = protocol
        self._engine: TransferEngine | None = None
        self._engine_lock = threading.Lock()
        self._source_registrations: dict[int, _SourceRegistration] = {}
        self._pending_unregister: dict[int, torch.Tensor] = {}
        self._registration_lock = threading.Lock()
        self._closed = False

    def _ensure_engine(self) -> TransferEngine:
        if self._engine is not None:
            return self._engine
        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            engine = TransferEngine()
            ret = engine.initialize(self._hostname, "P2PHANDSHAKE", self._protocol, "")
            if ret != 0:
                raise RuntimeError("Mooncake TransferEngine initialization failed.")
            self._engine = engine
            logger.info(
                "ECMooncakeConnector TransferEngine ready at %s:%d",
                self._hostname,
                engine.get_rpc_port(),
            )
        return self._engine

    def ensure_ready(self) -> None:
        self._ensure_engine()

    def local_session(self) -> str:
        engine = self._ensure_engine()
        return f"{self._hostname}:{engine.get_rpc_port()}"

    def register_memory(self, tensor: torch.Tensor) -> int:
        return self._ensure_engine().batch_register_memory(
            [tensor.data_ptr()], [tensor.nbytes]
        )

    def unregister_memory(self, tensor: torch.Tensor) -> bool:
        engine = self._ensure_engine()
        address = tensor.data_ptr()
        ret = engine.unregister_memory(address)
        with self._registration_lock:
            if ret != 0:
                logger.error(
                    "Mooncake EC memory unregistration failed for address %d: %d",
                    address,
                    ret,
                )
                self._pending_unregister[address] = tensor
                return False
            self._pending_unregister.pop(address, None)
            return True

    @staticmethod
    def _source_range(tensor: torch.Tensor) -> tuple[int, int]:
        # Encoder batches commonly split one storage into sibling tensor views.
        # Registering each view's exact bytes avoids overlapping memory regions.
        return tensor.data_ptr(), tensor.nbytes

    def acquire_sources(self, tensors: list[torch.Tensor]) -> list[int]:
        ranges: dict[int, tuple[int, torch.Tensor]] = {}
        for tensor in tensors:
            address, nbytes = self._source_range(tensor)
            ranges.setdefault(address, (nbytes, tensor))

        engine = self._ensure_engine()
        acquired: list[int] = []
        new_addresses: list[int] = []
        new_lengths: list[int] = []
        with self._registration_lock:
            for address, (nbytes, tensor) in ranges.items():
                entry = self._source_registrations.get(address)
                if entry is not None:
                    if entry.nbytes != nbytes:
                        raise RuntimeError(
                            "Mooncake EC source storage changed size while registered"
                        )
                    entry.users += 1
                    acquired.append(address)
                    continue
                new_addresses.append(address)
                new_lengths.append(nbytes)
                self._source_registrations[address] = _SourceRegistration(
                    tensor=tensor,
                    nbytes=nbytes,
                )
                acquired.append(address)

            if new_addresses:
                ret = engine.batch_register_memory(new_addresses, new_lengths)
                if ret != 0:
                    for address in acquired:
                        entry = self._source_registrations[address]
                        entry.users -= 1
                        if entry.users == 0:
                            del self._source_registrations[address]
                    raise RuntimeError("Mooncake EC source registration failed")
        return acquired

    def release_sources(self, addresses: list[int]) -> bool:
        if not addresses:
            return True
        with self._registration_lock:
            unused = []
            for address in addresses:
                entry = self._source_registrations.get(address)
                if entry is None:
                    continue
                entry.users -= 1
                if entry.users == 0:
                    unused.append(address)
            if not unused:
                return True
            ret = self._ensure_engine().batch_unregister_memory(unused)
            if ret != 0:
                logger.warning(
                    "Keeping %d EC source tensors registered after Mooncake "
                    "unregistration failure",
                    len(unused),
                )
                return False
            for address in unused:
                del self._source_registrations[address]
                self._pending_unregister.pop(address, None)
            return True

    def write(
        self,
        session: str,
        sources: list[int],
        destinations: list[int],
        lengths: list[int],
    ) -> None:
        """Write one synchronous batch, returning only at terminal status."""
        ret = self._ensure_engine().batch_transfer_sync_write(
            session, sources, destinations, lengths
        )
        if ret != 0:
            raise RuntimeError(
                f"Mooncake EC push to {session} failed with status {ret}"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        engine = self._engine
        if engine is None:
            return
        with self._registration_lock:
            addresses = list(self._source_registrations)
            addresses.extend(self._pending_unregister)
            if not addresses:
                return
            ret = engine.batch_unregister_memory(list(dict.fromkeys(addresses)))
            if ret != 0:
                logger.error("Mooncake EC batch memory unregistration failed: %d", ret)
                return
            self._source_registrations.clear()
            self._pending_unregister.clear()
