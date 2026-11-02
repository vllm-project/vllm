# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

# Bumped whenever the peer payload changes shape. Peers refuse to transfer
# across a mismatch instead of failing somewhere inside the engine.
WIRE_VERSION = 1


class TransferEngineKind(Enum):
    TE = "te"
    TENT = "tent"


class TransferOp(Enum):
    READ = "read"
    WRITE = "write"


class TransferState(Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class MemoryRegion:
    base_addr: int
    length: int


@dataclass(frozen=True)
class RangeRequest:
    local_addr: int
    remote_addr: int
    length: int


@dataclass(frozen=True)
class TransferEvent:
    state: TransferState
    error: str | None = None


def format_peer_address(hostname: str, port: int) -> str:
    """Names a peer the way both engines parse it.

    IPv6 addresses are bracketed, which is what the Transfer Engine's
    `parseHostNameWithPort` and TENT's segment naming both expect.
    """
    if ":" in hostname:
        return f"[{hostname}]:{port}"
    return f"{hostname}:{port}"


class TransferBackend(ABC):
    """Moves KV blocks between two workers and reports per-range outcomes."""

    @property
    @abstractmethod
    def kind(self) -> TransferEngineKind:
        """Engine backing this backend. Peers compare it during the handshake."""

    @property
    @abstractmethod
    def rpc_port(self) -> int:
        """Port peers use to reach this worker's engine."""

    @abstractmethod
    def register_regions(self, regions: Sequence[MemoryRegion]) -> None:
        """Pins local memory for transfers.

        An empty sequence is a no-op. Raises RuntimeError when the engine
        rejects the memory.
        """

    @abstractmethod
    def unregister_regions(self, regions: Sequence[MemoryRegion]) -> None:
        """Releases memory pinned by `register_regions`.

        An empty sequence is a no-op.
        """

    @abstractmethod
    def transfer(
        self, peer: str, op: TransferOp, ranges: Sequence[RangeRequest]
    ) -> list[TransferEvent]:
        """Moves every range and returns one terminal event for each.

        The result lines up with `ranges` by position, and an empty request
        returns an empty list. Blocks until every range reached a terminal
        state. Called from worker threads, so an implementation must accept
        concurrent calls.

        An engine fault is reported as an event, not raised: a caller books
        the blocks it submitted against the outcome of every range, and an
        exception escaping here would leave those blocks pinned.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Stops the backend.

        The caller stops submitting before calling this, but work already
        queued on its worker threads is not cancelled, so a `transfer` call may
        still be running or start afterwards. An implementation must tolerate
        that.
        """
