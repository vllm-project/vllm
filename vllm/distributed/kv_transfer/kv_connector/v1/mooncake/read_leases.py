# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefiller-side bookkeeping for blocks handed to pull readers.

The pull prefiller hands block ids to readers and must not reuse those blocks
while any reader may still be reading them. That decision used to be spread
over several dicts on the worker, which made it hard to tell when a block is
free. This module owns the whole state machine:

    none -> handed out -> reported            (reader finished, or failed)
                       -> lease expired       (reader disappeared)

`may_release` is the only question the worker asks before freeing blocks, and
`reported` tells it when every expected reader is done on the happy path.
"""

import time
from collections.abc import Iterator

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.pull_protocol import (
    ReaderKey,
    TransferId,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class ReadLeases:
    """Which readers hold a transfer's blocks, and until when.

    A lease is refreshed while its reader is reading, so an expired lease means
    the reader is gone rather than slow.
    """

    def __init__(self) -> None:
        # transfer -> {reader: last time the reader was known to be alive}
        self._held: dict[TransferId, dict[ReaderKey, float]] = {}
        # transfer -> readers that reported the read as finished
        self._reported: dict[TransferId, set[ReaderKey]] = {}

    def hand_out(
        self, transfer_id: TransferId, reader: ReaderKey, now: float | None = None
    ) -> None:
        """Record that `reader` was given `transfer_id`'s block ids."""
        self._held.setdefault(transfer_id, {})[reader] = (
            time.perf_counter() if now is None else now
        )

    def renew(
        self, transfer_id: TransferId, reader: ReaderKey, now: float | None = None
    ) -> bool:
        """Extend the lease of a reader that is still reading this transfer."""
        readers = self._held.get(transfer_id)
        if readers is None or reader not in readers:
            return False
        readers[reader] = time.perf_counter() if now is None else now
        return True

    def report(self, transfer_id: TransferId, reader: ReaderKey) -> None:
        """Record that `reader` is done with this transfer, whatever the outcome."""
        readers = self._held.get(transfer_id)
        if readers is not None:
            readers.pop(reader, None)
        self._reported.setdefault(transfer_id, set()).add(reader)

    def reported(self, transfer_id: TransferId) -> int:
        """How many readers reported this transfer as done."""
        return len(self._reported.get(transfer_id, ()))

    def may_release(
        self, transfer_id: TransferId, timeout: float, now: float | None = None
    ) -> bool:
        """Whether the blocks of this transfer are free to be released.

        That is the case once no reader holds a live lease: all of them either
        reported in or disappeared, which `timeout` decides.
        """
        readers = self._held.get(transfer_id)
        if not readers:
            return True
        now = time.perf_counter() if now is None else now
        quiet = [reader for reader, seen in readers.items() if now - seen > timeout]
        for reader in quiet:
            logger.warning(
                "Reader %s did not report the read of %s within %d seconds; "
                "releasing its blocks.",
                reader,
                transfer_id,
                timeout,
            )
            readers.pop(reader)
        return not readers

    def holders(self, transfer_id: TransferId) -> list[ReaderKey]:
        """Readers that still hold a lease on this transfer."""
        return list(self._held.get(transfer_id, ()))

    def forget(self, transfer_id: TransferId) -> None:
        """Drop every trace of a transfer that no longer needs holding."""
        self._held.pop(transfer_id, None)
        self._reported.pop(transfer_id, None)

    def __iter__(self) -> Iterator[TransferId]:
        return iter(list(self._held))
