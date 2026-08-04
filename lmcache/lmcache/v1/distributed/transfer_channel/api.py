# SPDX-License-Identifier: Apache-2.0
"""
Data types exposed by the transfer channel abstraction.
"""

# Standard
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TransferChannelAddress:
    """A transfer-channel-specific address, with starting position and size
    (in bytes).

    A single address corresponds to a single memory object.

    Note:
        Currently, this is no difference between a tuple of (offset, size).
        But we are wrapping this as a class for future extensibility (e.g.
        support non L1 memory).
    """

    offset: int
    """ The offset (in bytes) against the L1 base address. """

    size: int
    """ The size (in bytes) of the memory object. """

    def is_valid(self) -> bool:
        """Whether the address is valid (non-negative offset and size)."""
        return self.offset >= 0 and self.size > 0


@dataclass
class TransferChannelReadResult:
    """Result of querying a submitted read task."""

    finished: bool
    """ Whether the transfer has reached a terminal state (done or errored). """

    succeeded_mask: list[bool] = field(default_factory=list)
    """ Per-object success flags, aligned with the submitted addresses. Empty while
    the transfer is still in flight. """

    def is_finished(self) -> bool:
        """Whether the transfer reached a terminal state (done or errored)."""
        return self.finished
