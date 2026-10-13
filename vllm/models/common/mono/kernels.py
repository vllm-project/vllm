# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The bottom tier: one rank's persistent launches, as the runtime sees them.

Besides the FlyDSL itself, this is what a new MonoKernel writer implements. The
runtime reserves the memory these ask for, moves them to the next step's mailbox
slot once per step, and reads the polls that expired.

Every member has a default, so an implementation overrides only what is real for
it: kernels that move their own mailbox epoch say nothing about steps, kernels
that allocate their own scratch say nothing about bytes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MonoKernels(Protocol):
    """The launches of one :class:`MonoRuntime` key."""

    def scratch_bytes(self, width: int) -> int:
        """Device scratch a step of ``width`` rows needs, 0 to keep its own."""
        return 0

    def peer_bytes(self) -> int:
        """Symmetric peer memory the in-kernel reductions need, 0 to keep its own.

        Reserved once per runtime, at the first key whose kernels ask for it.
        """
        return 0

    def advance_step(self) -> None:
        """Move to the next step's mailbox slot, once per step the runtime goes."""
        return None

    def poll_error(self, clear: bool = False) -> tuple[str, ...]:
        """Stages whose bounded mailbox waits expired since the last clear."""
        return ()
