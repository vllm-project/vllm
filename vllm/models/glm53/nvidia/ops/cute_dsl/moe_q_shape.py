# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-visible state sizes for the fixed four- and eight-row W8A16 MoE."""


class Flag:
    """``Flags`` (u32 words)."""

    WORDS = 14952
    ACT = 144
    PLANES = 288
    LOCAL_O = 432
    DONE = 576
    ROUND = 11601
    PAD_FIRST = 11732
    PAGE = 1024
    PAD_SHARED = 128
    PAD_ROUND = 256
    #: The words after the three pages: each SM's mirror copy (0 or 1).
    SM_DOMAIN = PAD_FIRST + 3 * PAGE
    #: Words from a mirrored word to its copy in the other locality domain.
    MIRROR = 1 << 21


class Q8Flag(Flag):
    """Eight-row logit parity pages with the same mirrored counter prefix."""

    PAGE = 2048
    SM_DOMAIN = Flag.PAD_FIRST + 3 * PAGE
    WORDS = SM_DOMAIN + 148


class Q8Shape:
    """Fixed eight-row capacity and communication sizes shared by host and kernel."""

    ROWS = 8
    ROUNDS = 8
    MAX_UNION = ROWS * 8
    PLANES = MAX_UNION + 1
    RING_SLOTS = 3
    MAILBOX_BYTES = 7 * 2 * 128 * ROWS * 128
    SEAM_MAILBOX_BYTES = 8 * 2 * 128 * ROWS * 128


def flag_layout(rows: int = 4) -> type[Flag]:
    """Return the fixed-row flag layout used by kernel and locality allocation."""
    if rows == 4:
        return Flag
    if rows == Q8Shape.ROWS:
        return Q8Flag
    raise ValueError(f"W8A16 MoE supports 4 or 8 rows, got {rows}")
