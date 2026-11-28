# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Classify failed tier I/O jobs by errno. Used only after a job fails."""

import enum
import errno


class IOFailureKind(enum.Enum):
    """Why a tier I/O job failed, as far as its errno tells."""

    CAPACITY = "capacity"
    """The tier is out of space: `ENOSPC`, `EDQUOT`, `EFBIG`."""

    TRANSIENT = "transient"
    """Likely to succeed if tried again and not a statement about the tier's
    health: an interrupted or would-block call, or exhaustion of a process- or
    host-wide resource (file descriptors, kernel memory)."""

    PERSISTENT = "persistent"
    """Any other errno: the tier rejects the operation and will keep doing so
    until something outside vLLM changes (e.g. `EIO`, `EACCES`,
    `EROFS`)."""

    UNKNOWN = "unknown"
    """The tier did not report an errno."""


_CAPACITY_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT, errno.EFBIG})

_TRANSIENT_ERRNOS = frozenset(
    {
        errno.EINTR,
        errno.EAGAIN,
        errno.EWOULDBLOCK,
        errno.EBUSY,
        errno.EMFILE,
        errno.ENFILE,
        errno.ENOMEM,
    }
)


def classify_errno(err: int | None) -> IOFailureKind:
    """Map the errno of a failed tier I/O job to an `IOFailureKind`."""
    if err is None:
        return IOFailureKind.UNKNOWN
    if err in _CAPACITY_ERRNOS:
        return IOFailureKind.CAPACITY
    if err in _TRANSIENT_ERRNOS:
        return IOFailureKind.TRANSIENT
    return IOFailureKind.PERSISTENT


def indicates_tier_pressure(kind: IOFailureKind) -> bool:
    """Whether a store failure of this kind says the tier should receive fewer
    stores. Transient failures do not: they say nothing about the tier and are
    candidates for retry instead (#54363 item 4)."""
    return kind is not IOFailureKind.TRANSIENT
