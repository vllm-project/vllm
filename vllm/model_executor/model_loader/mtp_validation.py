# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped controls for MTP checkpoint completeness validation."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_mtp_completeness_check_enabled: ContextVar[bool] = ContextVar(
    "mtp_completeness_check_enabled", default=True
)


def is_mtp_completeness_check_enabled() -> bool:
    """Return whether MTP completeness validation is enabled in this scope."""
    return _mtp_completeness_check_enabled.get()


@contextmanager
def disable_mtp_completeness_check() -> Iterator[None]:
    """Temporarily disable MTP completeness validation for one weight load."""
    token = _mtp_completeness_check_enabled.set(False)
    try:
        yield
    finally:
        _mtp_completeness_check_enabled.reset(token)
