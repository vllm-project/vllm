# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mark the worker's `stage_weights` thread, which runs while the engine serves."""

import threading

__all__ = ["check_not_staging", "mark_staging_thread"]

_local = threading.local()


def mark_staging_thread() -> None:
    """Mark the current thread as the staging thread (executor initializer)."""
    _local.staging = True


def check_not_staging(what: str) -> None:
    """Raise if called from the staging thread: it must not touch the live model."""
    if getattr(_local, "staging", False):
        raise RuntimeError(
            f"{what} cannot run in stage_weights: staging runs while the engine "
            "serves and must not touch the live model"
        )
