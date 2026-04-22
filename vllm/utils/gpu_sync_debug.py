# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import sys
from collections.abc import Callable
from contextlib import contextmanager

import torch

import vllm.envs as envs
from vllm.platforms import current_platform

_GPU_SYNC_ALLOWED_COUNTS: dict[tuple[str, int], int] = {}


@contextmanager
def _suppress_gpu_sync_check(prev_mode: int):
    torch.cuda.set_sync_debug_mode(0)
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)


@contextmanager
def _noop_cm():
    yield


def gpu_sync_allowed(count: int | None = None):
    """Context manager that suppresses `torch.cuda.set_sync_debug_mode` for the
    duration of the `with` block.

    If `count` is given, only the first `count` entries from this call site
    suppress the sync check; subsequent entries from the same site are no-ops
    so any further GPU syncs will be reported. The "site" is the caller's
    (filename, lineno), so different `with gpu_sync_allowed(count=N):` lines
    track independent counters automatically.
    """
    prev_mode = torch.cuda.get_sync_debug_mode()
    if not prev_mode:
        return _noop_cm()
    if count is not None:
        frame = sys._getframe(1)
        key = (frame.f_code.co_filename, frame.f_lineno)
        used = _GPU_SYNC_ALLOWED_COUNTS.get(key, 0)
        if used >= count:
            return _noop_cm()
        _GPU_SYNC_ALLOWED_COUNTS[key] = used + 1
    return _suppress_gpu_sync_check(prev_mode)


def _parse_gpu_sync_check_mode() -> tuple[str, bool] | None:
    """Returns (mode, exclude_prefill) parsed from `VLLM_GPU_SYNC_CHECK`, or
    None if the env var is unset."""
    val = envs.VLLM_GPU_SYNC_CHECK
    if val is None:
        return None
    parts = [p.strip().lower() for p in val.split(",")]
    return parts[0], "exclude_prefill" in parts[1:]


def with_gpu_sync_check(check_if: Callable[..., bool] | None = None):
    """Decorator factory that enables `torch.cuda.set_sync_debug_mode` around
    `fn` when `VLLM_GPU_SYNC_CHECK` is set. If the env var is suffixed with
    `,exclude_prefill` and `check_if(*args, **kwargs)` returns False, the
    check is skipped for that call.

    The env var is parsed once at decoration time; this module is imported
    lazily after `VllmConfig.__post_init__` has finalized `VLLM_GPU_SYNC_CHECK`.
    """

    def decorator(fn):
        parsed = _parse_gpu_sync_check_mode()
        if parsed is None or not current_platform.is_cuda_alike():
            return fn
        mode, exclude_prefill = parsed

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if (
                exclude_prefill
                and check_if is not None
                and not check_if(*args, **kwargs)
            ):
                return fn(*args, **kwargs)

            prev_mode = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode(mode)
            try:
                return fn(*args, **kwargs)
            finally:
                torch.cuda.set_sync_debug_mode(prev_mode)

        return wrapper

    return decorator
