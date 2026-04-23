# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import sys
from contextlib import contextmanager

import torch

import vllm.envs as envs
from vllm.platforms import current_platform

_GPU_SYNC_ALLOWED_COUNTS: dict[tuple[str, int], int] = {}

# Global sync-check gate. Off during engine setup (model load, KV cache
# init, warmup/compile) so first-compile and lazy-init syncs pass through;
# flipped on by `enable_gpu_sync_check()` at the end of
# `GPUWorker.compile_or_warm_up_model`, after which `with_gpu_sync_check`-
# decorated functions activate the configured debug mode.
_sync_check_enabled: bool = False


def enable_gpu_sync_check() -> None:
    """Flip the sync-check gate on. Call once per worker, after warmup /
    first-compile is complete."""
    global _sync_check_enabled
    _sync_check_enabled = True


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


if current_platform.is_cuda_alike():

    def gpu_sync_allowed(count: int | None = None):
        """Context manager that suppresses `torch.cuda.set_sync_debug_mode` for the
        duration of the `with` block.

        If `count` is given, only the first `count` entries from this call site
        suppress the sync check; subsequent entries from the same site are no-ops
        so any further GPU syncs will be reported. The "site" is the caller's
        (filename, lineno), so different `with gpu_sync_allowed(count=N):` lines
        track independent counters automatically.
        """
        if torch.compiler.is_compiling():
            return _noop_cm()
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

    def with_gpu_sync_check(fn):
        """Decorator that enables `torch.cuda.set_sync_debug_mode` around `fn`
        when `VLLM_GPU_SYNC_CHECK` is set *and* the gate has been flipped by
        `enable_gpu_sync_check()`. Before the gate flips (i.e. during
        engine setup / warmup) the decorated function runs as-is.

        The env var is parsed once at decoration time; this module is imported
        lazily after `VllmConfig.__post_init__` has finalized `VLLM_GPU_SYNC_CHECK`.
        """
        mode = envs.VLLM_GPU_SYNC_CHECK
        if mode is None:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _sync_check_enabled:
                return fn(*args, **kwargs)
            prev_mode = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode(mode)
            try:
                return fn(*args, **kwargs)
            finally:
                torch.cuda.set_sync_debug_mode(prev_mode)

        return wrapper

else:

    def gpu_sync_allowed(count: int | None = None):
        return _noop_cm()

    def with_gpu_sync_check(fn):
        return fn
