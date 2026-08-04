# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import sys
import warnings
from contextlib import contextmanager
from contextvars import ContextVar

import torch

import vllm.envs as envs
from vllm.platforms import current_platform

SYNC_ERROR_MESSAGE = (
    "GPU<->CPU sync detected - avoid it or wrap with gpu_sync_allowed()"
)

# The warning text torch emits from `set_sync_debug_mode("warn")`.
_TORCH_SYNC_WARNING = "called a synchronizing CUDA operation"

_GPU_SYNC_ALLOWED_FIRST_SEEN: set[tuple[str, int]] = set()

# `torch.cuda.set_sync_debug_mode` is process-global, not thread-local: a
# background thread sees the mode the main thread armed, and a
# `set_sync_debug_mode(0)` on any thread clears it for every thread. So we
# never arm it at "error". Instead we leave it at "warn" for the whole
# process and decide whether a given sync is a failure in
# `_sync_warning_hook`, using the two ContextVars below. ContextVars are
# per-thread (each thread starts with a fresh context) and per-asyncio-task,
# which is exactly the scoping we want: an allow-region opened on one thread
# is invisible to every other thread.
#
# `_checking` holds the configured mode ("warn"/"error") while a thread is
# inside a `with_gpu_sync_check` function, and None elsewhere;
# `_allow_depth` counts nested `gpu_sync_allowed()` regions.
_checking: ContextVar[str | None] = ContextVar("vllm_gpu_sync_checking", default=None)
_allow_depth: ContextVar[int] = ContextVar("vllm_gpu_sync_allow_depth", default=0)

# Read once at import rather than per call: `envs.__getattr__` re-evaluates
# the env-var lambda on every access, which dominates the cost of an
# otherwise-disabled `gpu_sync_allowed()`. Tests that toggle the env var
# must patch this too (see `tests/utils_/test_gpu_sync_debug.py`).
_SYNC_CHECK_MODE: str | None = envs.VLLM_GPU_SYNC_CHECK

# Global sync-check gate. Off during engine setup (model load, KV cache
# init, warmup/compile) so first-compile and lazy-init syncs pass through;
# flipped on by `enable_gpu_sync_check()` at the end of
# `GPUWorker.compile_or_warm_up_model`, after which `with_gpu_sync_check`-
# decorated functions start policing their calling thread.
_sync_check_enabled: bool = False


def enable_gpu_sync_check() -> None:
    """Flip the sync-check gate on. Call once per worker, after warmup /
    first-compile is complete. No-op unless `VLLM_GPU_SYNC_CHECK` is set."""
    if _SYNC_CHECK_MODE is None:
        return
    global _sync_check_enabled
    _sync_check_enabled = True
    _arm_torch_warn_mode()
    _install_compile_time_sync_suppressors()


_torch_warn_mode_armed: bool = False


def _arm_torch_warn_mode() -> None:
    """Put torch in "warn" mode, once per process.

    "warn" rather than "error": torch's error mode raises on whichever
    thread synced, with no way to exempt one. Warnings are reported on the
    syncing thread too, but as a Python warning we can inspect and drop in
    `_sync_warning_hook`.
    """
    global _torch_warn_mode_armed
    if _torch_warn_mode_armed:
        return
    _torch_warn_mode_armed = True
    torch.cuda.set_sync_debug_mode("warn")


_prev_showwarning = warnings.showwarning
_sync_filter_head: tuple | None = None


def _sync_warning_hook(message, category, filename, lineno, file=None, line=None):
    """`warnings.showwarning` replacement that turns a torch sync warning into
    an error, but only for a thread that is inside `with_gpu_sync_check` and
    not inside `gpu_sync_allowed`."""
    if _TORCH_SYNC_WARNING in str(message):
        mode = _checking.get()
        if mode is None or _allow_depth.get():
            # Allowed region, or a thread we are not policing: drop it.
            return None
        if mode == "error":
            raise RuntimeError(SYNC_ERROR_MESSAGE)
        # "warn": report it, but keep going.
    return _prev_showwarning(message, category, filename, lineno, file, line)


def _arm_warning_hook():
    """Install the hook and a permissive filter for torch's sync warning.

    Re-asserted per checked call rather than installed once at startup
    because pytest runs each test inside `warnings.catch_warnings()`, which
    saves and restores both `warnings.showwarning` and `warnings.filters` --
    a hook installed at startup would be swapped out for the duration of
    every test, which is exactly when we need it.

    The hook is deliberately left in place afterwards rather than restored:
    torch stays in "warn" mode process-wide, so without it every sync
    outside a checked region would print a `UserWarning`. Non-sync warnings
    are delegated to whatever handler we displaced.
    """
    global _prev_showwarning, _sync_filter_head
    _arm_torch_warn_mode()
    prev = warnings.showwarning
    if prev is not _sync_warning_hook:
        _prev_showwarning = prev
        warnings.showwarning = _sync_warning_hook
    # The warning must survive filtering to reach `showwarning` at all, and
    # must not be deduplicated by `__warningregistry__`, hence "always".
    # `filterwarnings` prepends, so re-assert only when ours is not in front.
    if _sync_filter_head is None or warnings.filters[:1] != [_sync_filter_head]:
        warnings.filterwarnings(
            "always", message=_TORCH_SYNC_WARNING, category=UserWarning
        )
        _sync_filter_head = warnings.filters[0]


_compile_time_suppressors_installed: bool = False


def _suppressing(fn):
    """Wrap `fn` so syncs it performs are allowed on the calling thread."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # `_allow_syncs()` rather than `gpu_sync_allowed()`: everything
        # wrapped here runs *during* compilation, and `gpu_sync_allowed()`
        # short-circuits to a no-op when `torch.compiler.is_compiling()`,
        # so it would never open the allow region these entry points need.
        with _allow_syncs():
            return fn(*args, **kwargs)

    return wrapper


def _install_compile_time_sync_suppressors() -> None:
    """Wrap torch inductor/aot_autograd compile entry points so the
    synchronizing ops those passes perform don't trip the
    sync-check mode we set around `execute_model` / `sample_tokens`.

    Warmup-time compiles already run under the gate (before
    `enable_gpu_sync_check`), but post-warmup compiles fire inside
    `execute_model` and we want to avoid this tripping the sync check.
    """
    global _compile_time_suppressors_installed
    if _compile_time_suppressors_installed:
        return
    _compile_time_suppressors_installed = True

    try:  # noqa: BLE001
        from torch._inductor.fx_passes import joint_graph as _jg

        _orig_joint = _jg.joint_graph_passes
        _wrapped_joint = _suppressing(_orig_joint)

        # `compile_fx` does `from .fx_passes.joint_graph import
        # joint_graph_passes`, which binds the *function object* at import
        # time. Patching just the module attribute won't update that rebind,
        # so patch every already-imported reference we can find. Restrict
        # the scan to torch's compile-time modules.
        import sys as _sys

        setattr(_jg, "joint_graph_passes", _wrapped_joint)  # noqa: B010
        for _name, _mod in list(_sys.modules.items()):
            if _mod is None:
                continue
            if not (
                _name.startswith("torch._inductor")
                or _name.startswith("torch._functorch")
                or _name.startswith("torch._dynamo")
            ):
                continue
            if getattr(_mod, "joint_graph_passes", None) is _orig_joint:
                setattr(_mod, "joint_graph_passes", _wrapped_joint)  # noqa: B010
    except Exception:  # pragma: no cover
        pass

    try:  # noqa: BLE001
        # Inductor sets up its cudagraph tree lazily, so the first
        # post-warmup call of a partitioned graph runs
        # `deferred_cudagraphify` inside `execute_model`, and the
        # `cudaStreamSynchronize` in `CUDAGraph.capture_begin` trips the
        # check. `deferred_cudagraphify` resolves `cudagraphify` as a module
        # global at call time, so patching the attribute is enough.
        from torch._inductor import cudagraph_trees as _ct

        _ct.cudagraphify = _suppressing(_ct.cudagraphify)
    except Exception:  # pragma: no cover
        pass


@contextmanager
def _allow_syncs():
    token = _allow_depth.set(_allow_depth.get() + 1)
    try:
        yield
    finally:
        _allow_depth.reset(token)


class _NoopCM:
    """Stateless no-op context manager.

    A single shared instance is returned from the disabled path instead of
    building a fresh `@contextmanager` generator per call, which dominates
    the cost when the check is off. Being stateless it is safe to reuse,
    nest, and enter from several threads at once.
    """

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info) -> None:
        return None


_NOOP_CM = _NoopCM()


if current_platform.is_cuda_alike():

    def gpu_sync_allowed(first_only: bool = False):
        """Context manager that allows GPU<->CPU syncs for the duration of the
        `with` block, on the calling thread only.

        Scoping is via a `ContextVar`, so this never mutates the
        process-global `torch.cuda` sync debug mode and so cannot disarm the
        check on another thread.

        If `first_only` is True, only the first entry from this call site
        allows syncs; subsequent entries from the same site are no-ops so any
        further GPU syncs will be reported. The "site" is the caller's
        (filename, lineno), so different
        `with gpu_sync_allowed(first_only=True):` lines track independently.
        """
        if _SYNC_CHECK_MODE is None or torch.compiler.is_compiling():
            return _NOOP_CM
        if first_only:
            frame = sys._getframe(1)
            key = (frame.f_code.co_filename, frame.f_lineno)
            if key in _GPU_SYNC_ALLOWED_FIRST_SEEN:
                return _NOOP_CM
            _GPU_SYNC_ALLOWED_FIRST_SEEN.add(key)
        return _allow_syncs()

    def with_gpu_sync_check(fn):
        """Decorator that policies GPU<->CPU syncs performed by `fn` on the
        calling thread, when `VLLM_GPU_SYNC_CHECK` is set *and* the gate has
        been flipped by `enable_gpu_sync_check()`. Before the gate flips
        (i.e. during engine setup / warmup) the decorated function runs
        as-is.

        Only the thread running `fn` is policed; syncs on other threads
        (e.g. the EPLB async transfer worker) are ignored rather than
        raising there.
        """
        if _SYNC_CHECK_MODE is None:
            return fn

        mode = _SYNC_CHECK_MODE

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _sync_check_enabled:
                return fn(*args, **kwargs)
            _arm_warning_hook()
            token = _checking.set(mode)
            try:
                return fn(*args, **kwargs)
            finally:
                _checking.reset(token)

        return wrapper

else:
    # No-op the methods in non-CUDA cases.

    def gpu_sync_allowed(first_only: bool = False):
        return _NOOP_CM

    def with_gpu_sync_check(fn):
        return fn
