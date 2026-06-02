# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import sys
from contextlib import contextmanager

import torch

import vllm.envs as envs
from vllm.platforms import current_platform

SYNC_ERROR_MESSAGE = (
    "GPU<->CPU sync detected - avoid it or wrap with gpu_sync_allowed()"
)

_GPU_SYNC_ALLOWED_FIRST_SEEN: set[tuple[str, int]] = set()

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
    _install_compile_time_sync_suppressors()


_compile_time_suppressors_installed: bool = False


def _install_compile_time_sync_suppressors() -> None:
    """Wrap torch inductor/aot_autograd compile entry points so the
    synchronizing ops those passes perform (e.g. `constant_fold_uniform_value`
    calling `.item()` on uniform-valued constants) don't trip the
    sync-check mode we set around `execute_model` / `sample_tokens`.

    Warmup-time compiles already run under the gate (before
    `enable_gpu_sync_check`), but post-warmup compiles (runtime
    recompiles from dynamic shape variants, pipeline-parallel fresh
    compile cache, etc.) fire inside `execute_model`. We intentionally
    only want to flag *model-execution* syncs — compile-time work is
    third-party and unavoidable.
    """
    global _compile_time_suppressors_installed
    if _compile_time_suppressors_installed:
        return
    _compile_time_suppressors_installed = True

    try:  # noqa: BLE001
        from torch._inductor.fx_passes import joint_graph as _jg

        _orig_joint = _jg.joint_graph_passes

        @functools.wraps(_orig_joint)
        def _wrapped_joint(*args, **kwargs):
            prev_mode = torch.cuda.get_sync_debug_mode()
            if not prev_mode:
                return _orig_joint(*args, **kwargs)
            torch.cuda.set_sync_debug_mode(0)
            try:
                return _orig_joint(*args, **kwargs)
            finally:
                torch.cuda.set_sync_debug_mode(prev_mode)

        # `compile_fx` does `from .fx_passes.joint_graph import
        # joint_graph_passes`, which binds the *function object* at import
        # time. Patching just the module attribute won't update that rebind,
        # so patch every already-imported reference we can find. Restrict
        # the scan to torch's compile-time modules — iterating all of
        # `sys.modules` triggers `__getattr__` shims on third-party packages
        # (e.g. transformers image_processing modules emit a deprecation
        # warning on every attribute access).
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

    def gpu_sync_allowed(first_only: bool = False):
        """Context manager that suppresses `torch.cuda.set_sync_debug_mode` for the
        duration of the `with` block.

        If `first_only` is True, only the first entry from this call site
        suppresses the sync check; subsequent entries from the same site are
        no-ops so any further GPU syncs will be reported. The "site" is the
        caller's (filename, lineno), so different
        `with gpu_sync_allowed(first_only=True):` lines track independently.
        """
        if torch.compiler.is_compiling():
            return _noop_cm()
        prev_mode = torch.cuda.get_sync_debug_mode()
        if not prev_mode:
            return _noop_cm()
        if first_only:
            frame = sys._getframe(1)
            key = (frame.f_code.co_filename, frame.f_lineno)
            if key in _GPU_SYNC_ALLOWED_FIRST_SEEN:
                return _noop_cm()
            _GPU_SYNC_ALLOWED_FIRST_SEEN.add(key)
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
            except RuntimeError as re:
                if str(re) == "called a synchronizing CUDA operation":
                    raise RuntimeError(SYNC_ERROR_MESSAGE) from re
                raise re
            finally:
                torch.cuda.set_sync_debug_mode(prev_mode)

        return wrapper

else:

    def gpu_sync_allowed(first_only: bool = False):
        return _noop_cm()

    def with_gpu_sync_check(fn):
        return fn
