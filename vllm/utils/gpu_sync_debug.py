# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detect unintended GPU<->CPU syncs in the hot path.

`torch.cuda.set_sync_debug_mode` is process-global, so we arm it at "warn"
(which never raises) and decide in `_sync_warning_hook` whether a given sync
is a failure. The scoping lives in ContextVars, which are per-thread and
per-asyncio-task, so an allow region opened on one thread is invisible to
every other one.

torch's mode only instruments explicit synchronizing calls, so it misses
`non_blocking=True` CPU<->CUDA copies that the CUDA driver silently stages
through pageable memory (when the CPU tensor is not pinned or not densely
laid out). Those are caught by the wrappers on `Tensor.to`/`Tensor.cuda`/
`Tensor.copy_` installed by `_install_copy_checkers()`.
"""

import functools
import sys
import threading
import warnings
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import torch

import vllm.envs as envs
from vllm.platforms import current_platform

SYNC_ERROR_MESSAGE = (
    "GPU<->CPU sync detected - avoid it or wrap with gpu_sync_allowed()"
)

# Warning text torch emits in "warn" mode.
_TORCH_SYNC_WARNING = "called a synchronizing CUDA operation"

_GPU_SYNC_ALLOWED_FIRST_SEEN: set[tuple[str, int]] = set()

# Read once: `envs.__getattr__` re-evaluates the env lambda on every access,
# which dominates the cost of a disabled `gpu_sync_allowed()`. Tests toggling
# the env var must patch this attribute instead.
_SYNC_CHECK_MODE: str | None = (
    envs.VLLM_GPU_SYNC_CHECK if current_platform.is_cuda_alike() else None
)

# Configured mode while a thread is inside a checked call, else None.
_checking: ContextVar[str | None] = ContextVar("vllm_gpu_sync_checking", default=None)
# Depth of nested `gpu_sync_allowed()` regions on this thread.
_allow_depth: ContextVar[int] = ContextVar("vllm_gpu_sync_allow_depth", default=0)

# Off during engine setup so model load, KV cache init and warmup may sync
# freely; flipped on by `enable_gpu_sync_check()` once warmup completes.
_sync_check_enabled: bool = False


def enable_gpu_sync_check() -> None:
    """Flip the sync-check gate on, once per worker, after warmup."""
    if _SYNC_CHECK_MODE is None:
        return
    global _sync_check_enabled
    _sync_check_enabled = True
    _install_compile_time_sync_suppressors()
    _install_copy_checkers()


_arm_lock = threading.Lock()
_arm_count: int = 0
_saved_sync_debug_mode: int = 0
_prev_showwarning = warnings.showwarning
_sync_filter_head: tuple | None = None


def _sync_warning_hook(message, category, filename, lineno, file=None, line=None):
    """Turn torch's sync warning into an error, but only on a thread that is
    being checked and is outside any allow region."""
    if _TORCH_SYNC_WARNING in str(message):
        mode = _checking.get()
        if mode is None or _allow_depth.get():
            return None
        if mode == "error":
            raise RuntimeError(SYNC_ERROR_MESSAGE)
    return _prev_showwarning(message, category, filename, lineno, file, line)


def _install_warning_hook() -> None:
    """(Re)install the hook and a filter that lets torch's warning reach it.

    Done per checked call because pytest runs each test inside
    `warnings.catch_warnings()`, which restores both `showwarning` and
    `filters`. The hook is left in place afterwards: outside a checked call
    the debug mode is disarmed, so torch emits nothing for it to see.
    """
    global _prev_showwarning, _sync_filter_head
    if warnings.showwarning is not _sync_warning_hook:
        _prev_showwarning = warnings.showwarning
        warnings.showwarning = _sync_warning_hook
    # "always" so the warning survives filtering and isn't deduplicated by
    # `__warningregistry__`. `filterwarnings` prepends, so only re-assert it
    # once ours is no longer in front.
    if warnings.filters[:1] != [_sync_filter_head]:
        warnings.filterwarnings(
            "always", message=_TORCH_SYNC_WARNING, category=UserWarning
        )
        _sync_filter_head = warnings.filters[0]


@contextmanager
def _checked_region(mode: str):
    """Police syncs on this thread for the duration of the block.

    The debug mode is armed per call so that syncs outside a checked region
    emit nothing, and refcounted because `execute_model` and `sample_tokens`
    nest. "warn" rather than "error" because torch's error mode raises on
    whichever thread synced, with no way to exempt one.
    """
    global _arm_count, _saved_sync_debug_mode
    _install_warning_hook()
    with _arm_lock:
        if _arm_count == 0:
            _saved_sync_debug_mode = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("warn")
        _arm_count += 1
    token = _checking.set(mode)
    try:
        yield
    finally:
        _checking.reset(token)
        with _arm_lock:
            _arm_count -= 1
            if _arm_count == 0:
                torch.cuda.set_sync_debug_mode(_saved_sync_debug_mode)


@contextmanager
def _allow_syncs():
    token = _allow_depth.set(_allow_depth.get() + 1)
    try:
        yield
    finally:
        _allow_depth.reset(token)


# Shared, since `nullcontext` is stateless: reusing one instance avoids an
# allocation per call, which is most of the cost when the check is disabled.
_NOOP_CM = nullcontext()

_compile_time_suppressors_installed: bool = False


def _suppressing(fn):
    """Allow the syncs `fn` performs on its calling thread."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Not `gpu_sync_allowed()`, which no-ops while
        # `torch.compiler.is_compiling()` -- exactly when these run.
        with _allow_syncs():
            return fn(*args, **kwargs)

    return wrapper


def _install_compile_time_sync_suppressors() -> None:
    """Allow the syncs torch's compile passes perform.

    Warmup-time compiles run before the gate flips, but lazy ones fire inside
    `execute_model`.
    """
    global _compile_time_suppressors_installed
    if _compile_time_suppressors_installed:
        return
    _compile_time_suppressors_installed = True

    try:
        from torch._inductor.fx_passes import joint_graph as _jg

        orig = _jg.joint_graph_passes
        wrapped = _suppressing(orig)
        # `compile_fx` imports this by value, so patching the defining module
        # alone misses that rebind; patch every compile-time module still
        # holding the original.
        _jg.joint_graph_passes = wrapped
        for name, mod in list(sys.modules.items()):
            if (
                mod is not None
                and name.startswith(
                    ("torch._inductor", "torch._functorch", "torch._dynamo")
                )
                and getattr(mod, "joint_graph_passes", None) is orig
            ):
                setattr(mod, "joint_graph_passes", wrapped)  # noqa: B010
    except Exception:  # pragma: no cover
        pass

    try:
        # Inductor builds its cudagraph tree lazily, so `deferred_cudagraphify`
        # and the `capture_begin` sync inside it can fire during
        # `execute_model`. It resolves `cudagraphify` as a module global at
        # call time, so patching the attribute is enough.
        from torch._inductor import cudagraph_trees as _ct

        _ct.cudagraphify = _suppressing(_ct.cudagraphify)
    except Exception:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Implicit sync detection in `non_blocking` CPU<->CUDA copies.
#
# A `non_blocking=True` copy between CPU and CUDA still blocks the host when
# the CPU side is not pinned or not densely laid out (gapped strided slices
# stage through a pageable temp): D2H copies into such memory wait for the
# stream, and H2D copies from it do too beyond a driver size threshold.
# None of that goes through an API torch's sync debug mode instruments, so
# wrap the Python-level transfer entry points and flag such copies on threads
# in a checked region.

IMPLICIT_SYNC_ERROR_MESSAGE = (
    "Implicit GPU<->CPU sync detected - use a pinned, densely laid out CPU "
    "tensor, or wrap with gpu_sync_allowed()"
)

# Captured at import; `_install_copy_checkers` replaces these attributes.
_torch_to = torch.Tensor.to
_torch_cuda = torch.Tensor.cuda
_torch_copy_ = torch.Tensor.copy_

# torch's own parser for `Tensor.to`'s overloaded signature, returning
# (device, dtype, non_blocking, memory_format) (also used by torch._dynamo).
_parse_to = getattr(torch._C._nn, "_parse_to", None)


def _active_check_mode() -> str | None:
    """The mode the calling thread is being checked in, if any."""
    if not _sync_check_enabled or _allow_depth.get() or torch.compiler.is_compiling():
        return None
    return _checking.get()


def _is_dense(t: torch.Tensor) -> bool:
    """Whether `t`'s elements fill a contiguous storage range, possibly
    permuted (e.g. a transposed matrix). Such layouts can be copied with
    pitched cudaMemcpy2D/3DAsync directly from/to pinned memory, whereas
    gapped layouts (e.g. strided slices) stage through a pageable temp."""
    if t.is_contiguous():
        # The common case, answered by a memoized TensorImpl flag.
        return True
    expected_stride = 1
    for stride, size in sorted(zip(t.stride(), t.shape)):
        if size <= 1:
            continue
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True


def _cpu_copy_stall_reason(
    cpu_tensor: torch.Tensor, device: torch.device | None
) -> str | None:
    """Why a `non_blocking` copy between `cpu_tensor` and a CUDA tensor would
    silently block the host, or None if it is genuinely asynchronous.

    `device` describes the other side of the copy; a non-CUDA `device` means
    the copy does not involve the GPU at all.
    """
    if device is None or device.type != "cuda" or cpu_tensor.device.type != "cpu":
        return None
    if cpu_tensor.numel() == 0:
        return None  # Nothing to copy; torch issues no CUDA call at all.
    if not _is_dense(cpu_tensor):
        return "the CPU tensor is not densely laid out"
    if not cpu_tensor.is_pinned():
        return "the CPU tensor is not pinned"
    return None


def _report_implicit_sync(op: str, direction: str, reason: str, mode: str) -> None:
    msg = (
        f"Tensor.{op}: a non_blocking {direction} copy that may be silently "
        f"synchronous ({reason}). {IMPLICIT_SYNC_ERROR_MESSAGE}"
    )
    if mode == "error":
        raise RuntimeError(msg)
    warnings.warn(msg, stacklevel=3)


def _checked_to(self, *args, **kwargs):
    # Only H2D is checked: D2H `to` allocates a pinned destination in
    # supported torch versions, so it is genuinely asynchronous.
    if (mode := _active_check_mode()) is not None and _parse_to is not None:
        try:
            device, _, non_blocking, _ = _parse_to(*args, **kwargs)
        except Exception:
            pass  # Invalid args; let the original call raise.
        else:
            if non_blocking and (reason := _cpu_copy_stall_reason(self, device)):
                _report_implicit_sync("to", "H2D", reason, mode)
    return _torch_to(self, *args, **kwargs)


def _checked_cuda(self, *args, **kwargs):
    # Signature: cuda(device=None, non_blocking=False, memory_format=None).
    if (mode := _active_check_mode()) is not None:
        device = args[0] if args else kwargs.get("device")
        non_blocking = args[1] if len(args) > 1 else kwargs.get("non_blocking", False)
        if non_blocking:
            if device is None:
                device = torch.device("cuda")
            elif isinstance(device, int):
                device = torch.device("cuda", device)
            elif not isinstance(device, torch.device):
                device = torch.device(device)
            if reason := _cpu_copy_stall_reason(self, device):
                _report_implicit_sync("cuda", "H2D", reason, mode)
    return _torch_cuda(self, *args, **kwargs)


def _checked_copy_(self, *args, **kwargs):
    # Signature: copy_(src, non_blocking=False).
    if (
        (mode := _active_check_mode()) is not None
        and args
        and isinstance(src := args[0], torch.Tensor)
    ):
        non_blocking = args[1] if len(args) > 1 else kwargs.get("non_blocking", False)
        if non_blocking:
            if self.device.type == "cuda":
                reason = _cpu_copy_stall_reason(src, self.device)
                direction = "H2D"
            else:
                reason = _cpu_copy_stall_reason(self, src.device)
                direction = "D2H"
            if reason:
                _report_implicit_sync("copy_", direction, reason, mode)
    return _torch_copy_(self, *args, **kwargs)


_copy_checkers_installed: bool = False


def _install_copy_checkers() -> None:
    """Wrap the Python-level CPU<->CUDA transfer entry points with the checks
    above.

    The patch is process-global, but it only reports on threads inside a
    checked region, matching the scoping of the torch-level check. Copies
    issued from C++ (e.g. inside compiled graphs) bypass it.
    """
    global _copy_checkers_installed
    if _copy_checkers_installed:
        return
    _copy_checkers_installed = True
    torch.Tensor.to = _checked_to  # type: ignore[method-assign]
    torch.Tensor.cuda = _checked_cuda  # type: ignore[method-assign]
    torch.Tensor.copy_ = _checked_copy_  # type: ignore[method-assign]


def gpu_sync_allowed(first_only: bool = False):
    """Allow GPU<->CPU syncs inside the `with` block, on this thread only.

    With `first_only`, only the first entry from a given call site
    (filename, lineno) is allowed, so later syncs there are still reported.
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
    """Report GPU<->CPU syncs performed by `fn` on its calling thread.

    Active only once `enable_gpu_sync_check()` has flipped the gate. Other
    threads are never policed, so deliberate syncs there (e.g. the EPLB
    transfer worker) are unaffected.
    """
    if (mode := _SYNC_CHECK_MODE) is None:
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _sync_check_enabled:
            return fn(*args, **kwargs)
        with _checked_region(mode):
            return fn(*args, **kwargs)

    return wrapper
