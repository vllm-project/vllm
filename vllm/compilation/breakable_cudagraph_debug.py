# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug tools for breakable CUDA graph (BCG).

BCG records a sequence of CUDA graphs broken apart by eager functions that
can't be CUDA graphed. At each eager break, we copy the tensors returned by
the eager function in a static buffer that the next graph consumes. This
leaves room for developer errors that could be very difficult to debug: if
an eager function allocates a new tensor that wasn't returned explicitly but
survives outside of the eager call (like assigning to an instance variable if
the eager function was a class method), the new tensor won't be copied to the
next CUDA graph, causing a potential silent correctness issue.

``check_non_explicit_outputs`` detects this at capture time and returns a
list of tensors that were not returned explicitly.

We detect it from the CUDA caching allocator's own allocation history. Around
the eager call we turn on ``torch.cuda.memory._record_memory_history`` and read
the raw snapshot's ``device_traces`` (the public ``memory_snapshot()`` returns
only ``segments``). Replaying those alloc/free events tells us exactly which
blocks were allocated during the call and are still alive when it returns.
Subtracting the explicitly-returned (and ``mark_bcg_output``-marked) tensors
leaves the non-explicit outputs.

The tool doesn't help if a tensor is allocated outside the PyTorch caching
allocator (a raw ``cudaMalloc`` in C++) or borrowed via DLPack.

If the user is confident that a non-explicitly returned tensor is safe, they
may mark that tensor with ``mark_bcg_output`` and avoid a false detection.
"""

from __future__ import annotations

import textwrap
import weakref
from collections.abc import Callable
from contextvars import ContextVar

import torch
from torch.utils._pytree import tree_iter

# When debug capture is active this holds a list of `(weakref(storage),
# storage.data_ptr())` tensors that the check should ignore. Holds no
# strong reference, so it never keeps a tensor alive. Storage (not address) based: if
# a marked tensor is freed mid-break and its address is reused by a real escape, the
# dead weakref stops exempting that address, so the real escape is still caught.
_safe_non_explicit_outputs: ContextVar[list | None] = ContextVar(
    "bcg_safe_marks", default=None
)


def mark_bcg_output(*tensors: torch.Tensor) -> None:
    marked_outputs = _safe_non_explicit_outputs.get()
    if marked_outputs is None:
        return
    for t in tensors:
        if torch.is_tensor(t) and t.numel() > 0:
            st = t.untyped_storage()
            marked_outputs.append((weakref.ref(st), st.data_ptr()))


def _get_marked_output_ptrs(marked_outputs: list) -> set:
    return {
        ptr
        for ref, ptr in marked_outputs
        if (st := ref()) is not None and st.data_ptr() == ptr
    }


def _get_tensor_storage_ptrs(output) -> set:
    return {
        t.untyped_storage().data_ptr()
        for t in tree_iter(output)
        if torch.is_tensor(t) and t.numel() > 0
    }


def _get_device_mem_trace(device_idx: int) -> list:
    # The public `torch.cuda.memory_snapshot()` exposes only `segments`
    # `device_traces` is reachable only through the private C++ function
    snapshot = torch._C._cuda_memorySnapshot(None)
    trace = snapshot.get("device_traces", [])
    if trace and isinstance(trace[0], list):
        return trace[device_idx] if device_idx < len(trace) else []
    return trace


def _get_live_tensors(trace: list) -> dict:
    live_tensors: dict = {}
    for event in trace:
        action = event["action"]
        if action == "alloc":
            live_tensors[event["addr"]] = (event["size"], event.get("frames"))
        elif action == "free_requested":
            live_tensors.pop(event["addr"], None)
    return live_tensors


def _format_frames(frames) -> str | None:
    if not frames:
        return None
    return "\n".join(
        f'File "{f.get("filename")}", line {f.get("line")}, in {f.get("name")}'
        for f in frames
    )


def _make_error_msg(fn_name: str, non_explicit_outputs: list) -> str:
    lines = [
        f"[BCG] eager break '{fn_name}' produced "
        f"{len(non_explicit_outputs)} output(s) that survive the call but are "
        "not returned. On replay, these are re-allocated at fresh addresses "
        "with no copy-back, so downstream captured segments read stale memory. "
        "Return them from the break, or write the result in-place into a "
        "pre-existing buffer."
    ]
    for i, e in enumerate(non_explicit_outputs[:5], 1):
        lines.append(f"  #{i}: data_ptr={e['ptr']:#x} nbytes={e['nbytes']}")
        if e["frames"]:
            lines.append(textwrap.indent(e["frames"], "      "))
    if len(non_explicit_outputs) > 5:
        lines.append(f"  ... and {len(non_explicit_outputs) - 5} more")
    return "\n".join(lines)


def check_non_explicit_outputs(inner: Callable, args: tuple, kwargs: dict):
    """Run an eager break under allocator history recording and report leaks.

    Args:
        inner: The eager break callable to run.
        args: Positional args for ``inner``.
        kwargs: Keyword args for ``inner``.

    Returns:
        A ``(output, error_message_or_None)`` tuple, where ``output`` is the
        return value of ``inner`` and ``error_message_or_None`` describes any
        tensors that survive the call but are not explicitly returned.

    Raises:
        RuntimeError: If PyTorch CUDA memory history is already enabled. This
        check uses PyTorch CUDA memory history internally, so users should not
        use it while this check is active.
    """
    if torch._C._cuda_isHistoryEnabled():
        raise RuntimeError(
            "BCG debug uses PyTorch CUDA memory history internally. Disable "
            "PyTorch CUDA memory history before enabling "
            "VLLM_BREAKABLE_CUDAGRAPH_DEBUG."
        )

    device_idx = torch.accelerator.current_device_index()
    torch.cuda.memory._record_memory_history(
        "all",
        context="alloc",
        stacks="python",
        max_entries=1_000_000,
        clear_history=True,
    )
    token = _safe_non_explicit_outputs.set([])
    try:
        output = inner(*args, **kwargs)
    finally:
        marked_outputs = _safe_non_explicit_outputs.get() or []
        _safe_non_explicit_outputs.reset(token)
        trace = _get_device_mem_trace(device_idx)
        torch.cuda.memory._record_memory_history(None)

    explicit_outputs = _get_tensor_storage_ptrs(output) | _get_marked_output_ptrs(
        marked_outputs
    )
    non_explicit_outputs = [
        {"ptr": addr, "nbytes": size, "frames": _format_frames(frames)}
        for addr, (size, frames) in _get_live_tensors(trace).items()
        if not any(addr <= p < addr + size for p in explicit_outputs)
    ]
    name = getattr(inner, "__name__", repr(inner))
    return output, (
        _make_error_msg(name, non_explicit_outputs) if non_explicit_outputs else None
    )
