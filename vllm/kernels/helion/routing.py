# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Call-site routing for eager Helion quant ops (non-compiled path).

Helion custom ops carry a per-call Python dispatch overhead (~30us) that is a
loss in **eager** execution but is captured away under a **CUDA graph** (only the
GPU kernel launches are replayed). So we want Helion *only* when the op is being
captured into a CUDA graph, and the native ``_C`` kernel otherwise.

``route_quant`` is called at the ``torch.ops._C.<op>`` CALL SITES in eager code
(fused-MoE experts, MLA attention, linear methods). Ops dispatched from Python in
these eager regions never enter the torch.compile graph, so a graph-level swap
cannot reach them; this call-site hook gives them the same "Helion under
cudagraph, native in eager" behaviour. The decision is made at RUNTIME via
``torch.cuda.is_current_stream_capturing()`` -- a plain ``if`` at a compile-traced
call site would be baked to False at trace time, so the ``is_compiling()`` guard
short-circuits during tracing (the compiled path is handled by a separate pass).
"""

from __future__ import annotations

import torch

# --- Call-site routing for eager ops (MoE / MLA / linear) --------------------

_ROUTABLE_EAGER_OPS = frozenset(
    {
        "per_token_group_fp8_quant",
    }
)
_helion_ready: dict[str, bool] = {}


def _helion_available(op_name: str) -> bool:
    ready = _helion_ready.get(op_name)
    if ready is None:
        try:
            import vllm.kernels.helion  # noqa: F401  register ops
            from vllm.kernels.helion.register import _HOP_AVAILABLE

            ready = (not _HOP_AVAILABLE) and hasattr(
                torch.ops.vllm_helion, op_name
            )
        except Exception:
            ready = False
        _helion_ready[op_name] = ready
    return ready


def route_quant(op_name: str, *args):
    """Dispatch a quant op to Helion iff currently capturing a CUDA graph.

    Use at the ``torch.ops._C.<op>`` call sites in eager code (MoE/MLA/linear).
    The ``is_compiling()`` guard short-circuits during torch.compile tracing so
    the capture check never graph-breaks (traced call sites just use native; the
    FX swap covers them). Falls back to native for any unsupported/unknown op.
    """
    import vllm.envs as envs

    if (
        op_name in _ROUTABLE_EAGER_OPS
        and envs.VLLM_USE_HELION_KERNELS
        and not torch.compiler.is_compiling()
        and torch.cuda.is_current_stream_capturing()
        and _helion_available(op_name)
    ):
        return getattr(torch.ops.vllm_helion, op_name).default(*args)
    return getattr(torch.ops._C, op_name).default(*args)
