# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture custom op + global manager slot.

This module is the layer-side surface of the capture-consumer
framework. Responsibilities:

- Hold the process-global active ``CaptureManager`` reference (set by
  the model runner once per worker).
- Expose ``torch.ops.vllm.capture_residual`` as a custom op registered
  via :func:`vllm.utils.torch_utils.direct_register_custom_op` plus an
  FX-friendly fake impl.
- Expose :func:`maybe_capture_residual`, a ``None``-check gate that
  :func:`vllm.model_executor.layers.steering.apply_layer_steering` calls
  **before** adding the steering vector. Under ``torch.compile`` with
  no active manager, the gate constant-folds and the custom op never
  enters the compiled graph (spec invariant 3).

See ``docs/design/capture_consumers.md`` for the framework design. The
manager implementation lives in ``vllm/v1/capture/manager.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.v1.capture.manager import CaptureManager


# ---------------------------------------------------------------------------
# Hook-name / hook-id encoding
# ---------------------------------------------------------------------------
#
# The custom op takes an ``int hook_id`` rather than a string because
# ``torch.library`` cannot serialize Python strings across the compiled
# boundary. The manager internally stores capture state keyed by
# ``(layer_idx, hook_name)`` so the lookup tables below translate
# between the two representations.

_HOOK_NAME_TO_ID: dict[str, int] = {
    "pre_attn": 0,
    "post_attn": 1,
    "post_mlp": 2,
    "mlp_in": 3,
    "mlp_out": 4,
}
_HOOK_ID_TO_NAME: dict[int, str] = {v: k for k, v in _HOOK_NAME_TO_ID.items()}


# ---------------------------------------------------------------------------
# Module-global active manager
# ---------------------------------------------------------------------------
#
# The model runner calls :func:`set_active_capture_manager` once per
# worker. Tests may install fake/real managers via the same setter.
# ``None`` means the cold path is active and the custom op is
# constant-folded out of compiled graphs.

_ACTIVE_CAPTURE_MANAGER: CaptureManager | None = None


def set_active_capture_manager(mgr: CaptureManager | None) -> None:
    """Install ``mgr`` as the process-global active capture manager.

    Passing ``None`` disables capture entirely and restores the cold
    path. Subsequent calls to :func:`maybe_capture_residual` become
    no-ops that ``torch.compile`` constant-folds away.
    """
    global _ACTIVE_CAPTURE_MANAGER
    _ACTIVE_CAPTURE_MANAGER = mgr


def get_active_capture_manager() -> CaptureManager | None:
    """Return the currently installed capture manager, if any."""
    return _ACTIVE_CAPTURE_MANAGER


# ---------------------------------------------------------------------------
# Hook helper + custom op
# ---------------------------------------------------------------------------


def maybe_capture_residual(
    hidden_states: torch.Tensor,
    layer_idx: int,
    hook_name: str,
) -> None:
    """Cold-path-free gate around the capture custom op.

    Called from :func:`vllm.model_executor.layers.steering.apply_layer_steering`
    **before** the steering op is applied so captures always read the
    pristine residual stream.

    When no manager is installed (cold path / server started without
    any ``--capture-consumers``) this is a pure Python ``None``-check
    plus a dict lookup and returns early. Under ``torch.compile`` the
    early return constant-folds away and the compiled graph contains
    no ``capture_residual`` ops at all — spec invariant 3.
    """
    mgr = _ACTIVE_CAPTURE_MANAGER
    if mgr is None:
        return
    hook_id = _HOOK_NAME_TO_ID[hook_name]
    torch.ops.vllm.capture_residual(hidden_states, layer_idx, hook_id)


def _capture_residual_impl(
    hidden_states: torch.Tensor,
    layer_idx: int,
    hook_id: int,
) -> torch.Tensor:
    """Real impl of ``torch.ops.vllm.capture_residual``.

    Looks up the process-global active manager and forwards to
    :meth:`CaptureManager.on_hook`. When the manager is ``None`` the
    op degenerates to returning ``hidden_states`` unchanged; in
    practice :func:`maybe_capture_residual` gates the call so this
    branch is only reached in compiled graphs that baked in a
    previous manager reference.

    We mark the op as ``mutates_args=["hidden_states"]`` (see
    registration below) so ``torch.compile`` does not dead-code
    eliminate the call when its return value is discarded by the
    caller. The implementation itself never writes to
    ``hidden_states`` — we only need the side-effect annotation to
    preserve the op in the compiled graph.
    """
    mgr = _ACTIVE_CAPTURE_MANAGER
    if mgr is None:
        return hidden_states
    hook_name = _HOOK_ID_TO_NAME[hook_id]
    mgr.on_hook(layer_idx, hook_name, hidden_states)
    return hidden_states


def _capture_residual_fake(
    hidden_states: torch.Tensor,
    layer_idx: int,
    hook_id: int,
) -> torch.Tensor:
    """FX-tracing fake impl for the capture custom op."""
    return torch.empty_like(hidden_states)


# ``mutates_args=["hidden_states"]`` is a deliberate white lie: the
# real impl does not mutate its argument, but the annotation tells
# ``torch.compile`` the op has observable side effects and therefore
# must not be DCE'd even though the return value is discarded by
# ``apply_layer_steering``. Without this annotation some compile
# passes elide the op when its return equals its input and the
# return value is unused.
direct_register_custom_op(
    op_name="capture_residual",
    op_func=_capture_residual_impl,
    fake_impl=_capture_residual_fake,
    mutates_args=["hidden_states"],
)
