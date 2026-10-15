# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``vllm.v1.worker.xpu_model_runner`` (XPU worker / CUDA shims)."""

import pytest
import torch
from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

from vllm.v1.worker.xpu_model_runner import _torch_cuda_wrapper

# XPU-only: needs distinct torch.cuda vs torch.xpu current_stream symbols.
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not hasattr(torch.xpu, "current_stream"),
    reason="torch.xpu.current_stream is required",
)


# Child process: patched torch.cuda must not leak to other tests in the session.
@pytest.mark.forked
def test_torch_cuda_wrapper_allows_dynamo_handler_registration() -> None:
    """Guard against XPU CUDA shim breaking Torch Dynamo during AOT compile.

    Before the fix, ``_torch_cuda_wrapper`` assigned
    ``torch.cuda.current_stream = torch.xpu.current_stream`` (same function object).
    On the first AOT/profile run, Dynamo builds its in-graph handler table and
    registers ``torch.cuda.current_stream`` and ``torch.xpu.current_stream``
    separately; duplicate identity triggers::

        AssertionError: Handler already registered for <function current_stream ...>

    That surfaced as EngineCore failing in ``profile_run`` / ``_get_handlers()``.
    The fix uses distinct shim callables so both can be registered.

    This test replays the post-init state (wrapper applied, patches left on
    ``torch.cuda``) and checks that Dynamo's real ``_get_handlers()`` succeeds.
    """
    # Same entry point as XPUModelRunner.__init__ (patches persist after exit).
    with _torch_cuda_wrapper():
        pass

    # Fresh handler table build, as on first torch.compile / AOT in the worker.
    # Registers torch.cuda.current_stream and torch.xpu.current_stream separately;
    # if they are the same object (pre-fix alias), raises Handler already registered.
    TorchInGraphFunctionVariable._get_handlers.cache_clear()
    TorchInGraphFunctionVariable._get_handlers()


# Child process: patched torch.cuda must not leak to other tests in the session.
@pytest.mark.forked
def test_torch_cuda_wrapper_leaves_event_usable_as_a_type() -> None:
    """Guard against the XPU CUDA shim breaking ``torch.cuda.Event | None``
    annotations evaluated by unrelated code later in the same process.

    ``_torch_cuda_wrapper`` patches ``torch.cuda`` to redirect to
    ``torch.xpu`` and never undoes it (by design -- the redirection must
    stay in effect for the rest of the worker process, not just during
    ``__init__``). Before the fix, ``torch.cuda.Event`` was replaced by a
    plain function, which can't be used in ``SomeType | None`` annotations;
    any later-imported class with a ``torch.cuda.Event | None`` field
    default (e.g. ``vllm.models.deepseek_v41.common.engram.Engram``) raised
    ``TypeError: unsupported operand type(s) for |: 'function' and
    'NoneType'`` merely from being imported in the same test session.
    """
    with _torch_cuda_wrapper():
        pass

    assert torch.cuda.Event | None is not None
