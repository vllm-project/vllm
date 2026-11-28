# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred aux-stream launch of the shared experts during CUDA graph capture.

SharedExperts.maybe_forward_async(defer_launch=True) records the input-ready
event immediately but, in graph capture on ROCm, enqueues the shared-experts
work only when launch_deferred() / wait() is called. Results must be identical
to the eager and the immediate-launch paths, and the work must still be
ordered after the event (i.e. after the code preceding maybe_forward_async)."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner import shared_experts as se_mod
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs CUDA/ROCm streams"
)


def _make(layer, defer: bool) -> SharedExperts:
    se = object.__new__(SharedExperts)
    torch.nn.Module.__init__(se)
    se.enable_dbo = False
    se._output = [None, None]
    se._layer = layer
    se._stream = torch.cuda.Stream()
    se._input_ready_event = [torch.cuda.Event(), torch.cuda.Event()]
    se._output_ready_event = [torch.cuda.Event(), torch.cuda.Event()]
    se._defer_capture_launch = defer
    se._deferred_input = [None, None]
    se._determine_shared_experts_order = lambda x: (  # type: ignore[method-assign]
        SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
    )
    return se


def _step(se: SharedExperts, x, w_routed, defer_launch, explicit_launch):
    """router-like producer -> fork shared -> routed chain -> join."""
    h = x * 2.0  # producer on the main stream: shared must wait for it
    overlapping = se.maybe_forward_async(h, defer_launch=defer_launch)
    assert overlapping
    routed = (h @ w_routed).relu() @ w_routed.T
    if explicit_launch:
        se.launch_deferred()
        se.launch_deferred()  # idempotent
    se.wait()
    out = se.output
    se._output[0] = None
    return out + routed


@pytest.mark.parametrize("tokens", [8, se_mod._LATE_FORK_MAX_TOKENS, 65, 128])
@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("explicit_launch", [False, True])
def test_deferred_launch_matches_eager_and_graph(
    tokens, defer, explicit_launch, monkeypatch
):
    torch.manual_seed(0)
    d = 256
    w_shared = torch.randn(d, d, device="cuda", dtype=torch.float32) / 16
    w_routed = torch.randn(d, d, device="cuda", dtype=torch.float32) / 16
    layer = lambda t: (t @ w_shared).relu()  # noqa: E731
    x = torch.randn(tokens, d, device="cuda", dtype=torch.float32)

    ref_se = _make(layer, defer=False)
    ref = _step(ref_se, x, w_routed, False, False)
    torch.cuda.synchronize()

    se = _make(layer, defer=defer)
    monkeypatch.setattr(se_mod, "_is_capturing", lambda: True)  # force the path
    eager = _step(se, x, w_routed, True, explicit_launch)  # deferral path, eager run
    torch.cuda.synchronize()
    torch.testing.assert_close(eager, ref, rtol=0, atol=0)
    assert se._deferred_input == [None, None]
    monkeypatch.undo()

    # real capture
    static_x = x.clone()
    warm = torch.cuda.Stream()
    with torch.cuda.stream(warm):
        _step(se, static_x, w_routed, True, explicit_launch)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _step(se, static_x, w_routed, True, explicit_launch)
    assert se._deferred_input == [None, None]
    for scale in (1.0, -0.5, 3.0):
        static_x.copy_(x * scale)
        g.replay()
        torch.cuda.synchronize()
        want = _step(ref_se, x * scale, w_routed, False, False)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-5)


def test_defer_is_noop_outside_capture():
    layer = lambda t: t + 1  # noqa: E731
    se = _make(layer, defer=True)
    x = torch.zeros(4, 8, device="cuda")
    assert se.maybe_forward_async(x, defer_launch=True)
    assert se._deferred_input == [None, None]  # launched immediately in eager
    se.wait()
    assert torch.equal(se.output, x + 1)


@pytest.mark.parametrize(
    "tokens,deferred",
    [
        (1, True),
        (8, True),
        (se_mod._LATE_FORK_MAX_TOKENS, True),
        (65, False),
        (128, False),
    ],
)
def test_deferral_is_gated_on_token_count(tokens, deferred, monkeypatch):
    se = _make(lambda t: t + 1, defer=True)
    monkeypatch.setattr(se_mod, "_is_capturing", lambda: True)
    x = torch.zeros(tokens, 8, device="cuda")
    assert se.maybe_forward_async(x, defer_launch=True)
    assert (se._deferred_input[0] is not None) == deferred
    se.wait()
    assert se._deferred_input == [None, None]
    assert torch.equal(se.output, x + 1)
