# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Piecewise CUDA graph pieces are replayed reading their inputs from the
addresses recorded at capture time; the wrapper does not copy inputs. If a
producer feeding a piece (e.g. a custom op added to ``splitting_ops``) returns
freshly-allocated outputs, those land at new addresses every step and replay
reads stale memory -- silent output corruption. These tests pin that the
address-stability check runs on the first replay (not only under DEBUG) so the
failure is loud, and that a stable buffer still replays correctly.
"""

import pytest
import torch
import torch.nn as nn

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA graphs require a CUDA-like device",
)

D = 8


class _Piece(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _make_wrapper() -> tuple[CUDAGraphWrapper, VllmConfig]:
    cc = CompilationConfig(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    vllm_config = VllmConfig(compilation_config=cc)
    piece = _Piece().to(current_platform.device_type).eval()
    with set_current_vllm_config(vllm_config):
        set_cudagraph_capturing_enabled(True)
        wrapper = CUDAGraphWrapper(
            piece, vllm_config, runtime_mode=CUDAGraphMode.PIECEWISE
        )
    return wrapper, vllm_config


def _run(wrapper, vllm_config, x, mode):
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(
            None,
            vllm_config,
            cudagraph_runtime_mode=mode,
            batch_descriptor=BatchDescriptor(num_tokens=1),
        ),
    ):
        return wrapper(x)


@torch.no_grad()
def test_stable_input_address_replays_ok():
    """Reusing the capture-time buffer (stable address) replays without error
    and reproduces the eager result -- the common, correct case must not
    regress."""
    dev = current_platform.device_type
    wrapper, vllm_config = _make_wrapper()
    buf = torch.zeros(1, D, device=dev)

    # Warm up eagerly so cuBLAS workspaces exist before capture.
    _run(wrapper, vllm_config, buf, CUDAGraphMode.NONE)
    # Capture (records buf's address).
    _run(wrapper, vllm_config, buf, CUDAGraphMode.PIECEWISE)

    # Replay with new values written *into the same buffer* (address stable).
    buf.copy_(torch.randn(1, D, device=dev))
    out = _run(wrapper, vllm_config, buf, CUDAGraphMode.PIECEWISE).clone()
    ref = wrapper.unwrap()(buf)
    assert torch.equal(out, ref)


@torch.no_grad()
def test_unstable_input_address_raises_on_replay():
    """A fresh input tensor (new address) on replay must fail loudly instead of
    silently reading stale memory -- the ``splitting_ops`` fresh-output
    footgun."""
    dev = current_platform.device_type
    wrapper, vllm_config = _make_wrapper()
    captured = torch.zeros(1, D, device=dev)

    _run(wrapper, vllm_config, captured, CUDAGraphMode.NONE)
    _run(wrapper, vllm_config, captured, CUDAGraphMode.PIECEWISE)  # capture

    # First replay with a *different* tensor. `captured` is kept alive so the
    # allocator hands `fresh` a different address, mimicking a producer that
    # returns a freshly-allocated output each step.
    fresh = torch.randn(1, D, device=dev)
    assert fresh.data_ptr() != captured.data_ptr()
    with pytest.raises(RuntimeError, match="different input addresses"):
        _run(wrapper, vllm_config, fresh, CUDAGraphMode.PIECEWISE)
