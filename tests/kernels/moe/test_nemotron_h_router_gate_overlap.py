# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise tests for VLLM_NEMOTRON_H_MOE_ROUTER_OVERLAP.

The router gate GEMM is forked onto a side stream (NemotronHRouterGateOverlap)
while fc1_latent runs on the current stream; the consumer joins on a CUDA
event. Both outputs must be bitwise identical to the sequential order, eagerly
and under CUDA graph capture/replay, across many layers with distinct weights.
"""

import pytest
import torch
from torch import nn

from vllm.model_executor.models.nemotron_h import (
    NemotronHRouterGateOverlap,
    _router_gate_overlaps,
)
from vllm.utils.torch_utils import _encode_layer_name

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

NUM_LAYERS = 40
HIDDEN = 4096
LATENT = 1024
NUM_EXPERTS = 512


class _MMGate(nn.Module):
    """Stand-in for GateLinear's cuBLAS tier: bf16 x bf16 -> fp32."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.mm(x, self.weight.T, out_dtype=torch.float32), None


def _make_layers(max_tokens: int):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    fc1 = [
        torch.randn(LATENT, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02
        for _ in range(NUM_LAYERS)
    ]
    gates = [
        _MMGate(
            torch.randn(NUM_EXPERTS, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02
        )
        for _ in range(NUM_LAYERS)
    ]
    overlaps = [
        NemotronHRouterGateOverlap(g, num_experts=NUM_EXPERTS, max_tokens=max_tokens)
        for g in gates
    ]
    return fc1, gates, overlaps


def _sequential(x, fc1, gates):
    outs = []
    for w, gate in zip(fc1, gates):
        logits, _ = gate(x)
        latent = x @ w.T
        outs.append((latent, logits))
        # Serial dependency between layers (value-preserving).
        x = x + latent[:, :1].to(x.dtype) * 0
    return outs


def _overlapped(x, fc1, overlaps):
    outs = []
    for w, overlap in zip(fc1, overlaps):
        logits = overlap.fork(x)
        latent = x @ w.T
        overlap.wait()  # what MoERunner._forward_impl does first
        outs.append((latent, logits))
        x = x + latent[:, :1].to(x.dtype) * 0
    return outs


def _assert_bitwise(ref, out):
    assert len(ref) == len(out)
    for (lat_r, log_r), (lat_o, log_o) in zip(ref, out):
        assert log_o.dtype == torch.float32
        assert torch.equal(lat_r, lat_o)
        assert torch.equal(log_r, log_o)


def _capture(fn, *args):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(*args)  # warmup
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = fn(*args)
    torch.cuda.current_stream().wait_stream(stream)
    return graph, out


@pytest.mark.parametrize("num_tokens", [1, 6, 8, 48])
def test_router_gate_overlap_eager_bitwise(num_tokens):
    fc1, gates, overlaps = _make_layers(max_tokens=256)
    x = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref = _sequential(x, fc1, gates)
    out = _overlapped(x, fc1, overlaps)
    torch.cuda.synchronize()
    _assert_bitwise(ref, out)


@pytest.mark.parametrize("num_tokens", [1, 6, 8, 48])
def test_router_gate_overlap_cuda_graph_bitwise(num_tokens):
    fc1, gates, overlaps = _make_layers(max_tokens=256)
    x = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    g_ref, ref = _capture(_sequential, x, fc1, gates)
    g_out, out = _capture(_overlapped, x, fc1, overlaps)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        g_ref.replay()
        g_out.replay()
        torch.cuda.synchronize()
        _assert_bitwise(ref, out)


def test_router_gate_overlap_inline_above_max_tokens():
    fc1, gates, overlaps = _make_layers(max_tokens=8)
    x = torch.randn(16, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref = _sequential(x, fc1, gates)
    out = _overlapped(x, fc1, overlaps)
    torch.cuda.synchronize()
    _assert_bitwise(ref, out)
    assert not any(o._pending for o in overlaps)


def test_router_gate_fork_custom_op():
    fc1, gates, overlaps = _make_layers(max_tokens=256)
    name = "test.layers.0.mixer.gate"
    _router_gate_overlaps[name] = overlaps[0]
    try:
        x = torch.randn(4, HIDDEN, device="cuda", dtype=torch.bfloat16)
        logits = torch.ops.vllm.nemotron_h_moe_router_gate_fork(
            x, NUM_EXPERTS, _encode_layer_name(name)
        )
        overlaps[0].wait()
        ref, _ = gates[0](x)
        torch.cuda.synchronize()
        assert logits.shape == (4, NUM_EXPERTS)
        assert torch.equal(logits, ref)
    finally:
        del _router_gate_overlaps[name]
