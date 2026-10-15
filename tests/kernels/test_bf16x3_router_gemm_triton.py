# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SM90 Triton BF16x3 router GEMM."""

import pytest
import torch

import vllm.envs as envs
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((9, 0)),
    reason="SM90 BF16x3 router GEMM requires a Hopper GPU",
)

# (hidden_size, num_experts) of models that keep fp32 router weights.
ROUTER_SHAPES = [
    (3072, 256),  # MiniMax-M2/M2.5
    (6144, 128),  # MiniMax-M3
    (5120, 160),  # GLM-4.5/4.6/4.7
    (4096, 192),  # Hunyuan-V3
    (2816, 256),  # Hunyuan-V4
    (4096, 352),  # Step-5
]
# M=1 is left out: there cuBLAS's fp32 GEMV is slightly more accurate for K >= 6144.
NUM_TOKENS = [8, 33, 64, 100, 256, 513, 1024, 2048, 8192]
# Match the observed router weight scale
ROUTER_WEIGHT_SCALE = 0.053


def _inputs(num_tokens, hidden_size, num_experts, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, hidden_size, device="cuda").to(torch.bfloat16)
    w = torch.randn(num_experts, hidden_size, device="cuda") * ROUTER_WEIGHT_SCALE
    return x, w


def _mae(out, ref):
    return (out.double() - ref).abs().mean().item()


@pytest.mark.parametrize(("hidden_size", "num_experts"), ROUTER_SHAPES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_more_accurate_than_fp32_gemm(num_tokens, hidden_size, num_experts):
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_triton import (  # noqa: E501
        bf16x3_router_gemm,
    )

    x, w = _inputs(num_tokens, hidden_size, num_experts)
    out = bf16x3_router_gemm(x, w)
    ref = torch.nn.functional.linear(x.double(), w.double())
    # The fp32 path this kernel replaces on SM90.
    fp32 = torch.nn.functional.linear(x.float(), w)

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == torch.float32
    assert _mae(out, ref) <= _mae(fp32, ref)


def test_weight_split_is_exact():
    """One-hot activations select single weights, so the three bf16 terms of
    each weight must sum back to it bit for bit."""
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_triton import (  # noqa: E501
        bf16x3_router_gemm,
    )

    num_tokens, hidden_size, num_experts = 300, 4096, 160
    torch.manual_seed(0)
    mantissas = 1 + torch.rand(num_experts, hidden_size, device="cuda")
    exponents = torch.randint(-40, 8, (num_experts, hidden_size), device="cuda")
    signs = torch.randint(0, 2, (num_experts, hidden_size), device="cuda") * 2 - 1
    w = (signs * mantissas * torch.pow(2.0, exponents)).float()
    cols = torch.randperm(hidden_size, device="cuda")[:num_tokens]
    x = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    x[torch.arange(num_tokens, device="cuda"), cols] = 1.0

    out = bf16x3_router_gemm(x, w)
    assert torch.equal(out, w[:, cols].T)


@pytest.mark.parametrize("num_tokens", [48, 512])
def test_cuda_graph_replay(num_tokens):
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_triton import (  # noqa: E501
        bf16x3_router_gemm,
    )

    x, w = _inputs(num_tokens, 3072, 256)
    expected = bf16x3_router_gemm(x, w)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = bf16x3_router_gemm(x, w)

    for seed in range(3):
        new_x, new_w = _inputs(num_tokens, 3072, 256, seed=seed + 1)
        x.copy_(new_x)
        w.copy_(new_w)
        graph.replay()
        torch.testing.assert_close(out, bf16x3_router_gemm(x, w), rtol=0, atol=0)
    assert not torch.equal(out, expected)


def test_strided_activations():
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_triton import (  # noqa: E501
        bf16x3_router_gemm,
    )

    x, w = _inputs(96, 3072, 256)
    padded = torch.zeros(96, 3072 + 128, dtype=torch.bfloat16, device="cuda")
    padded[:, :3072] = x
    torch.testing.assert_close(
        bf16x3_router_gemm(padded[:, :3072], w),
        bf16x3_router_gemm(x, w),
        rtol=0,
        atol=0,
    )


# =================================================================
# GateLinear dispatch
# =================================================================


def _make_gate_linear(monkeypatch, input_size, output_size):
    for module in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(f"{module}.get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(f"{module}.get_tensor_model_parallel_world_size", lambda: 1)

    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    return GateLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        out_dtype=torch.float32,
        params_dtype=torch.float32,
    ).cuda()


def _record_bf16x3_calls(monkeypatch):
    from vllm.model_executor.layers.fused_moe.router import (
        bf16x3_router_gemm_triton,
    )

    calls = []
    for name in ("bf16x3_router_gemm", "_bf16x3_router_gemm"):
        real = getattr(bf16x3_router_gemm_triton, name)

        def record(x, w, real=real):
            calls.append(x.shape[0])
            return real(x, w)

        monkeypatch.setattr(bf16x3_router_gemm_triton, name, record)
    return calls


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [
        (3072, 256),  # fp32 GEMV shape: tier 2 dispatches above 32 tokens
        (5120, 160),  # generic shape: tier 4
    ],
)
@pytest.mark.parametrize("num_tokens", [33, 1024])
def test_gate_linear_uses_bf16x3_above_32_tokens(
    monkeypatch, input_size, output_size, num_tokens
):
    gate = _make_gate_linear(monkeypatch, input_size, output_size)
    assert gate.allow_bf16x3_router_gemm
    gate.weight.data.normal_(0, ROUTER_WEIGHT_SCALE)
    calls = _record_bf16x3_calls(monkeypatch)

    x = torch.randn(num_tokens, input_size, device="cuda").to(torch.bfloat16)
    out, _ = gate(x)

    assert calls == [num_tokens]
    ref = torch.nn.functional.linear(x.double(), gate.weight.double())
    fp32 = torch.nn.functional.linear(x.float(), gate.weight)
    assert _mae(out, ref) <= _mae(fp32, ref)


def test_gate_linear_keeps_fp32_gemv_for_small_batches(monkeypatch):
    gate = _make_gate_linear(monkeypatch, 3072, 256)
    assert gate.allow_fp32_router_gemm
    calls = _record_bf16x3_calls(monkeypatch)

    gate(torch.randn(32, 3072, device="cuda").to(torch.bfloat16))

    assert calls == []


def test_gate_linear_batch_invariant_disables_bf16x3(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    gate = _make_gate_linear(monkeypatch, 3072, 256)

    assert not gate.allow_bf16x3_router_gemm
