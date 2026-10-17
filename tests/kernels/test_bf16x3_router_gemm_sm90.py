# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SM90 CuteDSL BF16x3 router GEMM."""

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
    (8192, 64),  # ERNIE-4.5-300B-A47B
    (5120, 256),  # Beam-1.0
    (8192, 256),  # Muse Spark Open
]
# M=1 is left out: there cuBLAS's fp32 GEMV is slightly more accurate for K >= 6144.
NUM_TOKENS = [8, 33, 64, 100, 256, 384, 513, 1024, 1536, 2048, 3072, 8192]
# Match the observed router weight scale
ROUTER_WEIGHT_SCALE = 0.053


def _inputs(num_tokens, hidden_size, num_experts, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, hidden_size, device="cuda").to(torch.bfloat16)
    w = torch.randn(num_experts, hidden_size, device="cuda") * ROUTER_WEIGHT_SCALE
    return x, w


def _mae(out, ref):
    return (out.double() - ref).abs().mean().item()


def _check_against_fp32(x, w, out):
    ref = torch.nn.functional.linear(x.double(), w.double())
    # The fp32 path this kernel replaces on SM90.
    fp32 = torch.nn.functional.linear(x.float(), w)
    assert out.shape == (x.shape[0], w.shape[0])
    assert out.dtype == torch.float32
    assert _mae(out, ref) <= _mae(fp32, ref)


@pytest.mark.parametrize(("hidden_size", "num_experts"), ROUTER_SHAPES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_more_accurate_than_fp32_gemm(num_tokens, hidden_size, num_experts):
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_sm90 import (  # noqa: E501
        bf16x3_router_gemm,
    )

    x, w = _inputs(num_tokens, hidden_size, num_experts)
    _check_against_fp32(x, w, bf16x3_router_gemm(x, w))


@pytest.mark.parametrize(
    ("hidden_size", "num_experts"),
    [
        (1032, 96),  # partial K-tile and partial expert tile
        (256, 64),  # fewer K-tiles than the largest split
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 40, 777])
def test_partial_tiles(num_tokens, hidden_size, num_experts):
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_sm90 import (  # noqa: E501
        bf16x3_router_gemm,
    )

    x, w = _inputs(num_tokens, hidden_size, num_experts)
    out = bf16x3_router_gemm(x, w)
    ref = torch.nn.functional.linear(x.double(), w.double())
    torch.testing.assert_close(out.double(), ref, rtol=0, atol=1e-5)


def test_weight_split_is_exact():
    """One-hot activations select single weights, so the three bf16 terms of
    each weight must sum back to it bit for bit."""
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_sm90 import (  # noqa: E501
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


@pytest.mark.parametrize("num_tokens", [48, 512, 4096])
def test_cuda_graph_replay(num_tokens):
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_sm90 import (  # noqa: E501
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
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_sm90 import (  # noqa: E501
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


def test_warmup_covers_runtime_keys():
    """Warmup compiles from fake tensors every key the runtime can pick."""
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_sm90 import (  # noqa: E501
        Sm90BF16x3RouterGemmKernel,
    )

    kernel = Sm90BF16x3RouterGemmKernel()
    hidden_size, num_experts, max_tokens = 5120, 160, 2048
    kernel.warmup(K=hidden_size, E=num_experts, max_tokens=max_tokens)
    warmed = set(kernel._compiled_cache)
    assert len(warmed) > 1

    for num_tokens in (1, 33, 200, 777, max_tokens):
        x, w = _inputs(num_tokens, hidden_size, num_experts)
        out = kernel(x, w)
        if num_tokens >= 8:
            _check_against_fp32(x, w, out)
    assert set(kernel._compiled_cache) == warmed


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
    from vllm.model_executor.layers.fused_moe.router import bf16x3_router_gemm_sm90

    calls = []
    for name in ("bf16x3_router_gemm", "_bf16x3_router_gemm"):
        real = getattr(bf16x3_router_gemm_sm90, name)

        def record(x, w, real=real):
            calls.append(x.shape[0])
            return real(x, w)

        monkeypatch.setattr(bf16x3_router_gemm_sm90, name, record)
    return calls


@pytest.mark.parametrize(
    ("input_size", "output_size", "num_tokens"),
    [
        # fp32 GEMV shapes: tier 2 dispatches above 24 tokens
        (3072, 256, 25),
        (3072, 256, 1024),
        (5120, 256, 32),
        (5120, 256, 48),
        # generic shape: tier 4
        (5120, 160, 33),
        (5120, 160, 1024),
    ],
)
def test_gate_linear_uses_bf16x3(monkeypatch, input_size, output_size, num_tokens):
    gate = _make_gate_linear(monkeypatch, input_size, output_size)
    assert gate.allow_bf16x3_router_gemm
    gate.weight.data.normal_(0, ROUTER_WEIGHT_SCALE)
    calls = _record_bf16x3_calls(monkeypatch)

    x = torch.randn(num_tokens, input_size, device="cuda").to(torch.bfloat16)
    out, _ = gate(x)

    assert calls == [num_tokens]
    _check_against_fp32(x, gate.weight, out)


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [(3072, 256), (6144, 128), (5120, 256), (4096, 352), (8192, 256)],
)
def test_gate_linear_keeps_fp32_gemv_for_small_batches(
    monkeypatch, input_size, output_size
):
    gate = _make_gate_linear(monkeypatch, input_size, output_size)
    assert gate.allow_fp32_router_gemm
    calls = _record_bf16x3_calls(monkeypatch)

    for num_tokens in (1, 16, 24):
        gate(torch.randn(num_tokens, input_size, device="cuda").to(torch.bfloat16))

    assert calls == []


def test_gate_linear_registers_warmup(monkeypatch, default_vllm_config):
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_sm90 import (  # noqa: E501
        _SM90_BF16X3_ROUTER_GEMM_KERNEL,
    )
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    registry = JitWarmupRegistry(default_vllm_config)
    with registry.activate():
        _make_gate_linear(monkeypatch, 5120, 160)

    max_tokens = default_vllm_config.scheduler_config.max_num_batched_tokens
    assert registry._registrations[_SM90_BF16X3_ROUTER_GEMM_KERNEL] == [
        ((), {"K": 5120, "E": 160, "max_tokens": max_tokens})
    ]


def test_gate_linear_batch_invariant_disables_bf16x3(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    gate = _make_gate_linear(monkeypatch, 3072, 256)

    assert not gate.allow_bf16x3_router_gemm
