# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CuteDSL BF16x3 router GEMMs (SM90 and SM100)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

# Match the observed router weight scale
ROUTER_WEIGHT_SCALE = 0.053


def _requires_sm100_cutedsl():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("bf16x3 router GEMM requires SM100-class GPU")
    if not has_cutedsl():
        pytest.skip("cutedsl (cutlass) not installed")


def _requires_sm90_cutedsl():
    if not current_platform.is_device_capability((9, 0)):
        pytest.skip("SM90 bf16x3 router GEMM requires a Hopper GPU")
    if not has_cutedsl():
        pytest.skip("cutedsl (cutlass) not installed")


@pytest.mark.parametrize(
    ("num_tokens", "hidden_dim", "num_experts"),
    [
        (48, 6144, 128),
        (96, 3072, 256),
        (129, 3072, 17),
        # long-K cases exercise the multi-accumulation path (the split-K
        # heuristic leaves chains of 15 and 32 K-tiles here, above the
        # kernel's num_tmem_acc bound)
        (1024, 8192, 256),
        (2048, 8192, 256),
    ],
)
def test_bf16x3_router_gemm_matches_reference(
    num_tokens: int, hidden_dim: int, num_experts: int
):
    _requires_sm100_cutedsl()
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_sm100 import (  # noqa: E501
        bf16x3_router_gemm,
    )

    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(num_experts, hidden_dim, dtype=torch.float32, device="cuda")
    w *= ROUTER_WEIGHT_SCALE
    out = bf16x3_router_gemm(x, w)
    # FP64 reference: the FP32 reference itself drifts by ~5e-6 at N=2048
    ref = torch.nn.functional.linear(x.double(), w.double())

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == torch.float32
    assert torch.mean(torch.abs(out.double() - ref)).item() < 5e-6


# (hidden_size, num_experts) of models that keep fp32 router weights.
SM90_ROUTER_SHAPES = [
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


def _sm90_inputs(num_tokens, hidden_size, num_experts):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, device="cuda").to(torch.bfloat16)
    w = torch.randn(num_experts, hidden_size, device="cuda") * ROUTER_WEIGHT_SCALE
    return x, w


def _mae(out, ref):
    return (out.double() - ref).abs().mean().item()


@pytest.mark.parametrize(("hidden_size", "num_experts"), SM90_ROUTER_SHAPES)
@pytest.mark.parametrize("num_tokens", [8, 33, 100, 513, 1024, 3072, 8192])
def test_sm90_bf16x3_router_gemm_beats_fp32_gemm(num_tokens, hidden_size, num_experts):
    _requires_sm90_cutedsl()
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_sm90 import (  # noqa: E501
        bf16x3_router_gemm,
    )

    x, w = _sm90_inputs(num_tokens, hidden_size, num_experts)
    out = bf16x3_router_gemm(x, w)
    ref = torch.nn.functional.linear(x.double(), w.double())
    # The fp32 SGEMM that SM90 runs without this kernel.
    fp32 = torch.nn.functional.linear(x.float(), w)

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == torch.float32
    assert _mae(out, ref) <= _mae(fp32, ref)


@pytest.mark.parametrize(
    ("hidden_size", "num_experts"),
    [
        (1032, 96),  # partial K-tile and partial expert tile
        (256, 64),  # fewer K-tiles than the largest split
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 40, 777])
def test_sm90_bf16x3_router_gemm_partial_tiles(num_tokens, hidden_size, num_experts):
    _requires_sm90_cutedsl()
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_sm90 import (  # noqa: E501
        bf16x3_router_gemm,
    )

    x, w = _sm90_inputs(num_tokens, hidden_size, num_experts)
    out = bf16x3_router_gemm(x, w)
    ref = torch.nn.functional.linear(x.double(), w.double())
    torch.testing.assert_close(out.double(), ref, rtol=0, atol=1e-5)


def test_sm90_bf16x3_weight_split_is_exact():
    """One-hot activations select single weights, so the three bf16 terms of
    each weight must sum back to it bit for bit."""
    _requires_sm90_cutedsl()
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


@pytest.mark.parametrize(
    ("input_size", "output_size", "num_tokens", "uses_bf16x3"),
    [
        # fp32 GEMV shapes keep the fp32 GEMV up to 24 tokens on SM90,
        # including the GB200 and B300 rules for Beam-1.0 and Step-5
        (5120, 256, 24, False),
        (5120, 256, 25, True),
        (5120, 256, 48, True),
        (4096, 352, 16, False),
        (4096, 352, 32, True),
        # other shapes use the kernel above the fp32 GEMV's 32 tokens
        (5120, 160, 33, True),
    ],
)
def test_sm90_gate_linear_dispatch(
    monkeypatch, input_size, output_size, num_tokens, uses_bf16x3
):
    _requires_sm90_cutedsl()
    for module in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(f"{module}.get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(f"{module}.get_tensor_model_parallel_world_size", lambda: 1)
    from vllm.model_executor.layers.fused_moe.router import bf16x3_router_gemm_sm90
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    calls = []
    # The fp32 GEMV tier calls the op body, the generic tier the public op.
    for name in ("_bf16x3_router_gemm", "bf16x3_router_gemm"):
        real = getattr(bf16x3_router_gemm_sm90, name)

        def record(x, w, real=real):
            calls.append(x.shape[0])
            return real(x, w)

        monkeypatch.setattr(bf16x3_router_gemm_sm90, name, record)
    gate = GateLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        out_dtype=torch.float32,
        params_dtype=torch.float32,
    ).cuda()
    gate.weight.data.normal_(0, ROUTER_WEIGHT_SCALE)
    x = torch.randn(num_tokens, input_size, device="cuda").to(torch.bfloat16)
    out, _ = gate(x)

    assert calls == ([num_tokens] if uses_bf16x3 else [])
    ref = torch.nn.functional.linear(x.double(), gate.weight.double())
    torch.testing.assert_close(out.double(), ref, rtol=0, atol=1e-4)
