# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the gfx950 low-M fp32 router GEMM."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.gate_linear import (
    fp32_router_gemm_dispatch_impl,
)
from vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm import (
    rocm_fp32_router_gemm,
)
from vllm.platforms import current_platform

SHAPES = [
    (3072, 256),
    (4096, 8),
    (4096, 192),
    (6144, 128),
    (6144, 256),
]
MAX_TOKENS = 32
ATOL = 5e-4
RTOL = 0.0


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import on_gfx950

        return on_gfx950()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="rocm_fp32_router_gemm requires ROCm gfx950"
)


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x.float(), weight)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("hidden_size", "num_experts"), SHAPES)
@pytest.mark.parametrize("num_tokens", range(MAX_TOKENS + 1))
@torch.inference_mode()
def test_rocm_fp32_router_gemm_matches_reference(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(41 + num_tokens + hidden_size + num_experts)
    device = torch.device("cuda")
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)

    output = rocm_fp32_router_gemm(x, weight)
    expected = _reference(x, weight)

    assert output.shape == (num_tokens, num_experts)
    assert output.dtype == torch.float32
    assert output.device == x.device
    torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 24, 32])
@pytest.mark.parametrize(("hidden_size", "num_experts"), SHAPES)
@torch.inference_mode()
def test_rocm_fp32_router_gemm_preserves_topk(
    num_tokens: int, hidden_size: int, num_experts: int
) -> None:
    torch.manual_seed(1000 + num_tokens + hidden_size + num_experts)
    device = torch.device("cuda")
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)

    top_k = 2 if (hidden_size, num_experts) == (4096, 8) else 8
    actual = rocm_fp32_router_gemm(x, weight).topk(top_k, dim=-1).indices
    reference = _reference(x, weight)
    reference_values, expected = reference.topk(top_k, dim=-1)
    for token in range(num_tokens):
        actual_set = set(actual[token].tolist())
        expected_set = set(expected[token].tolist())
        if actual_set == expected_set:
            continue
        kth_value = reference_values[token, -1].item()
        for expert in actual_set.symmetric_difference(expected_set):
            gap = abs(reference[token, expert].item() - kth_value)
            assert gap < 1e-3


@torch.inference_mode()
def test_rocm_fp32_router_gemm_rejects_invalid_inputs() -> None:
    device = torch.device("cuda")
    hidden_size, num_experts = 6144, 128
    x = torch.randn(4, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="num_tokens"):
        rocm_fp32_router_gemm(
            torch.randn(
                MAX_TOKENS + 1,
                hidden_size,
                dtype=torch.bfloat16,
                device=device,
            ),
            weight,
        )

    with pytest.raises(ValueError, match="shape"):
        rocm_fp32_router_gemm(x[:, :2048].contiguous(), weight[:, :2048].contiguous())

    with pytest.raises(ValueError, match="contiguous"):
        wide_x = torch.randn(4, hidden_size * 2, dtype=torch.bfloat16, device=device)
        rocm_fp32_router_gemm(wide_x[:, ::2], weight)

    with pytest.raises(ValueError, match="float32"):
        rocm_fp32_router_gemm(x, weight.to(torch.bfloat16))

    with pytest.raises(ValueError, match="bfloat16 or float32"):
        rocm_fp32_router_gemm(x.to(torch.float16), weight)


@torch.inference_mode()
def test_rocm_fp32_router_gemm_cuda_graph_observes_input_mutation() -> None:
    torch.manual_seed(2026)
    device = torch.device("cuda")
    hidden_size, num_experts = 6144, 128
    static_x = torch.randn(16, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)

    rocm_fp32_router_gemm(static_x, weight)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = rocm_fp32_router_gemm(static_x, weight)
    torch.accelerator.synchronize()

    first_input = torch.randn_like(static_x)
    static_x.copy_(first_input)
    graph.replay()
    torch.accelerator.synchronize()
    first_output = captured_output.clone()

    second_input = torch.randn_like(static_x)
    static_x.copy_(second_input)
    graph.replay()
    torch.accelerator.synchronize()
    second_output = captured_output.clone()

    torch.testing.assert_close(
        first_output, _reference(first_input, weight), atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        second_output, _reference(second_input, weight), atol=ATOL, rtol=RTOL
    )
    assert not torch.equal(first_output, second_output)


@pytest.mark.parametrize("num_tokens", [4, 32, 33])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_rocm_fp32_router_gemm_custom_op_dispatch(
    num_tokens: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(3100 + num_tokens)
    device = torch.device("cuda")
    hidden_size, num_experts = 6144, 128
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)

    output = torch.ops.vllm.fp32_router_gemm_dispatch(x, weight, False)
    direct_output = fp32_router_gemm_dispatch_impl(x, weight, False)

    torch.testing.assert_close(output, direct_output, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(output, _reference(x, weight), atol=ATOL, rtol=RTOL)


@torch.inference_mode()
def test_rocm_fp32_router_gemm_dispatch_accepts_noncontiguous_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(4100)
    device = torch.device("cuda")
    hidden_size, num_experts = 6144, 128
    wide_x = torch.randn(4, hidden_size * 2, dtype=torch.bfloat16, device=device)
    x = wide_x[:, ::2]
    weight = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)
    assert not x.is_contiguous()

    expected = _reference(x, weight)

    def fail_fallback(*args, **kwargs):
        raise AssertionError("noncontiguous low-M input used the linear fallback")

    monkeypatch.setattr(torch.nn.functional, "linear", fail_fallback)
    output = torch.ops.vllm.fp32_router_gemm_dispatch(x, weight, False)

    torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)


@torch.inference_mode()
def test_rocm_fp32_router_gemm_dynamic_compile_dispatch() -> None:
    torch.manual_seed(5100)
    device = torch.device("cuda")
    hidden_size, num_experts = 6144, 128
    weight = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)

    def dispatch(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.fp32_router_gemm_dispatch(x, weight, False)

    compiled_dispatch = torch.compile(dispatch, dynamic=True, fullgraph=True)
    for num_tokens in (4, 16, 33, 5, 32):
        x = torch.randn(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        output = compiled_dispatch(x)
        torch.testing.assert_close(output, _reference(x, weight), atol=ATOL, rtol=RTOL)
