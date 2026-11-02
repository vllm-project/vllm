# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the gfx950 low-M fp32 router GEMM."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router import (
    rocm_fp32_router_gemm as rocm_router,
)
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
    (8192, 256),
    (2560, 512),
    (2880, 128),
    (5120, 16),
    (5120, 384),
    (7168, 384),
    (1000, 100),
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

    top_k = 2 if num_experts <= 16 else 8
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

    with pytest.raises(ValueError, match="hidden sizes"):
        rocm_fp32_router_gemm(x, weight[:, :2048].contiguous())

    with pytest.raises(ValueError, match="num_experts"):
        big_weight = torch.empty(1024, 7168, dtype=torch.float32, device=device)
        rocm_fp32_router_gemm(
            torch.empty(MAX_TOKENS, 7168, dtype=torch.bfloat16, device=device),
            big_weight,
        )

    with pytest.raises(ValueError, match="contiguous"):
        wide_x = torch.randn(4, hidden_size * 2, dtype=torch.bfloat16, device=device)
        rocm_fp32_router_gemm(wide_x[:, ::2], weight)

    with pytest.raises(ValueError, match="float32 or bfloat16"):
        rocm_fp32_router_gemm(x, weight.to(torch.float16))

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


@pytest.mark.parametrize("hidden_size", [64, 100, 513, 1000, 2049, 3000])
@pytest.mark.parametrize("num_tokens", [1, 5, 32])
@torch.inference_mode()
def test_rocm_fp32_router_gemm_any_hidden_size(num_tokens: int, hidden_size: int):
    """Hidden sizes that do not divide the K tile exercise the masked tail."""
    torch.manual_seed(6100 + num_tokens + hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(64, hidden_size, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(
        rocm_fp32_router_gemm(x, weight), _reference(x, weight), atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("case", ["large", "near_zero", "zero_input", "ones"])
@torch.inference_mode()
def test_rocm_fp32_router_gemm_special_values(case: str) -> None:
    torch.manual_seed(7100)
    x = torch.randn(4, 5120, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(128, 5120, dtype=torch.float32, device="cuda")
    if case in ("large", "near_zero"):
        scale = 100.0 if case == "large" else 1e-4
        x = (x * scale).to(torch.bfloat16)
        weight *= scale
    elif case == "zero_input":
        x.zero_()
    else:
        x.fill_(1.0)
    output = rocm_fp32_router_gemm(x, weight)
    expected = _reference(x, weight)
    assert torch.isfinite(output).all()
    if case == "zero_input":
        assert torch.equal(output, torch.zeros_like(output))
    else:
        assert ((output - expected).norm() / expected.norm()).item() < 1e-5


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32])
@torch.inference_mode()
def test_rocm_fp32_router_gemm_deterministic(num_tokens: int) -> None:
    torch.manual_seed(8100 + num_tokens)
    x = torch.randn(num_tokens, 7168, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(384, 7168, dtype=torch.float32, device="cuda")
    first = rocm_fp32_router_gemm(x, weight)
    assert torch.equal(first, rocm_fp32_router_gemm(x, weight))


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32])
@pytest.mark.parametrize(("hidden_size", "num_experts"), SHAPES)
@torch.inference_mode()
def test_rocm_fp32_router_gemm_bf16_weight(
    num_tokens: int, hidden_size: int, num_experts: int
) -> None:
    torch.manual_seed(9100 + num_tokens + hidden_size + num_experts)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16, device="cuda")
    # Past this size hipBLASLt beats the kernel for BF16 weights on MI355.
    eligible = num_tokens * num_experts <= 8192
    assert rocm_router.can_use_rocm_fp32_router_gemm(x, weight) == eligible
    if eligible:
        torch.testing.assert_close(
            rocm_fp32_router_gemm(x, weight),
            _reference(x.float(), weight.float()),
            atol=ATOL,
            rtol=RTOL,
        )


def _make_gate_linear(
    monkeypatch: pytest.MonkeyPatch,
    input_size,
    output_size,
    params_dtype=torch.float32,
):
    for module in ("layers.linear", "parameter"):
        monkeypatch.setattr(
            f"vllm.model_executor.{module}.get_tensor_model_parallel_rank", lambda: 0
        )
        monkeypatch.setattr(
            f"vllm.model_executor.{module}.get_tensor_model_parallel_world_size",
            lambda: 1,
        )
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    gate = GateLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        out_dtype=torch.float32,
        params_dtype=params_dtype,
    ).cuda()
    with torch.no_grad():
        gate.weight.normal_()
    gate.quant_method.process_weights_after_loading(gate)
    return gate


@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens", [1, 32, 33])
@torch.inference_mode()
def test_gate_linear_uses_rocm_fp32_router_gemm(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int, x_dtype: torch.dtype
) -> None:
    """A router shape outside the old tuned list takes the fast path for M<=32."""
    gate = _make_gate_linear(monkeypatch, 2560, 512)
    assert gate.allow_fp32_router_gemm
    calls = []
    kernel = rocm_router.rocm_fp32_router_gemm

    def spy(hidden_states, router_weight):
        calls.append(hidden_states.shape[0])
        return kernel(hidden_states, router_weight)

    monkeypatch.setattr(rocm_router, "rocm_fp32_router_gemm", spy)
    x = torch.randn(num_tokens, 2560, dtype=x_dtype, device="cuda")
    output, bias = gate(x)

    assert bias is None
    assert calls == ([num_tokens] if num_tokens <= MAX_TOKENS else [])
    torch.testing.assert_close(output, _reference(x, gate.weight), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    ("num_tokens", "num_experts", "uses_kernel"),
    [(1, 256, True), (32, 256, True), (33, 256, False), (32, 512, False)],
)
@torch.inference_mode()
def test_gate_linear_bf16_weight_uses_rocm_router_gemm(
    monkeypatch: pytest.MonkeyPatch, num_tokens, num_experts, uses_kernel
) -> None:
    """BF16-weight gates with FP32 logits use the kernel only where it beats
    hipBLASLt; everything else keeps the torch.mm path."""
    gate = _make_gate_linear(monkeypatch, 7168, num_experts, torch.bfloat16)
    assert gate.allow_cublas_router_gemm
    calls = []
    kernel = rocm_router.rocm_fp32_router_gemm

    def spy(hidden_states, router_weight):
        calls.append(hidden_states.shape[0])
        return kernel(hidden_states, router_weight)

    monkeypatch.setattr(rocm_router, "rocm_fp32_router_gemm", spy)
    x = torch.randn(num_tokens, 7168, dtype=torch.bfloat16, device="cuda")
    output, _ = gate(x)

    assert output.dtype == torch.float32
    assert calls == ([num_tokens] if uses_kernel else [])
    torch.testing.assert_close(
        output, _reference(x.float(), gate.weight.float()), atol=ATOL, rtol=RTOL
    )
