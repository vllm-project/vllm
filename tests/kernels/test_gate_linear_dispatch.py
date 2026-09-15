# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-free eligibility tests for GateLinear's router GEMMs.

These assert the dispatch flags directly, so they run device-free by mocking
the platform predicates. ``allow_cublas_router_gemm`` selects the
bf16xbf16->fp32 ``torch.mm`` epilogue, while ``allow_fp32_router_gemm`` selects
the CUDA or gfx950 low-M kernel with fp32 weights and output.

The ROCm branch is guarded on ``not bias`` because ``torch.mm`` has no bias
term; a biased gate must fall back so the bias is not silently dropped.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.router.gate_linear as gate_linear_mod
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.platforms.interface import DeviceCapability


def _make_gate(
    monkeypatch,
    *,
    is_rocm: bool,
    is_cuda: bool = False,
    bias: bool = False,
    params_dtype: torch.dtype = torch.bfloat16,
    out_dtype: torch.dtype | None = torch.float32,
    input_size: int = 2048,
    output_size: int = 64,
    on_gfx950: bool = False,
    parallel_world_size: int = 1,
    device_capability: tuple[int, int] | None = None,
    force_fp32_compute: bool = False,
) -> GateLinear:
    """Build a GateLinear with platform predicates mocked, no GPU needed."""
    for target in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_rank",
            lambda: 0,
        )
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_world_size",
            lambda: parallel_world_size,
        )

    platform = gate_linear_mod.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(platform, "is_rocm", lambda: is_rocm)
    monkeypatch.setattr(
        type(platform),
        "get_device_capability",
        classmethod(
            lambda cls, device_id=0: (
                DeviceCapability(*device_capability) if device_capability else None
            )
        ),
    )
    if is_rocm:
        # The ROCm module queries the GCN architecture at import time.
        with monkeypatch.context() as mp:
            mp.setattr(
                torch.cuda,
                "get_device_properties",
                lambda device: SimpleNamespace(gcnArchName="gfx950"),
            )
            import vllm.platforms.rocm as rocm_platform

        monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: on_gfx950)

    return GateLinear(
        input_size=input_size,
        output_size=output_size,
        bias=bias,
        out_dtype=out_dtype,
        params_dtype=params_dtype,
        force_fp32_compute=force_fp32_compute,
    )


@pytest.mark.parametrize(
    "input_size,output_size,sm120_enabled", [(3072, 256, False), (6144, 128, True)]
)
@pytest.mark.parametrize(
    "device_capability,enabled",
    [
        ((9, 0), True),
        ((10, 0), True),
        ((10, 3), True),
        ((12, 0), True),
        ((12, 1), False),
        ((8, 6), False),
    ],
)
def test_cuda_fp32_router_architecture_gate(
    monkeypatch, input_size, output_size, device_capability, enabled, sm120_enabled
):
    """Admit exact SM120 without extending admission to the whole family."""
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        params_dtype=torch.float32,
        input_size=input_size,
        output_size=output_size,
        device_capability=device_capability,
    )
    expected = sm120_enabled if device_capability == (12, 0) else enabled
    assert gate.allow_fp32_router_gemm == expected
    if device_capability == (12, 0):
        assert not gate.allow_specialized_router_gemm
        assert not gate.allow_ll_bf16_gemm
        assert not gate.allow_bf16x3_router_gemm
        assert not gate.allow_cublas_router_gemm


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_cuda": False},
        {"params_dtype": torch.bfloat16},
        {"params_dtype": torch.float16},
        {"input_size": 4096},
        {"output_size": 256},
        {"bias": True},
    ],
)
def test_sm120_fp32_router_rejects_unsupported_configs(monkeypatch, overrides):
    kwargs = dict(
        is_rocm=False,
        is_cuda=True,
        params_dtype=torch.float32,
        input_size=6144,
        output_size=128,
        device_capability=(12, 0),
    )
    gate = _make_gate(monkeypatch, **(kwargs | overrides))
    assert not gate.allow_fp32_router_gemm


@pytest.mark.parametrize(
    "overrides,input_dtype",
    [
        ({"bias": True}, torch.bfloat16),
        ({"input_size": 4096}, torch.bfloat16),
        ({"input_size": 3072, "output_size": 256}, torch.bfloat16),
        ({"params_dtype": torch.bfloat16}, torch.float32),
        ({"params_dtype": torch.float16}, torch.float32),
        ({"device_capability": (12, 1)}, torch.bfloat16),
        ({"device_capability": (8, 6)}, torch.bfloat16),
        ({}, torch.float16),
    ],
)
@torch.inference_mode()
def test_sm120_exclusions_use_linear_without_dropping_bias(
    monkeypatch, overrides, input_dtype
):
    kwargs = dict(
        is_rocm=False,
        is_cuda=True,
        params_dtype=torch.float32,
        input_size=6144,
        output_size=128,
        device_capability=(12, 0),
    )
    gate = _make_gate(monkeypatch, **(kwargs | overrides))
    gate.weight.zero_()
    if gate.bias is not None:
        gate.bias.fill_(2)

    def reject_native(*args):
        pytest.fail("Excluded configuration reached FP32 router dispatch")

    monkeypatch.setattr(torch.ops.vllm, "fp32_router_gemm_dispatch", reject_native)
    x = torch.ones(2, gate.weight.shape[1], dtype=input_dtype)
    output, output_bias = gate(x)
    assert output_bias is None
    assert output.dtype == torch.float32
    expected = torch.full(
        (2, gate.weight.shape[0]), 2.0 if gate.bias is not None else 0.0
    )
    torch.testing.assert_close(output, expected)


def test_sm120_force_fp32_compute_preserves_fp32_weights(monkeypatch):
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        params_dtype=torch.bfloat16,
        input_size=6144,
        output_size=128,
        device_capability=(12, 0),
        force_fp32_compute=True,
    )
    assert gate.weight.dtype == torch.float32
    assert gate.allow_fp32_router_gemm
    assert not gate.allow_specialized_router_gemm


@pytest.mark.parametrize("deferred", [False, True])
def test_sm120_fp32_router_preserves_requested_output_dtype(monkeypatch, deferred):
    """The FP32-compute path must honor explicit and deferred output casts."""
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        params_dtype=torch.float32,
        out_dtype=None if deferred else torch.bfloat16,
        input_size=6144,
        output_size=128,
        device_capability=(12, 0),
    )
    if deferred:
        gate.set_out_dtype(torch.bfloat16)
    # Run the real large-M dispatch on CPU, without CUDA op registration.
    monkeypatch.setattr(
        torch.ops.vllm,
        "fp32_router_gemm_dispatch",
        gate_linear_mod.fp32_router_gemm_dispatch_impl,
    )
    with torch.no_grad():
        gate.weight.fill_(1.0 / 6144)
    x = torch.ones(33, 6144, dtype=torch.bfloat16)
    output, bias = gate(x)
    assert bias is None
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, torch.ones(33, 128, dtype=torch.bfloat16))


@pytest.mark.parametrize("device_capability", [(9, 0), (10, 0), (12, 0)])
@pytest.mark.parametrize(
    "dtype,num_tokens,sm120_native",
    [
        (torch.float32, 16, True),
        (torch.float32, 17, False),
        (torch.float32, 32, False),
        (torch.bfloat16, 17, True),
        (torch.bfloat16, 32, True),
        (torch.bfloat16, 33, False),
    ],
)
def test_fp32_router_runtime_batch_limit(
    monkeypatch, device_capability, dtype, num_tokens, sm120_native
):
    """Bound SM120 FP32 batches without changing BF16 or earlier CUDA dispatch."""
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        params_dtype=torch.float32,
        input_size=6144,
        output_size=128,
        device_capability=device_capability,
    )
    native_calls = []

    def native_router(x, weight):
        native_calls.append(x.shape[0])
        return torch.nn.functional.linear(x.float(), weight)

    monkeypatch.setattr(gate_linear_mod.ops, "fp32_router_gemm", native_router)
    with torch.no_grad():
        gate.weight.zero_()
        x = torch.ones(num_tokens, 6144, dtype=dtype)
        output = gate_linear_mod.fp32_router_gemm_dispatch_impl(x, gate.weight, False)
    expected_native = sm120_native if device_capability == (12, 0) else num_tokens <= 32
    assert bool(native_calls) == expected_native
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.zeros(num_tokens, 128))


@pytest.mark.parametrize("device_capability", [(9, 0), (10, 0), None])
@pytest.mark.parametrize("num_tokens", [1, 33])
@pytest.mark.parametrize("deferred", [False, True])
@torch.inference_mode()
def test_existing_fp32_router_routes_keep_fp32_output(
    monkeypatch, device_capability, num_tokens, deferred
):
    """SM120 output casts must not change the pre-existing CUDA/gfx950 tier."""
    is_rocm = device_capability is None
    gate = _make_gate(
        monkeypatch,
        is_rocm=is_rocm,
        is_cuda=not is_rocm,
        params_dtype=torch.float32,
        out_dtype=None if deferred else torch.bfloat16,
        input_size=6144,
        output_size=128,
        device_capability=device_capability,
        on_gfx950=is_rocm,
    )
    if deferred:
        gate.set_out_dtype(torch.bfloat16)
    monkeypatch.setattr(
        gate_linear_mod.ops, "fp32_router_gemm", torch.nn.functional.linear
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "fp32_router_gemm_dispatch",
        gate_linear_mod.fp32_router_gemm_dispatch_impl,
    )
    gate.weight.fill_(1.0 / 6144)
    output, bias = gate(torch.ones(num_tokens, 6144))
    assert bias is None
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.ones(num_tokens, 128))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided_operand", ["input", "weight"])
@torch.inference_mode()
def test_sm120_fp32_router_preserves_strided_linear_inputs(
    monkeypatch, dtype, strided_operand
):
    """Unsupported native layouts must retain the linear fallback contract."""
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        params_dtype=torch.float32,
        input_size=6144,
        output_size=128,
        device_capability=(12, 0),
    )
    x = torch.ones(2, 6144, dtype=dtype)
    if strided_operand == "input":
        x = torch.ones(2, 12288, dtype=dtype)[:, ::2]
    else:
        gate.weight.data = torch.empty(128, 12288)[:, ::2]
    gate.weight.fill_(1.0 / 6144)
    assert not x.is_contiguous() or not gate.weight.is_contiguous()

    def reject_native(*args):
        pytest.fail("Strided input reached the contiguous-only CUDA router")

    monkeypatch.setattr(gate_linear_mod.ops, "fp32_router_gemm", reject_native)
    monkeypatch.setattr(
        torch.ops.vllm,
        "fp32_router_gemm_dispatch",
        gate_linear_mod.fp32_router_gemm_dispatch_impl,
    )
    output, bias = gate(x)
    assert bias is None
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.ones(2, 128))
