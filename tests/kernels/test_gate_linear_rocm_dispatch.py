# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-agnostic eligibility tests for GateLinear's fused router GEMM.

These assert the ``allow_cublas_router_gemm`` dispatch flag directly, so they
run device-free by mocking the platform predicates. The flag decides whether
the bf16xbf16->fp32 router GEMM uses ``torch.mm``'s fused out_dtype epilogue
(one kernel) or falls back to a bf16 matmul plus a standalone bf16->fp32 copy.

The ROCm branch is guarded on ``not bias`` because ``torch.mm`` has no bias
term; a biased gate must fall back so the bias is not silently dropped.
"""

import torch

import vllm.model_executor.layers.fused_moe.router.gate_linear as gate_linear_mod
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


def _make_gate(
    monkeypatch,
    *,
    is_rocm: bool,
    is_cuda: bool = False,
    bias: bool = False,
    params_dtype: torch.dtype = torch.bfloat16,
    out_dtype: torch.dtype | None = torch.float32,
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
            lambda: 1,
        )

    platform = gate_linear_mod.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(platform, "is_rocm", lambda: is_rocm)
    # Force the CUDA specialized-kernel gate off so ROCm eligibility is the
    # only thing under test (these are the SM90/SM100 capability checks).
    monkeypatch.setattr(platform, "is_device_capability", lambda *a, **k: False)
    monkeypatch.setattr(platform, "is_device_capability_family", lambda *a, **k: False)

    return GateLinear(
        input_size=2048,
        output_size=64,
        bias=bias,
        out_dtype=out_dtype,
        params_dtype=params_dtype,
    )


def test_rocm_no_bias_bf16_fp32_enables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=False)
    assert not gate.allow_specialized_router_gemm
    assert gate.allow_cublas_router_gemm


def test_rocm_bias_disables_fused_gemm(monkeypatch):
    # torch.mm cannot add a bias, so a biased gate must not take the fused path.
    gate = _make_gate(monkeypatch, is_rocm=True, bias=True)
    assert not gate.allow_cublas_router_gemm


def test_rocm_fp32_weight_disables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, params_dtype=torch.float32)
    assert not gate.allow_cublas_router_gemm


def test_rocm_non_fp32_out_dtype_disables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, out_dtype=torch.bfloat16)
    assert not gate.allow_cublas_router_gemm


def test_non_rocm_non_cuda_disables_fused_gemm(monkeypatch):
    # Neither the CUDA specialized path nor the ROCm branch applies.
    gate = _make_gate(monkeypatch, is_rocm=False, is_cuda=False)
    assert not gate.allow_cublas_router_gemm


def test_rocm_set_out_dtype_enables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=False, out_dtype=None)
    assert not gate.allow_cublas_router_gemm
    gate.set_out_dtype(torch.float32)
    assert gate.allow_cublas_router_gemm


def test_rocm_set_out_dtype_respects_bias_guard(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=True, out_dtype=None)
    gate.set_out_dtype(torch.float32)
    assert not gate.allow_cublas_router_gemm
