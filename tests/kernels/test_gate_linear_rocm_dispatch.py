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


# ---------------------------------------------------------------------------
# ROCm BF16x3 tier (fp32 router weights)
# ---------------------------------------------------------------------------
def _make_bf16x3_gate(monkeypatch, *, supported=True, **kwargs):
    """Build a gate with the ROCm bf16x3 platform check mocked."""
    from vllm.model_executor.layers.fused_moe.router import (
        bf16x3_router_gemm_rocm as rocm_bf16x3,
    )

    monkeypatch.setattr(rocm_bf16x3, "platform_supported", lambda: supported)
    kwargs.setdefault("params_dtype", torch.float32)
    return _make_gate(monkeypatch, is_rocm=True, **kwargs)


def test_rocm_bf16x3_enabled_for_fp32_weight(monkeypatch):
    gate = _make_bf16x3_gate(monkeypatch)
    assert gate.allow_rocm_bf16x3_router_gemm
    # Built by process_weights_after_loading, not lazily in forward().
    # Registered up front but still None, so named_buffers() skips it.
    assert gate._bf16x3_weight is None
    assert "_bf16x3_weight" in gate._buffers


def test_rocm_bf16x3_disabled_by_bias(monkeypatch):
    # torch.mm has no bias term.
    gate = _make_bf16x3_gate(monkeypatch, bias=True)
    assert not gate.allow_rocm_bf16x3_router_gemm


def test_rocm_bf16x3_disabled_for_bf16_weight(monkeypatch):
    # A bf16 weight belongs to the cuBLAS tier; there is nothing to split.
    gate = _make_bf16x3_gate(monkeypatch, params_dtype=torch.bfloat16)
    assert not gate.allow_rocm_bf16x3_router_gemm


def test_rocm_bf16x3_disabled_for_non_fp32_out_dtype(monkeypatch):
    gate = _make_bf16x3_gate(monkeypatch, out_dtype=torch.bfloat16)
    assert not gate.allow_rocm_bf16x3_router_gemm


def test_rocm_bf16x3_disabled_on_unsupported_arch(monkeypatch):
    # Only gfx950 is benchmarked; gfx942 has a different fp32:bf16 MFMA ratio.
    gate = _make_bf16x3_gate(monkeypatch, supported=False)
    assert not gate.allow_rocm_bf16x3_router_gemm
    assert not gate._rocm_bf16x3_weight_eligible


def test_rocm_bf16x3_no_split_built_when_tier_disabled(monkeypatch):
    """An ineligible out_dtype must not allocate a split that is never read;
    it is 1.5x the router weight on every layer."""
    gate = _make_bf16x3_gate(monkeypatch, out_dtype=torch.bfloat16)
    gate.quant_method.process_weights_after_loading(gate)
    assert gate._bf16x3_weight is None


def test_rocm_bf16x3_set_out_dtype_enables_tier(monkeypatch):
    gate = _make_bf16x3_gate(monkeypatch, out_dtype=None)
    assert not gate.allow_rocm_bf16x3_router_gemm
    gate.set_out_dtype(torch.float32)
    assert gate.allow_rocm_bf16x3_router_gemm


def test_rocm_bf16x3_dispatch_is_a_custom_op():
    """The num_tokens branch must stay opaque to Dynamo; see the op's impl."""
    assert hasattr(torch.ops.vllm, "rocm_bf16x3_router_gemm_dispatch")
    x = torch.zeros(8192, 6144, dtype=torch.bfloat16, device="meta")
    w = torch.zeros(128, 6144, dtype=torch.float32, device="meta")
    split = torch.zeros(3, 128, 6144, dtype=torch.bfloat16, device="meta")
    out = torch.ops.vllm.rocm_bf16x3_router_gemm_dispatch(x, w, split)
    assert out.shape == (8192, 128)
    assert out.dtype == torch.float32
