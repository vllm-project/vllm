# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron-H batch-invariant pieces outside the BI kernel library.

Each kernel must give a token the same bits whether it runs alone or inside a
larger batch: the first m rows of an M-row call equal an m-row call. Shapes are
the Nemotron-H Lightning ones (hidden 2688, 128 experts, shared expert 3712).
"""

import pytest
import torch
from utils import skip_if_not_cuda

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

HIDDEN = 2688
ROWS = [1, 2, 7, 16, 33, 64, 256, 1024]
MAX_ROWS = 4096

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="Nemotron-H batch-invariant kernels target SM10x",
)


def _assert_rows_invariant(fn, x):
    full = fn(x)
    for m in ROWS:
        part = fn(x[:m].contiguous())
        if isinstance(full, tuple):
            for p, f in zip(part, full):
                assert torch.equal(p, f[:m]), m
        else:
            assert torch.equal(part, full[:m]), m


@skip_if_not_cuda
@torch.inference_mode()
def test_nemotron_h_norms_are_row_invariant():
    from vllm.model_executor.layers.layernorm import cuda_rms_norm as rms_forward
    from vllm.model_executor.layers.mamba.mamba_mixer2 import (
        grouped_gated_rms_norm as gated_forward,
    )

    torch.manual_seed(0)
    x = torch.randn(MAX_ROWS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)
    _assert_rows_invariant(lambda t: rms_forward(t, weight, 1e-5), x)
    _assert_rows_invariant(
        lambda t: rms_forward(t, weight, 1e-5, residual[: t.shape[0]]), x
    )
    # Out of place: the caller's tensors are left untouched.
    x_copy, residual_copy = x.clone(), residual.clone()
    rms_forward(x, weight, 1e-5, residual)
    assert torch.equal(x, x_copy) and torch.equal(residual, residual_copy)

    # Mamba gated norm: intermediate 4096 in 8 groups.
    z = torch.randn(MAX_ROWS, 4096, device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(z)
    gate_weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    _assert_rows_invariant(
        lambda t: gated_forward(t, z[: t.shape[0]], gate_weight, 512, 1e-5), y
    )


def _nemotron_gate(monkeypatch):
    for module in ("layers.linear", "parameter"):
        prefix = f"vllm.model_executor.{module}.get_tensor_model_parallel"
        monkeypatch.setattr(f"{prefix}_rank", lambda: 0)
        monkeypatch.setattr(f"{prefix}_world_size", lambda: 1)
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    # As nemotron_h.py builds it under batch invariance: 128 experts, BF16
    # weight, FP32 logits.
    gate = GateLinear(
        HIDDEN,
        128,
        out_dtype=torch.float32,
        params_dtype=torch.bfloat16,
        force_fp32_compute=False,
    ).cuda()
    assert gate.weight.dtype == torch.bfloat16
    with torch.no_grad():
        gate.weight.normal_(generator=torch.Generator("cuda").manual_seed(1))
        gate.weight.div_(50)
    return gate


@requires_sm100
def test_nemotron_h_router_logits_are_row_invariant(monkeypatch):
    """Under batch invariance the router GEMM is the persistent Triton matmul
    with an FP32 output, whose K reduction order does not depend on M; cuBLAS,
    which serves it otherwise, gives no such guarantee. The trainer must call
    mm_batch_invariant(x, weight.T, out_dtype=torch.float32) too."""
    from vllm.model_executor.determinism.batch_invariant import mm_batch_invariant

    gate = _nemotron_gate(monkeypatch)
    torch.manual_seed(0)
    # Trainer micro-batches reach 8K-16K packed tokens.
    x = torch.randn(16384, HIDDEN, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        full, _ = gate(x)
        assert full.dtype == torch.float32
        expected = mm_batch_invariant(x, gate.weight.T, out_dtype=torch.float32)
        assert torch.equal(full, expected)
        for m in (1, 7, 128, 4096, 8192):
            head, _ = gate(x[:m].contiguous())
            tail, _ = gate(x[-m:].contiguous())
            assert torch.equal(head, full[:m]), m
            assert torch.equal(tail, full[-m:]), m
        reference = x.float() @ gate.weight.float().T
    torch.testing.assert_close(full, reference, rtol=1e-4, atol=1e-4)


@requires_sm100
@torch.inference_mode()
@pytest.mark.parametrize("n,k", [(3712, HIDDEN), (HIDDEN, 3712)])
def test_nemotron_h_shared_expert_w4a16_is_row_invariant(n, k):
    from vllm.model_executor.kernels.linear.nvfp4.base import (
        NvFp4LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
        NemotronSharedNvFp4LinearKernel,
    )

    supported, reason = NemotronSharedNvFp4LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    torch.manual_seed(0)
    weight = (torch.randn(n, k, device="cuda") / 30).to(torch.bfloat16)
    global_scale = (448.0 * 6.0 / weight.abs().max()).float()
    packed, block_scale = ops.scaled_fp4_quant(
        weight, global_scale, is_sf_swizzled_layout=False
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(block_scale, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        (1.0 / global_scale).reshape(()), requires_grad=False
    )
    layer.output_size_per_partition = n
    expected_scale = 1.0 / (1.0 / layer.weight_global_scale.detach().clone())
    kernel = NemotronSharedNvFp4LinearKernel(NvFp4LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    assert torch.equal(
        layer.weight_global_scale.reshape(()), expected_scale.reshape(())
    )

    x = torch.randn(MAX_ROWS, k, device="cuda", dtype=torch.bfloat16) / 4
    _assert_rows_invariant(lambda t: kernel.apply_weights(layer, t), x)


@requires_sm100
@torch.inference_mode()
@pytest.mark.parametrize("n,k", [(10304, HIDDEN), (HIDDEN, 4096)])
def test_flashinfer_fp8_gemm_is_row_invariant(n, k):
    from vllm.utils.flashinfer import (
        _bmm_fp8_backend,
        flashinfer_scaled_fp8_mm,
        has_flashinfer,
    )

    if not has_flashinfer():
        pytest.skip("FlashInfer is not installed")
    assert _bmm_fp8_backend() == "cutlass"
    torch.manual_seed(0)
    weight = (torch.randn(n, k, device="cuda") / 30).to(torch.float8_e4m3fn)
    scale_a = torch.tensor(0.02, device="cuda")
    scale_b = torch.tensor(0.01, device="cuda")
    x = (torch.randn(MAX_ROWS, k, device="cuda") * 4).to(torch.float8_e4m3fn)
    _assert_rows_invariant(
        lambda t: flashinfer_scaled_fp8_mm(
            t, weight.t(), scale_a, scale_b, torch.bfloat16
        ),
        x,
    )


@skip_if_not_cuda
def test_batch_invariant_w4a16_linear_uses_humming(monkeypatch, caplog_vllm):
    from vllm.model_executor import kernels
    from vllm.model_executor.kernels import linear
    from vllm.model_executor.kernels.linear.nvfp4.humming import (
        HummingNvFp4LinearKernel,
    )

    monkeypatch.setattr(
        linear, "_get_linear_backend", lambda quantization: "flashinfer-cutedsl"
    )
    kernel = kernels.linear.init_nvfp4_linear_kernel(use_a16=True)
    assert isinstance(kernel, HummingNvFp4LinearKernel)
    assert "overrides --linear-backend=flashinfer-cutedsl" in caplog_vllm.text
