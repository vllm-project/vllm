# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the RDNA3/RDNA3.5 (gfx11) MXFP4 W4A8 GEMV and its linear kernel.

Run `pytest tests/kernels/quantization/test_mxfp4_w4a8_rdna.py`.
"""

from unittest.mock import patch

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.kernels.linear import (
    MxFp4LinearLayerConfig,
    RdnaW4A8MxFp4LinearKernel,
    init_mxfp4_linear_kernel,
)
from vllm.model_executor.kernels.linear.mxfp4.rdna_w4a8 import (
    MAX_W4A8_BATCH_SIZE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
    kMxfp6E3M2Dynamic,
)
from vllm.platforms import current_platform

_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = torch.tensor(_E2M1_MAG + [-v for v in _E2M1_MAG], dtype=torch.float64)


def _on_gfx11() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx11

    return on_gfx11()


def _has_op() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "mxfp4_w4a8_gemv"
    )


gfx11_only = pytest.mark.skipif(
    not (_on_gfx11() and _has_op()),
    reason="requires gfx11 and _rocm_C.mxfp4_w4a8_gemv",
)


def _random_mxfp4(N: int, K: int, device: str = "cuda"):
    w_q = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
    # E8M0 scales near 2^0 keep the products well inside fp16/bf16 range.
    w_s = torch.randint(122, 131, (N, K // 32), dtype=torch.uint8, device=device)
    return w_q, w_s


def _dequant_ref(w_q: torch.Tensor, w_s: torch.Tensor) -> torch.Tensor:
    """[N, K/2] uint8 E2M1 + [N, K/32] uint8 E8M0 -> [N, K] float64."""
    N, k_half = w_q.shape
    codes = torch.stack([w_q & 0xF, w_q >> 4], dim=-1).reshape(N, 2 * k_half)
    vals = E2M1.to(w_q.device)[codes.long()]
    scale = torch.pow(2.0, w_s.double() - 127.0).repeat_interleave(32, dim=1)
    return vals * scale


def _w4a8_ref(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor) -> torch.Tensor:
    """Reference for the W4A8 math: int8 per-32 activation quant, exact dot."""
    M, K = x.shape
    xg = x.double().view(M, K // 32, 32)
    a_s = (xg.abs().amax(dim=-1) / 127.0).clamp_min(1e-12)
    a_q = torch.round(xg / a_s[..., None]).clamp(-127, 127)
    x_qdq = (a_q * a_s[..., None]).view(M, K)
    return x_qdq @ _dequant_ref(w_q, w_s).t()


@gfx11_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M", list(range(1, MAX_W4A8_BATCH_SIZE + 1)))
@pytest.mark.parametrize("N,K", [(64, 256), (256, 512), (128, 4096), (40, 5120)])
def test_w4a8_gemv_matches_reference(dtype, M, N, K):
    torch.manual_seed(0)
    w_q, w_s = _random_mxfp4(N, K)
    x = (torch.randn(M, K, device="cuda") * 0.1).to(dtype)

    out = torch.ops._rocm_C.mxfp4_w4a8_gemv(x, w_q, w_s)
    assert out.shape == (M, N) and out.dtype == dtype

    ref = _w4a8_ref(x, w_q, w_s)
    # Bound by the output rounding of the accumulated magnitude; a quant code
    # that rounds the other way on a tie stays well inside this.
    accum = x.double().abs() @ _dequant_ref(w_q, w_s).abs().t()
    tol = 2 * torch.finfo(dtype).eps * accum + 1e-3
    err = (out.double() - ref).abs()
    assert bool((err <= tol).all()), f"max excess {(err - tol).max().item():.3e}"


@gfx11_only
@pytest.mark.parametrize("M", [1, 4, 8])
def test_w4a8_gemv_close_to_bf16_dequant(M):
    """The int8 activation rounding stays a small relative error."""
    torch.manual_seed(0)
    N, K = 256, 4096
    w_q, w_s = _random_mxfp4(N, K)
    x = (torch.randn(M, K, device="cuda") * 0.1).to(torch.bfloat16)

    out = torch.ops._rocm_C.mxfp4_w4a8_gemv(x, w_q, w_s).double()
    ref = x.double() @ _dequant_ref(w_q, w_s).t()
    rel = (out - ref).norm() / ref.norm()
    assert rel < 1e-2, f"relative error {rel.item():.3e}"


@gfx11_only
@pytest.mark.parametrize("M", [0, MAX_W4A8_BATCH_SIZE + 1])
def test_w4a8_gemv_rejects_out_of_range_m(M):
    w_q, w_s = _random_mxfp4(64, 256)
    x = torch.randn(M, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="1 <= M <= 8"):
        torch.ops._rocm_C.mxfp4_w4a8_gemv(x, w_q, w_s)


@gfx11_only
def test_w4a8_gemv_opcheck():
    w_q, w_s = _random_mxfp4(64, 256)
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    torch.library.opcheck(torch.ops._rocm_C.mxfp4_w4a8_gemv, (x, w_q, w_s))


@gfx11_only
def test_w4a8_gemv_cuda_graph():
    w_q, w_s = _random_mxfp4(256, 512)
    x = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
    eager = torch.ops._rocm_C.mxfp4_w4a8_gemv(x, w_q, w_s)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops._rocm_C.mxfp4_w4a8_gemv(x, w_q, w_s)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = torch.ops._rocm_C.mxfp4_w4a8_gemv(x, w_q, w_s)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


def _make_layer(N: int, K: int):
    w_q, w_s = _random_mxfp4(N, K)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_q, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(w_s, requires_grad=False)
    return layer


@gfx11_only
@pytest.mark.parametrize("activation_quant_key", [None, kMxfp4Dynamic])
@pytest.mark.parametrize("M", [1, 8, 9, 64])
def test_linear_kernel_dispatch(activation_quant_key, M, monkeypatch):
    """M <= 8 takes the W4A8 GEMV, larger M the weight-only dequant + GEMM
    fallback, for both W4A16 and W4A4 checkpoints."""
    pytest.importorskip("quark")
    monkeypatch.setattr(envs, "VLLM_ROCM_MXFP4_W4A8", True)
    torch.manual_seed(0)
    N, K = 256, 512
    kernel = RdnaW4A8MxFp4LinearKernel(
        MxFp4LinearLayerConfig(activation_quant_key=activation_quant_key)
    )
    layer = _make_layer(N, K)
    kernel.process_weights_after_loading(layer)
    x = (torch.randn(M, K, device="cuda") * 0.1).to(torch.bfloat16)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)

    with patch(
        "vllm._custom_ops.mxfp4_w4a8_gemv", wraps=torch.ops._rocm_C.mxfp4_w4a8_gemv
    ) as gemv:
        out = kernel.apply_weights(layer, x, bias)
    uses_w4a8 = M <= MAX_W4A8_BATCH_SIZE
    assert gemv.called == uses_w4a8

    ref = x.double() @ _dequant_ref(layer.weight, layer.weight_scale).t()
    ref = ref + bias.double()
    rel = (out.double() - ref).norm() / ref.norm()
    assert rel < 2e-2, f"relative error {rel.item():.3e}"


@gfx11_only
def test_init_selects_w4a8_only_when_enabled(monkeypatch):
    pytest.importorskip("quark")
    monkeypatch.setattr(envs, "VLLM_ROCM_MXFP4_W4A8", True)
    kernel = init_mxfp4_linear_kernel(activation_quant_key=kMxfp4Dynamic)
    assert isinstance(kernel, RdnaW4A8MxFp4LinearKernel)

    monkeypatch.setattr(envs, "VLLM_ROCM_MXFP4_W4A8", False)
    kernel = init_mxfp4_linear_kernel(activation_quant_key=kMxfp4Dynamic)
    assert not isinstance(kernel, RdnaW4A8MxFp4LinearKernel)


@pytest.mark.cpu_test
def test_can_implement_rejects_non_mxfp4_activation():
    config = MxFp4LinearLayerConfig(activation_quant_key=kMxfp6E3M2Dynamic)
    can_implement, reason = RdnaW4A8MxFp4LinearKernel.can_implement(config)
    assert not can_implement
    assert reason


@pytest.mark.cpu_test
def test_is_supported_requires_rocm_gfx11():
    with patch(
        "vllm.model_executor.kernels.linear.mxfp4.rdna_w4a8.current_platform.is_rocm",
        return_value=False,
    ):
        is_supported, reason = RdnaW4A8MxFp4LinearKernel.is_supported()
    assert not is_supported
    assert reason
