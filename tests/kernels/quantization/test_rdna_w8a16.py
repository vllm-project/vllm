# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the RDNA W8A16 op (HIP skinny + Triton).

Run `pytest tests/kernels/quantization/test_rdna_w8a16.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

pytest.importorskip("triton")

from vllm.model_executor.kernels.linear import (  # noqa: E402
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision import (  # noqa: E402
    rdna_w8a16 as w8a16,
)
from vllm.model_executor.layers.quantization.online.int8 import (  # noqa: E402
    quantize_int8_per_group_sym,
    quantize_int8_to_mp_layout,
)
from vllm.model_executor.utils import set_weight_attrs  # noqa: E402
from vllm.platforms.rocm import on_gfx1x  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402

MAX_SKINNY_BATCH_SIZE = w8a16.MAX_SKINNY_BATCH_SIZE
LDS_CAPACITY_ELEMENTS = w8a16.LDS_CAPACITY_ELEMENTS

requires_gfx1x = pytest.mark.skipif(
    not on_gfx1x(), reason="RDNA W8A16 is gfx11/gfx12 only"
)
device = "cuda"


def _quantize(
    n: int, k: int, dtype: torch.dtype, row_pad: int = 0, group_size: int = -1
):
    """int8 codes [N, K] and scales [N], or [N, K // group_size] if grouped."""
    w = (torch.randn(n, k, device=device) * 0.02).to(dtype)
    q, scale = quantize_int8_per_group_sym(w, group_size)
    if row_pad:
        padded = torch.zeros(n, k + row_pad, dtype=torch.int8, device=device)
        padded[:, :k] = q
        q = padded[:, :k]
    return q, scale.squeeze(1) if group_size == -1 else scale


def _reference(x, q, scale, bias):
    groups = scale.view(q.shape[0], -1).float()
    weight = q.float().unflatten(-1, (groups.shape[1], -1)) * groups[..., None]
    out = x.float() @ weight.flatten(-2).t()
    if bias is not None:
        out += bias.float()
    return out


def _assert_close(out, ref, dtype):
    # The only rounding beyond the reference is the output cast.
    rtol = 2e-2 if dtype == torch.bfloat16 else 3e-3
    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < rtol, f"relative error {err:.3e} >= {rtol}"


@requires_gfx1x
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M", [1, MAX_SKINNY_BATCH_SIZE, MAX_SKINNY_BATCH_SIZE + 1, 17, 64, 257]
)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize(
    "N,K,row_pad",
    [(1000, 2560, 0), (4099, 2016, 0), (8192, 4096, 128)],
    ids=["ragged_n", "ragged_k", "padded_rows"],
)
@pytest.mark.parametrize("group_size", [-1, 32], ids=["per_channel", "g32"])
def test_rdna_w8a16_apply_matches_reference(
    dtype, M, has_bias, N, K, row_pad, group_size
):
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(0)
    q, scale = _quantize(N, K, dtype, row_pad, group_size)
    x = torch.randn(M, K, dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device) if has_bias else None

    out = torch.ops.vllm.rdna_w8a16_apply(x, q, scale, bias, num_compute_units())

    assert out.shape == (M, N) and out.dtype == dtype
    _assert_close(out, _reference(x, q, scale, bias), dtype)


@requires_gfx1x
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", range(1, MAX_SKINNY_BATCH_SIZE + 1))
@pytest.mark.parametrize("N,K", [(151936, 2560), (4096, 4096)])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("group_size", [-1, 32], ids=["per_channel", "g32"])
def test_wvsplitk_int8_matches_reference(dtype, M, N, K, has_bias, group_size):
    """Cover the HIP kernel at every batch size it supports, without dispatch."""
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    if M * K > LDS_CAPACITY_ELEMENTS:
        pytest.skip("activations do not fit in LDS")
    set_random_seed(0)
    q, scale = _quantize(N, K, dtype, group_size=group_size)
    x = torch.randn(M, K, dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device) if has_bias else None

    out = ops.wvSplitK_int8(q, x, scale, num_compute_units(), bias)

    assert out.shape == (M, N) and out.dtype == dtype
    ref = _reference(x, q, scale, bias)
    tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    assert ((out.float() - ref).norm() / ref.norm()).item() < tol


def _bad_wvsplitk_int8_inputs():
    K = 4096
    q = torch.zeros(256, K, dtype=torch.int8, device=device)
    scale = torch.ones(256, dtype=torch.float16, device=device)

    def x(m, k=K, dtype=torch.float16):
        return torch.randn(m, k, dtype=dtype, device=device)

    return {
        "batch_above_max": (q, x(MAX_SKINNY_BATCH_SIZE + 1), scale),
        "k_mismatch": (q, x(2, K // 2), scale),
        "non_contiguous_x": (q, x(K).t()[:2], scale),
        "scale_dtype": (q, x(2), scale.to(torch.bfloat16)),
        "scale_shape": (q, x(2), scale[:-1]),
        "unsupported_group_size": (q, x(2), scale[:, None]),
        "fp16_weight": (q.half(), x(2), scale),
        "over_lds": (
            torch.zeros(256, 8192, dtype=torch.int8, device=device),
            x(MAX_SKINNY_BATCH_SIZE, 8192),
            scale,
        ),
        "bias_shape": (q, x(2), scale, scale[:-1]),
        "bias_2d": (q, x(2), scale, scale.repeat(2, 1)),
    }


@requires_gfx1x
@pytest.mark.parametrize(
    "case",
    [
        "batch_above_max",
        "k_mismatch",
        "non_contiguous_x",
        "scale_dtype",
        "scale_shape",
        "unsupported_group_size",
        "fp16_weight",
        "over_lds",
        "bias_shape",
        "bias_2d",
    ],
)
def test_wvsplitk_int8_rejects_bad_inputs(case):
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    w, x, scale, *bias = _bad_wvsplitk_int8_inputs()[case]
    with pytest.raises(RuntimeError):
        ops.wvSplitK_int8(w, x, scale, num_compute_units(), *bias)
        torch.accelerator.synchronize()


@requires_gfx1x
@pytest.mark.parametrize("M", [2, 64])
@pytest.mark.parametrize("group_size", [-1, 32], ids=["per_channel", "g32"])
def test_triton_w8a16_matches_reference(M, group_size):
    """Cover the Triton kernel at batch sizes the op routes to HIP."""
    set_random_seed(0)
    q, scale = _quantize(2048, 1024, torch.bfloat16, group_size=group_size)
    x = torch.randn(M, 1024, dtype=torch.bfloat16, device=device)

    out = w8a16.triton_w8a16_gemm(x, q, scale)

    _assert_close(out, _reference(x, q, scale, None), torch.bfloat16)


@requires_gfx1x
def test_triton_w8a16_weight_offsets_past_int32():
    """A [vocab, hidden] head can exceed 2^31 bytes; addressing must not wrap."""
    K = 4096
    N = 2**31 // K + 64
    if torch.accelerator.get_memory_info()[0] < 3 * N * K:
        pytest.skip("not enough free device memory")
    set_random_seed(0)
    q = torch.zeros(N, K, dtype=torch.int8, device=device)
    q[-64:] = torch.randint(-127, 128, (64, K), dtype=torch.int8, device=device)
    scale = torch.full((N,), 1e-2, dtype=torch.float16, device=device)
    x = torch.randn(MAX_SKINNY_BATCH_SIZE + 1, K, dtype=torch.float16, device=device)

    out = w8a16.triton_w8a16_gemm(x, q, scale)

    _assert_close(
        out[:, -64:], _reference(x, q[-64:], scale[-64:], None), torch.float16
    )
    assert out[:, :-64].abs().max() == 0


def _mp_layer(packed: torch.Tensor, scale: torch.Tensor, gptq_layout: bool):
    """A layer holding an `MPLinearKernel` uint8b128 weight as checkpoints load
    it: [N, K // 4] (compressed-tensors) or [K // 4, N] (GPTQ)."""
    dims = {"input_dim": 1, "output_dim": 0}
    if gptq_layout:
        packed, scale = packed.t().contiguous(), scale.t().contiguous()
        dims = {"input_dim": 0, "output_dim": 1}
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(packed, requires_grad=False)
    set_weight_attrs(layer.weight, {**dims, "packed_dim": dims["input_dim"]})
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    set_weight_attrs(layer.weight_scale, dims)
    return layer


@requires_gfx1x
@pytest.mark.parametrize("K", [2560, 4096])
@pytest.mark.parametrize("M", [1, MAX_SKINNY_BATCH_SIZE + 1])
@pytest.mark.parametrize("group_size", [-1, 32], ids=["per_channel", "g32"])
@pytest.mark.parametrize("gptq_layout", [False, True], ids=["ct", "gptq"])
def test_rdna_w8a16_kernel_repacks_mp_layout(K, M, group_size, gptq_layout):
    """The kernel is selected for uint8b128, and both paths read its repacked
    load-time layout, padded or not."""
    from vllm.platforms.rocm import on_gfx1151

    set_random_seed(0)
    N = 512
    weight = (torch.randn(N, K, device=device) * 0.02).to(torch.bfloat16)
    q, scale = quantize_int8_per_group_sym(weight, group_size)
    packed, packed_scale, _ = quantize_int8_to_mp_layout(weight, group_size)
    layer = _mp_layer(packed, packed_scale, gptq_layout)
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint8b128,
        act_type=torch.bfloat16,
        group_size=group_size,
        zero_points=False,
    )
    kernel_cls = choose_mp_linear_kernel(config)
    assert kernel_cls is w8a16.RDNAW8A16LinearKernel
    kernel = kernel_cls(config, w_q_param_name="weight", w_s_param_name="weight_scale")

    kernel.process_weights_after_loading(layer)

    assert torch.equal(layer.weight, q)
    expect_pad = on_gfx1151() and K % 1024 == 0
    assert (layer.weight.stride(0) > K) == expect_pad
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    out = kernel.apply_weights(layer, x)
    _assert_close(out, _reference(x, q, scale, None), torch.bfloat16)


@pytest.mark.parametrize(
    "weight_type,group_size,zero_points,N,expected_ok",
    [
        (scalar_types.uint8b128, -1, False, 256, True),
        (scalar_types.uint8b128, 32, False, 256, True),
        (scalar_types.uint8b128, 128, False, 256, False),
        (scalar_types.uint8b128, 32, False, 254, False),
        (scalar_types.uint8b128, 32, True, 256, False),
        (scalar_types.uint4b8, 32, False, 256, False),
    ],
    ids=["per_channel", "g32", "g128", "bad_n", "zero_points", "wrong_qtype"],
)
def test_rdna_w8a16_can_implement(
    weight_type, group_size, zero_points, N, expected_ok, monkeypatch
):
    monkeypatch.setattr(w8a16, "_on_gfx1x", lambda: True)
    config = MPLinearLayerConfig(
        full_weight_shape=(1024, N),
        partition_weight_shape=(1024, N),
        weight_type=weight_type,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=zero_points,
    )
    ok, _ = w8a16.RDNAW8A16LinearKernel.can_implement(config)
    assert ok is expected_ok


@pytest.mark.parametrize(
    "M,K,expected_path",
    [
        (
            MAX_SKINNY_BATCH_SIZE,
            LDS_CAPACITY_ELEMENTS // MAX_SKINNY_BATCH_SIZE // 16 * 16,
            "hip",
        ),
        (MAX_SKINNY_BATCH_SIZE + 1, 1024, "triton"),
        (4, 8192, "hip"),
        (4, 8192 + 16, "triton"),
        (1, 2056, "triton"),
    ],
    ids=["max_batch", "batch_too_large", "lds_limit", "above_lds", "ragged_k"],
)
def test_rdna_w8a16_dispatch_boundaries(M, K, expected_path, monkeypatch):
    import vllm._custom_ops as ops

    N = 16
    x = torch.empty((M, K), dtype=torch.float16)
    w_q = torch.empty((N, K), dtype=torch.int8)
    scale = torch.empty((N,), dtype=torch.float16)
    called = []

    def fake(path):
        def run(*args, **kwargs):
            called.append(path)
            return torch.empty((M, N), dtype=x.dtype)

        return run

    monkeypatch.setattr(ops, "wvSplitK_int8", fake("hip"), raising=False)
    monkeypatch.setattr(w8a16, "triton_w8a16_gemm", fake("triton"))

    out = w8a16._rdna_w8a16_apply_impl(x, w_q, scale, None, 40)

    assert out.shape == (M, N)
    assert called == [expected_path]
