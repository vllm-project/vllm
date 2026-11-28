#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AiterHipbW4A16LinearKernel: checkpoint weights through
process_weights_after_loading and apply_weights against the dequantized
reference.

Skipped unless the installed AITER exposes aiter.hipb_w4a16_mm.
"""

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

if not rocm_aiter_ops.has_hipb_w4a16():
    pytest.skip("AITER has no hipb_w4a16_mm", allow_module_level=True)

# Same directory, and pytest's default import mode puts it on sys.path.
from test_rdna_hybrid_w4a16 import (  # noqa: E402
    _build_dummy_layer,
    _pack_int4_along_k_to_ckpt,
    _pack_int4_along_n_for_zp,
    _pack_zp_rows_for_kernel,
    _rdna_hybrid_w4a16_reference,
)

from vllm.model_executor.kernels.linear.mixed_precision.aiter_hipb_w4a16 import (  # noqa: E402
    AiterHipbW4A16LinearKernel,
    _on_gfx115x,
    k_order_bytes,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E402
    MPLinearLayerConfig,
)
from vllm.scalar_type import scalar_types  # noqa: E402

device = "cuda"


@pytest.fixture
def aiter_hipb_w4a16_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR_HIPB_W4A16", "1")
    rocm_aiter_ops.refresh_env_variables()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def _config(K, N, group_size, has_zp, dtype=torch.float16) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4 if has_zp else scalar_types.uint4b8,
        act_type=dtype,
        group_size=group_size,
        zero_points=has_zp,
    )


def _build_layer(K, N, group_size, has_zp, dtype):
    """A checkpoint-layout layer and the values it holds: [N, K] weights,
    [N, K/G] scales and zero-points (None when symmetric)."""
    num_groups = K // group_size
    w_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_nkg = (
        0.05 * torch.rand((N, num_groups), device=device, dtype=torch.float32)
    ).to(dtype)
    zp_nkg = None
    zeros_ckpt = None
    if has_zp:
        zp_gn = torch.randint(0, 16, (num_groups, N), device=device, dtype=torch.int32)
        zp_nkg = zp_gn.t().contiguous()
        zeros_ckpt = _pack_int4_along_n_for_zp(zp_gn).t().contiguous()
    layer = _build_dummy_layer(
        _pack_int4_along_k_to_ckpt(w_int4_kn), scales_nkg.clone(), zeros_ckpt
    )
    return layer, w_int4_kn.t().contiguous(), scales_nkg, zp_nkg


def _kernel(config: MPLinearLayerConfig) -> AiterHipbW4A16LinearKernel:
    return AiterHipbW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point" if config.zero_points else None,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", AiterHipbW4A16LinearKernel.SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize("M", [1, 5, 300], ids=["M=1", "M=5", "M=300"])
def test_aiter_hipb_w4a16_matches_reference(
    dtype, group_size, has_zp, M, aiter_hipb_w4a16_enabled, dist_init
):
    """The kernel reproduces (nibble - zp) * scale + bias from checkpoint
    weights, for decode and prefill batches."""
    set_random_seed(0)
    K, N = 1024, 256
    config = _config(K, N, group_size, has_zp, dtype)
    ok, why = AiterHipbW4A16LinearKernel.can_implement(config)
    if not ok:
        pytest.skip(why)
    layer, w_int4_nk, scales_nkg, zp_nkg = _build_layer(K, N, group_size, has_zp, dtype)
    kernel = _kernel(config)
    kernel.process_weights_after_loading(layer)

    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    out = kernel.apply_weights(layer, x_mk, bias)

    ref = _rdna_hybrid_w4a16_reference(
        x_mk, w_int4_nk, scales_nkg, zp_nkg, group_size, bias=bias
    )
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _on_gfx115x(), reason="the weight row pad is gfx115x only")
@pytest.mark.parametrize("has_zp", [False, True])
def test_process_weights_layout(has_zp, dist_init):
    """Weight rows are padded to an odd multiple of 128 B and scales stay
    dense; with zero-points the scale parameter is the GEMM's combined buffer,
    the scales at its head. The scale row here (256 groups x 2 B = 512 B) is
    one RDNAHybrid would pad."""
    set_random_seed(0)
    K, N, G = 8192, 128, 32
    num_groups = K // G
    layer, _, scales_nkg, _ = _build_layer(K, N, G, has_zp, torch.float16)
    _kernel(_config(K, N, G, has_zp)).process_weights_after_loading(layer)

    assert layer.weight_packed.stride(0) == K // 2 + 128
    if has_zp:
        head = layer.weight_scale[: scales_nkg.numel() * 2].view(torch.float16)
        torch.testing.assert_close(head.view(N, num_groups), scales_nkg)
    else:
        assert layer.weight_scale.stride() == (num_groups, 1)


@pytest.mark.parametrize("layout", ["compressed_tensors", "gptq", "packed_along_n"])
def test_k_order_bytes(layout, dist_init):
    """Checkpoints packed along K (compressed-tensors, GPTQ) are reinterpreted
    without unpacking; others are unpacked. Both give hipBLASLt's K-order bytes."""
    from vllm.model_executor.parameter import PackedvLLMParameter

    K, N = 64, 16
    w_nk = torch.randint(0, 16, (N, K), dtype=torch.int32)
    shifts = torch.arange(8, dtype=torch.int32) * 4
    along_k = torch.sum((w_nk.view(N, K // 8, 8) << shifts), dim=2, dtype=torch.int32)
    if layout == "compressed_tensors":
        data, dims = along_k, dict(input_dim=1, output_dim=0, packed_dim=1)
    elif layout == "gptq":
        data, dims = (
            along_k.t().contiguous(),
            dict(input_dim=0, output_dim=1, packed_dim=0),
        )
    else:
        w_kn = w_nk.t().contiguous()
        data = torch.sum((w_kn.view(K, N // 8, 8) << shifts), dim=2, dtype=torch.int32)
        dims = dict(input_dim=0, output_dim=1, packed_dim=1)
    w_q = PackedvLLMParameter(
        data=data, weight_loader=lambda *a, **k: None, packed_factor=8, **dims
    )
    expected = (w_nk[:, 0::2] | (w_nk[:, 1::2] << 4)).to(torch.uint8)
    torch.testing.assert_close(k_order_bytes(w_q, scalar_types.uint4b8), expected)


@pytest.mark.parametrize("group_size", [32, 128])
def test_scale_buffer_zero_point_layout(group_size):
    """The zero-point region matches hipBLASLt's addressing, nibbles untouched.

    Checked against hipBLASLt's own addressing (w4a16_datagen.hpp): (n, g)
    lives at word (n//8)*ceil(K/G) + g, bits 4*(n%8), 256 B after the scales.
    """
    K, N = 512, 32
    num_groups = K // group_size
    scales = torch.zeros((N, num_groups), dtype=torch.float16, device=device)
    zp_nkg = torch.randint(0, 16, (N, num_groups), device=device, dtype=torch.int32)

    buf = rocm_aiter_ops.hipb_w4a16_scale_buffer(
        scales, _pack_zp_rows_for_kernel(zp_nkg)
    )

    scale_bytes = N * num_groups * scales.element_size()
    align = 256  # c_blockScaleAZeroPointAlignment in hipBLASLt's tensile_host.cpp
    zp_region = buf[-(-scale_bytes // align) * align :].cpu()

    n = torch.arange(N).repeat_interleave(num_groups)
    g = torch.arange(num_groups).repeat(N)
    word = zp_region.view(torch.int32)[(n // 8) * num_groups + g]
    got = ((word >> (4 * (n % 8)).to(torch.int32)) & 0xF).to(torch.int32)
    torch.testing.assert_close(got, zp_nkg.cpu().reshape(-1))


@pytest.mark.parametrize(
    "use_aiter,use_linear,use_w4a16,prefill_dequant,why",
    [
        ("0", "1", "1", "disable", "VLLM_ROCM_USE_AITER=1"),
        ("1", "0", "1", "disable", "VLLM_ROCM_USE_AITER_LINEAR=1"),
        ("1", "1", "0", "disable", "VLLM_ROCM_USE_AITER_LINEAR_HIPB_W4A16=1"),
        ("1", "1", "1", "enable", "rdna_hybrid_w4a16_prefill_dequant"),
    ],
)
def test_can_implement_declines(
    use_aiter, use_linear, use_w4a16, prefill_dequant, why, monkeypatch
):
    """The kernel steps aside when AITER, its linear ops or its W4A16 GEMM is
    disabled, and when RDNAHybrid is asked to cache a dense prefill copy."""
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", use_aiter)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", use_linear)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR_HIPB_W4A16", use_w4a16)
    rocm_aiter_ops.refresh_env_variables()
    vllm_config = VllmConfig(
        kernel_config=KernelConfig(rdna_hybrid_w4a16_prefill_dequant=prefill_dequant)
    )
    try:
        with set_current_vllm_config(vllm_config):
            ok, reason = AiterHipbW4A16LinearKernel.can_implement(
                _config(1024, 256, 128, False)
            )
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()
    assert not ok
    assert reason is not None and why in reason


def test_can_implement_declines_when_aiter_raises(
    aiter_hipb_w4a16_enabled, monkeypatch
):
    """A failing AITER support query (e.g. its JIT build) declines the layer
    instead of failing the model load."""

    def broken(*args, **kwargs):
        raise RuntimeError("JIT build failed")

    monkeypatch.setattr(rocm_aiter_ops, "hipb_w4a16_is_supported", broken)
    ok, reason = AiterHipbW4A16LinearKernel.can_implement(
        _config(1024, 256, 128, False)
    )
    assert not ok
    assert reason is not None and "JIT build failed" in reason


def test_linear_backend_aiter_selects_kernel(monkeypatch):
    """`--linear-backend aiter` picks this kernel without the AITER env flags."""
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear import choose_mp_linear_kernel

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    rocm_aiter_ops.refresh_env_variables()
    config = _config(1024, 256, 128, False)
    try:
        with set_current_vllm_config(
            VllmConfig(kernel_config=KernelConfig(linear_backend="aiter"))
        ):
            ok, why = AiterHipbW4A16LinearKernel.can_implement(config)
            if not ok:
                pytest.skip(why)
            kernel = choose_mp_linear_kernel(config)
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()
    assert kernel is AiterHipbW4A16LinearKernel
