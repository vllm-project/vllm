#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCm Hybrid W4A16 kernel (HIP skinny + Triton prefill).

Run `pytest tests/kernels/quantization/test_rdna_hybrid_w4a16.py`.
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

pytest.importorskip("triton")

from vllm.platforms.rocm import on_gfx1x  # noqa: E402

device = "cuda"

hybrid_module = importlib.import_module(
    "vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16"
)
RDNAHybridW4A16LinearKernel = hybrid_module.RDNAHybridW4A16LinearKernel
pack_int4_exllama_shuffle = hybrid_module.pack_int4_exllama_shuffle
SUPPORTED_GROUP_SIZES = hybrid_module.SUPPORTED_GROUP_SIZES
MAX_SKINNY_BATCH_SIZE = hybrid_module.MAX_SKINNY_BATCH_SIZE
LDS_CAPACITY_ELEMENTS = hybrid_module.LDS_CAPACITY_ELEMENTS
MEDIUM_SKINNY_LIMIT_ELEMENTS = hybrid_module.MEDIUM_SKINNY_LIMIT_ELEMENTS
triton_w4a16_skinny_fmt_gemm = hybrid_module.triton_w4a16_skinny_fmt_gemm
select_skinny_gfx1151_config = hybrid_module._select_skinny_gfx1151_config


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _pack_zp_rows_for_kernel(zp_nkg: torch.Tensor) -> torch.Tensor:
    """Pack raw uint4 zero points along N: [N, G] int32 -> [N//8, G] int32.

    Row n's nibble lands in word[n//8] at bits 4*(n%8).
    """
    assert zp_nkg.dtype == torch.int32
    N, G = zp_nkg.shape
    assert N % 8 == 0
    shifts = (torch.arange(8, device=zp_nkg.device, dtype=torch.int32) * 4)[:, None]
    return torch.sum(
        (zp_nkg.view(N // 8, 8, G) & 0xF) << shifts, dim=1, dtype=torch.int32
    ).contiguous()


def _rdna_hybrid_w4a16_reference(
    x_mk: torch.Tensor,
    w_int4_nk: torch.Tensor,
    scales_nkg: torch.Tensor,
    zp_nkg: torch.Tensor | None,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Reference for the Hybrid W4A16 op.

    x_mk: [M, K] fp16/bf16
    w_int4_nk: [N, K] int32 with raw uint4 values in [0, 15]
    scales_nkg: [N, K//G] fp16/bf16
    zp_nkg: [N, K//G] int32 raw zero points in [0, 15], or None for
            symmetric (uint4b8, dequant subtracts 8)
    """
    G = group_size
    N, K = w_int4_nk.shape
    assert K % G == 0
    s_full = scales_nkg.repeat_interleave(G, dim=1).to(torch.float32)  # [N, K]
    if zp_nkg is None:
        z_full = torch.full((N, K), 8.0, device=x_mk.device, dtype=torch.float32)
    else:
        z_full = zp_nkg.repeat_interleave(G, dim=1).to(torch.float32)
    w_fp = (w_int4_nk.to(torch.float32) - z_full) * s_full  # [N, K]
    out = x_mk.to(torch.float32) @ w_fp.t()  # [M, N]
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(x_mk.dtype)


# ---------------------------------------------------------------------------
# Forward correctness: decode (HIP skinny) + prefill (Triton)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize(
    "M",
    [1, MAX_SKINNY_BATCH_SIZE, MAX_SKINNY_BATCH_SIZE + 1, 64],
    ids=["M=1_decode", "M=5_decode", "M=6_prefill", "M=64_prefill"],
)
def test_rdna_hybrid_w4a16_apply_matches_reference(dtype, group_size, has_zp, M):
    """Smoke test the registered custom op for both decode and prefill batches.

    Verifies the dispatch logic in `_rdna_hybrid_w4a16_apply_impl`:
      - supported M and K*M: HIP wvSplitK_int4_g
      - otherwise: Triton prefill kernel
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    set_random_seed(0)

    K, N = 1024, 256
    assert K % group_size == 0 and K % 8 == 0 and N % 8 == 0

    # Activations.
    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)

    # Weights as raw uint4 in [N, K], packed to ExLlama shuffle [N, K//8].
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    w_q_i32 = pack_int4_exllama_shuffle(w_int4_nk).contiguous()  # [N, K//8] int32
    w_q = w_q_i32.view(torch.int8)  # same bytes viewed as int8 [N, K//2]

    # Scales [N, K//G] in act dtype.
    scales_nkg = (
        0.05 * torch.rand((N, K // group_size), device=device, dtype=torch.float32)
    ).to(dtype)

    # Optional raw zero points, [N, K//G] int32 for the reference and packed
    # along N for the op.
    if has_zp:
        zp_nkg = torch.randint(
            0, 16, (N, K // group_size), device=device, dtype=torch.int32
        )
        w_zp = _pack_zp_rows_for_kernel(zp_nkg)
    else:
        zp_nkg = None
        w_zp = None

    from vllm.utils.platform_utils import num_compute_units

    out = torch.ops.vllm.rdna_hybrid_w4a16_apply(
        x_mk,
        w_q,
        scales_nkg,
        w_zp,
        None,  # bias
        num_compute_units(),
        group_size,
    )

    ref = _rdna_hybrid_w4a16_reference(
        x_mk, w_int4_nk, scales_nkg, zp_nkg, group_size, bias=None
    )

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1, MAX_SKINNY_BATCH_SIZE + 1])
def test_rdna_hybrid_w4a16_apply_with_bias(dtype, M):
    """Bias is added correctly on both decode and prefill paths."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    set_random_seed(0)
    K, N, G = 1024, 128, 128

    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    w_q_i32 = pack_int4_exllama_shuffle(w_int4_nk).contiguous()
    w_q = w_q_i32.view(torch.int8)
    scales_nkg = (
        0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)
    ).to(dtype)
    bias = torch.randn(N, device=device, dtype=dtype) * 0.1

    from vllm.utils.platform_utils import num_compute_units

    out = torch.ops.vllm.rdna_hybrid_w4a16_apply(
        x_mk,
        w_q,
        scales_nkg,
        None,
        bias,
        num_compute_units(),
        G,
    )
    ref = _rdna_hybrid_w4a16_reference(x_mk, w_int4_nk, scales_nkg, None, G, bias=bias)

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# Triton prefill path
# ---------------------------------------------------------------------------


def _make_prefill_case(M, K, N, G, dtype, has_zp):
    """Random [M,K] activations + skinny [N,K//8] weights and their metadata.

    Zero points come back twice: ``zp_raw`` as [N, K//G] int32 nibbles for the
    float32 oracle, and ``zp_packed`` as [N//8, K//G] int32 in the packed layout
    the kernel reads.
    """
    x = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4 = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    b_q = pack_int4_exllama_shuffle(w_int4)
    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )
    if has_zp:
        zp_raw = torch.randint(0, 16, (N, K // G), device=device, dtype=torch.int32)
        zp_packed = _pack_zp_rows_for_kernel(zp_raw)
    else:
        zp_raw = zp_packed = None
    return x, w_int4, b_q, scales, zp_raw, zp_packed


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        (17, 256, 512, 32),
        (32, 512, 256, 64),
        (33, 512, 512, 128),
        (64, 1024, 256, 128),
        (129, 256, 512, 32),
        (256, 512, 512, 128),
        (257, 512, 512, 128),
    ],
)
def test_triton_prefill_gemm_matches_reference(dtype, has_zp, M, K, N, G):
    """The Triton prefill GEMM matches a float32 dequantize-then-matmul reference.

    Covers fp16 and bf16, with and without zero points, at M values on both
    sides of where gfx1151 switches fp16 to the packed dequant.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    set_random_seed(0)

    x, w_int4, b_q, scales, zp_raw, zp_packed = _make_prefill_case(
        M, K, N, G, dtype, has_zp
    )
    out = triton_w4a16_skinny_fmt_gemm(
        a=x, b_q=b_q, scales=scales, group_size=G, zp=zp_packed
    )
    ref = _rdna_hybrid_w4a16_reference(x, w_int4, scales, zp_raw, G, bias=None)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gfx1151_tile_table_never_straddles_a_quant_group(dtype):
    """Every gfx1151 tile config keeps each K tile inside one quant group.

    The kernel loads one scale per K tile, so a BLOCK_K wider than group_size
    would apply the wrong scale to the rest of the tile.
    """
    for group_size in SUPPORTED_GROUP_SIZES:
        for M in (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            for N, K in [
                (512, 2048),
                (4096, 4096),
                (24576, 4096),
                (4096, 12288),
                (32768, 2048),
                (1024, 8192),
            ]:
                _, _, block_k, _, _, _ = select_skinny_gfx1151_config(
                    M, N, K, group_size, dtype
                )
                assert block_k <= group_size, (
                    f"BLOCK_K={block_k} > group_size={group_size} "
                    f"at M={M} N={N} K={K} dtype={dtype}"
                )
                assert block_k % 8 == 0, (
                    f"BLOCK_K={block_k} must be a multiple of 8 "
                    f"(8 nibbles per packed int32)"
                )


# ---------------------------------------------------------------------------
# pack_int4_exllama_shuffle round-trips correctly
# ---------------------------------------------------------------------------


def test_pack_int4_exllama_shuffle_layout():
    """Pack 8 K-values per int32 in interleave [0,2,4,6,1,3,5,7] order."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    set_random_seed(0)
    N, K = 4, 16
    w = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    packed = pack_int4_exllama_shuffle(w)
    assert packed.shape == (N, K // 8) and packed.dtype == torch.int32

    # Manual unshuffle using ExLlama shifts [0,16,4,20,8,24,12,28].
    shifts = torch.tensor(
        [0, 16, 4, 20, 8, 24, 12, 28], device=device, dtype=torch.int32
    )
    unshuffled = (packed.unsqueeze(-1) >> shifts) & 0xF
    unshuffled = unshuffled.reshape(N, K)
    torch.testing.assert_close(unshuffled, w)


# ---------------------------------------------------------------------------
# process_weights_after_loading: layout repack and zp normalization
# ---------------------------------------------------------------------------


def _pack_int4_along_k_to_ckpt(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """Pack int4 values along K into CT checkpoint layout: [K,N] -> [N, K//8]."""
    assert w_int4_kn.dtype == torch.int32
    K, N = w_int4_kn.shape
    assert K % 8 == 0
    out = torch.zeros((N, K // 8), dtype=torch.int32, device=w_int4_kn.device)
    for i in range(8):
        out |= (w_int4_kn[i::8, :].t() & 0xF) << (i * 4)
    return out.contiguous()


def _pack_int4_along_n_for_zp(zp_int4_gn: torch.Tensor) -> torch.Tensor:
    """Pack int4 zero points along N: [G, N] -> [G, N//8] int32 (CT layout)."""
    assert zp_int4_gn.dtype == torch.int32
    G, N = zp_int4_gn.shape
    assert N % 8 == 0
    shifts = torch.arange(8, device=zp_int4_gn.device, dtype=torch.int32) * 4
    return torch.sum(
        (zp_int4_gn.view(G, N // 8, 8) & 0xF) << shifts, dim=2, dtype=torch.int32
    ).contiguous()


def _build_dummy_layer(
    w_ckpt_nk8: torch.Tensor,
    scales_ckpt_nkg: torch.Tensor,
    zeros_ckpt: torch.Tensor | None,
):
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedColumnParameter,
        PackedvLLMParameter,
    )

    weight_loader = lambda *args, **kwargs: None

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter(
        "weight_packed",
        PackedvLLMParameter(
            data=w_ckpt_nk8,
            weight_loader=weight_loader,
            input_dim=1,
            output_dim=0,
            packed_factor=8,
            packed_dim=1,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        GroupQuantScaleParameter(
            data=scales_ckpt_nkg,
            weight_loader=weight_loader,
            input_dim=1,
            output_dim=0,
        ),
    )
    if zeros_ckpt is not None:
        layer.register_parameter(
            "weight_zero_point",
            PackedColumnParameter(
                data=zeros_ckpt,
                weight_loader=weight_loader,
                output_dim=0,
                packed_factor=8,
                packed_dim=0,
            ),
        )
    return layer


@pytest.mark.parametrize("group_size", SUPPORTED_GROUP_SIZES)
def test_rdna_hybrid_w4a16_process_weights_symmetric_repack(group_size, dist_init):
    """uint4b8 (symmetric): w_q -> [N, K//8] int8 ExLlama shuffle, no zp param."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    set_random_seed(0)

    K, N = 256, 128
    G = group_size
    assert K % G == 0

    # Reference unpacked weights, then pack into CT checkpoint layout [N, K//8].
    w_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    w_ckpt_nk8 = _pack_int4_along_k_to_ckpt(w_int4_kn)
    scales_ckpt_nkg = 0.05 * torch.rand((N, K // G), device=device, dtype=torch.float16)

    layer = _build_dummy_layer(w_ckpt_nk8, scales_ckpt_nkg, zeros_ckpt=None)

    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=G,
        zero_points=False,
    )
    kernel = RDNAHybridW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name=None,
    )
    kernel.process_weights_after_loading(layer)

    # Skinny weight is stored once as int8 [N, K//2]; the Triton path
    # reinterprets it as int32 [N, K//8] via a view (no separate parameter).
    assert layer.weight_packed.dtype == torch.int8
    assert tuple(layer.weight_packed.shape) == (N, K // 2)
    w_q_i32 = layer.weight_packed.view(torch.int32)
    assert tuple(w_q_i32.shape) == (N, K // 8)

    expected_packed = pack_int4_exllama_shuffle(w_int4_kn.t().contiguous())
    torch.testing.assert_close(w_q_i32, expected_packed)

    # Scales: [N, K//G] (skinny layout, no transpose since CT already had it).
    assert tuple(layer.weight_scale.shape) == (N, K // G)
    torch.testing.assert_close(layer.weight_scale, scales_ckpt_nkg)


@pytest.mark.parametrize("group_size", SUPPORTED_GROUP_SIZES)
def test_rdna_hybrid_w4a16_process_weights_asymmetric_repack(group_size, dist_init):
    """uint4 (asymmetric): zero points are packed [N//8, K//G] int32."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    set_random_seed(0)

    K, N = 256, 128
    G = group_size
    assert K % G == 0 and N % 8 == 0

    w_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    w_ckpt_nk8 = _pack_int4_along_k_to_ckpt(w_int4_kn)
    scales_ckpt_nkg = 0.05 * torch.rand((N, K // G), device=device, dtype=torch.float16)

    # CT zero-point layout is N-packed: [N//8, K//G] int32.
    zeros_int4_gn = torch.randint(0, 16, (K // G, N), device=device, dtype=torch.int32)
    zeros_packed_gn8 = _pack_int4_along_n_for_zp(zeros_int4_gn)  # [K//G, N//8]
    zeros_ckpt_n8kg = zeros_packed_gn8.t().contiguous()  # [N//8, K//G]

    layer = _build_dummy_layer(w_ckpt_nk8, scales_ckpt_nkg, zeros_ckpt=zeros_ckpt_n8kg)

    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4,
        act_type=torch.float16,
        group_size=G,
        zero_points=True,
    )
    kernel = RDNAHybridW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point",
    )
    kernel.process_weights_after_loading(layer)

    # Zero-points: [N//8, K//G] int32, row n's nibble at word[n//8]
    # bits 4*(n%8).
    assert layer.weight_zero_point.dtype == torch.int32
    assert tuple(layer.weight_zero_point.shape) == (N // 8, K // G)
    expected_zp = _pack_zp_rows_for_kernel(zeros_int4_gn.t().contiguous())
    torch.testing.assert_close(layer.weight_zero_point, expected_zp)

    # Quantized weights match symmetric path's layout regardless of zp.
    w_q_i32 = layer.weight_packed.view(torch.int32)
    assert tuple(w_q_i32.shape) == (N, K // 8)
    expected_packed = pack_int4_exllama_shuffle(w_int4_kn.t().contiguous())
    torch.testing.assert_close(w_q_i32, expected_packed)


# ---------------------------------------------------------------------------
# can_implement enforces the supported-group-size policy
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize(
    "group_size,expected_ok", [(32, True), (64, True), (128, True), (256, False)]
)
def test_hybrid_can_implement_group_size(group_size, expected_ok):
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    K, N = 1024, 256
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=False,
    )
    ok, _ = RDNAHybridW4A16LinearKernel.can_implement(config)
    assert ok is expected_ok


# ---------------------------------------------------------------------------
# Tests for the HIP wvSplitK_int4_g decode kernel
# ---------------------------------------------------------------------------


def _hip_skinny_reference(
    a_mk: torch.Tensor,
    w_int4_nk: torch.Tensor,
    scales_nkg: torch.Tensor,
    *,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    """Reference for symmetric HIP skinny: C = A @ (W - zp_bias) * S."""
    K = a_mk.shape[1]
    N = w_int4_nk.shape[0]
    num_groups = K // group_size

    w_fp = (w_int4_nk.to(torch.float32) - zp_bias).view(N, num_groups, group_size)
    s = scales_nkg.to(torch.float32).unsqueeze(-1)
    w_dequant = (w_fp * s).view(N, K)

    return (a_mk.to(torch.float32) @ w_dequant.t()).to(a_mk.dtype)


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        (1, 256, 256, 32),
        (1, 256, 256, 64),
        (1, 512, 256, 128),
        (2, 512, 256, 64),
        (3, 256, 512, 64),
    ],
)
def test_hip_skinny_wvSplitK_int4_g(dtype, M, K, N, G):
    """Test HIP wvSplitK_int4_g kernel directly via _custom_ops."""
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(0)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)

    b_packed_i32 = pack_int4_exllama_shuffle(w_int4_nk)
    b_packed_i8 = b_packed_i32.view(torch.int8)

    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )

    cu_count = num_compute_units()
    out = ops.wvSplitK_int4_g(b_packed_i8, a, scales, cu_count, G)

    ref = _hip_skinny_reference(a, w_int4_nk, scales, group_size=G, zp_bias=8)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Tests for the full hybrid dispatch (HIP decode + Triton prefill)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        (1, 256, 256, 64),
        (1, 512, 256, 32),
        (1, 512, 256, 128),
        (32, 512, 256, 64),
        (64, 1024, 256, 128),
    ],
)
def test_rdna_hybrid_w4a16_dispatch(dtype, M, K, N, G):
    """Test the full hybrid dispatch via the custom op."""
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(0)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)

    b_packed_i32 = pack_int4_exllama_shuffle(w_int4_nk)
    b_packed_i8 = b_packed_i32.view(torch.int8)

    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )

    cu_count = num_compute_units()
    out = torch.ops.vllm.rdna_hybrid_w4a16_apply(
        a, b_packed_i8, scales, None, None, cu_count, G
    )

    ref = _hip_skinny_reference(a, w_int4_nk, scales, group_size=G, zp_bias=8)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.skipif(
    not hasattr(torch.ops, "_rocm_C")
    or not hasattr(torch.ops._rocm_C, "wvSplitK_int4_g"),
    reason="wvSplitK_int4_g not built",
)
def test_wvsplitk_int4_g_rejects_unpacked_zero_points():
    """Act-dtype zero points raise instead of being read as packed words.

    Both layouts are 2D with compatible extents, so only the dtype separates
    them; misreading one as the other returns wrong numbers silently.
    """
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    K, N, G, M = 256, 64, 128, 1
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    w = torch.randint(0, 255, (N, K // 2), device=device, dtype=torch.uint8).view(
        torch.int8
    )
    scales = torch.rand((N, K // G), device=device, dtype=torch.float16)
    zp_unpacked = torch.zeros((N, K // G), device=device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="Zero points must be int32 or uint32"):
        ops.wvSplitK_int4_g(w, a, scales, num_compute_units(), G, zp_unpacked, None)


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.skipif(
    not hasattr(torch.ops, "_rocm_C")
    or not hasattr(torch.ops._rocm_C, "wvSplitK_int4_g"),
    reason="wvSplitK_int4_g not built",
)
def test_wvsplitk_int4_g_rejects_undersized_weight_view():
    """A sufficient row stride does not prove the row holds K/2 packed bytes.

    Row padding makes stride(0) > K/2 legitimate, but the logical size still
    has to be checked: this view has a valid stride and half the required
    bytes, and without the size check the kernel reads past it.
    """
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    K, N, G, M = 512, 256, 128, 1
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    # Full-size backing allocation, so a missing check misreads rather than
    # running off the end of the buffer.
    backing = torch.randint(0, 255, (N, K // 2), device=device, dtype=torch.uint8).view(
        torch.int8
    )
    undersized = backing[:, : K // 4]
    assert undersized.stride(0) == K // 2
    scales = torch.rand((N, K // G), device=device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match=r"must contain M\*K/2 bytes"):
        ops.wvSplitK_int4_g(undersized, a, scales, num_compute_units(), G, None, None)


# ---------------------------------------------------------------------------
# gfx11 weight row-stride padding
# ---------------------------------------------------------------------------

_weight_pad_bytes = hybrid_module._weight_pad_bytes
_act_pad_bytes = hybrid_module._act_pad_bytes
_WEIGHT_CLIFF_BYTES = hybrid_module._WEIGHT_CLIFF_BYTES
_ACT_CLIFF_BYTES = hybrid_module._ACT_CLIFF_BYTES
_STRIDE_PAD_BYTES = hybrid_module._STRIDE_PAD_BYTES


def test_weight_pad_bytes_only_moves_strides_on_the_cliff():
    """Pad 1024 B multiples, leave everything else dense."""
    # On the cliff (K % 2048 == 0): 1024 B multiples.
    for row_bytes in (1024, 2048, 4096, 5120, 6144, 7168, 8192):
        assert _weight_pad_bytes(row_bytes) == _STRIDE_PAD_BYTES

    # Off it, including strides that are 512 B multiples but not 1024 B ones:
    # padding those measured as a real loss on gfx1151 (-14% at M=1 on a
    # 4864 B row).
    for row_bytes in (1280, 1536, 2560, 4864, 9472, 12800):
        assert _weight_pad_bytes(row_bytes) == 0

    # The padded stride is never back on the cliff.
    for row_bytes in range(16, 16384, 16):
        padded = row_bytes + _weight_pad_bytes(row_bytes)
        assert padded % _WEIGHT_CLIFF_BYTES != 0 or row_bytes % _WEIGHT_CLIFF_BYTES


def test_act_pad_bytes_only_moves_strides_on_the_cliff():
    """Activations sit on a wider cliff than the packed weight: 2048 B."""
    # On the cliff (K % 1024 == 0 for a 2-byte dtype): 2048 B multiples.
    for row_bytes in (2048, 4096, 8192, 16384, 20480, 24576, 43008):
        assert _act_pad_bytes(row_bytes) == _STRIDE_PAD_BYTES

    # Off it, including 1024 B multiples that are not 2048 B ones -- the
    # weight cliff must not be applied to activations.
    for row_bytes in (1024, 3072, 5120, 10752, 19456, 37888):
        assert _act_pad_bytes(row_bytes) == 0

    # The padded stride is never back on the cliff.
    for row_bytes in range(16, 65536, 16):
        padded = row_bytes + _act_pad_bytes(row_bytes)
        assert padded % _ACT_CLIFF_BYTES != 0 or row_bytes % _ACT_CLIFF_BYTES


def _row_padded_copy(t: torch.Tensor, pad_cols: int) -> torch.Tensor:
    """Copy of ``t`` whose stride(0) is ``t.shape[1] + pad_cols``."""
    rows, cols = t.shape
    buf = torch.empty((rows, cols + pad_cols), dtype=t.dtype, device=t.device)
    buf[:, :cols].copy_(t)
    return buf[:, :cols]


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize("M", [1, MAX_SKINNY_BATCH_SIZE + 1])
def test_weight_row_padding_does_not_change_results(dtype, has_zp, M):
    """Padding the weight rows is a pure layout change.

    Both the HIP skinny kernel (M small) and the Triton prefill kernel take the
    weight row stride from the tensor, so a padded layout must produce exactly
    the same numbers as the dense one.
    """
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(0)
    K, N, G = 512, 256, 128

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    w_q = pack_int4_exllama_shuffle(w_int4_nk).view(torch.int8)
    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )
    zp = (
        _pack_zp_rows_for_kernel(
            torch.randint(0, 16, (N, K // G), device=device, dtype=torch.int32)
        )
        if has_zp
        else None
    )

    # 32 int32 columns = 128 B of pad, the production pad size.
    w_q_pad = _row_padded_copy(w_q.view(torch.int32), 32).view(torch.int8)
    assert w_q_pad.stride(0) == w_q.stride(0) + _STRIDE_PAD_BYTES

    cu_count = num_compute_units()
    dense = torch.ops.vllm.rdna_hybrid_w4a16_apply(
        a, w_q, scales, zp, None, cu_count, G
    )
    padded = torch.ops.vllm.rdna_hybrid_w4a16_apply(
        a, w_q_pad, scales, zp, None, cu_count, G
    )
    torch.testing.assert_close(padded, dense, rtol=0, atol=0)


@pytest.mark.skipif(
    not hybrid_module._on_gfx1151(),
    reason="row-stride padding is only enabled on gfx1151",
)
@pytest.mark.parametrize(
    "K,expected_weight_stride",
    [
        (512, 256),  # 256 B row: off the cliff, stored dense
        (8192, 4096 + 128),  # 4096 B row: on the cliff, padded
    ],
)
def test_process_weights_pads_cliff_rows(K, expected_weight_stride, dist_init):
    """The stored weight row stride follows the padding rule."""
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    set_random_seed(0)
    N, G = 128, 128

    w_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    layer = _build_dummy_layer(
        _pack_int4_along_k_to_ckpt(w_int4_kn),
        0.05 * torch.rand((N, K // G), device=device, dtype=torch.float16),
        zeros_ckpt=None,
    )
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=G,
        zero_points=False,
    )
    RDNAHybridW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name=None,
    ).process_weights_after_loading(layer)

    # Weight rows: stride is in int8 elements, i.e. bytes.
    assert layer.weight_packed.stride(0) == expected_weight_stride
    # The int32 view the Triton path uses survives the padded stride.
    assert tuple(layer.weight_packed.view(torch.int32).shape) == (N, K // 8)
    # Metadata is untouched by this change.
    assert layer.weight_scale.is_contiguous()


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize(
    "M,K,expected_path",
    [
        (4, 8192, "hip"),
        (4, 9728, "hip"),
        (4, 9856, "triton"),
        (MAX_SKINNY_BATCH_SIZE + 1, 1024, "triton"),
    ],
    ids=["regular_limit", "medium_range", "above_medium", "batch_too_large"],
)
def test_rdna_hybrid_w4a16_dispatch_boundaries(M, K, expected_path, monkeypatch):
    """Route only supported regular and medium skinny shapes to HIP."""
    import vllm._custom_ops as ops

    N, G = 16, 128
    x_mk = torch.empty((M, K), dtype=torch.float16)
    w_q = torch.empty((N, K // 2), dtype=torch.int8)
    scales = torch.empty((N, K // G), dtype=torch.float16)
    called = []

    def fake_hip(*args, **kwargs):
        called.append("hip")
        return torch.empty((M, N), dtype=x_mk.dtype)

    def fake_triton(*args, **kwargs):
        called.append("triton")
        return torch.empty((M, N), dtype=x_mk.dtype)

    monkeypatch.setattr(ops, "wvSplitK_int4_g", fake_hip)
    monkeypatch.setattr(hybrid_module, "triton_w4a16_skinny_fmt_gemm", fake_triton)

    output = hybrid_module._rdna_hybrid_w4a16_apply_impl(
        x_mk, w_q, scales, None, None, 40, G
    )

    assert output.shape == (M, N)
    assert called == [expected_path]


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_zp", [False, True])
def test_rdna_hybrid_w4a16_medium_skinny_matches_reference(dtype, has_zp):
    """Validate the HIP medium path against an FP32 dequantization reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(0)
    M, K, N, G = 4, 9728, 256, 128
    assert LDS_CAPACITY_ELEMENTS < K * M <= MEDIUM_SKINNY_LIMIT_ELEMENTS

    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    w_q = pack_int4_exllama_shuffle(w_int4_nk).contiguous().view(torch.int8)
    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )
    zp = (
        torch.randint(0, 16, (N, K // G), device=device, dtype=torch.int32)
        if has_zp
        else None
    )
    w_zp = _pack_zp_rows_for_kernel(zp) if zp is not None else None

    output = torch.ops.vllm.rdna_hybrid_w4a16_apply(
        x_mk, w_q, scales, w_zp, None, num_compute_units(), G
    )
    reference = _rdna_hybrid_w4a16_reference(x_mk, w_int4_nk, scales, zp, G, bias=None)

    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
