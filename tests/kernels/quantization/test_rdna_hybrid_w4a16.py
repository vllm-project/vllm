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

from vllm.platforms.rocm import on_gfx1x, on_gfx115x  # noqa: E402

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
    zp_nkg: torch.Tensor | None,
) -> torch.Tensor:
    """Reference for HIP skinny: C = A @ (W - zp) * S.

    ``zp_nkg`` is [N, K//G] int32 raw zero points for the asymmetric case, or
    None for symmetric uint4b8, where the kernel subtracts a constant 8.
    """
    K = a_mk.shape[1]
    N = w_int4_nk.shape[0]
    num_groups = K // group_size

    w_g = w_int4_nk.to(torch.float32).view(N, num_groups, group_size)
    zp = 8.0 if zp_nkg is None else zp_nkg.to(torch.float32).unsqueeze(-1)
    s = scales_nkg.to(torch.float32).unsqueeze(-1)
    w_dequant = ((w_g - zp) * s).view(N, K)

    return (a_mk.to(torch.float32) @ w_dequant.t()).to(a_mk.dtype)


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        (1, 256, 256, 32),
        (1, 256, 256, 64),
        (1, 512, 256, 128),
        (2, 512, 256, 64),
        (3, 256, 512, 64),
        # M is the kernel's batch dimension; 4 and 5 reach dispatch tuples that
        # 1..3 never take, and 5 is MAX_SKINNY_BATCH_SIZE.
        (4, 512, 256, 32),
        (4, 256, 512, 64),
        (5, 512, 256, 128),
        (5, 256, 256, 32),
        # N == 8 is exactly one packed zero-point word.
        (1, 256, 8, 32),
        # K must exceed THRDS * A_CHUNK * UNRL for the K loop to run a whole
        # unrolled block; every K above is below that bound, so without these
        # the deep-K path goes untested. 2560 is not a multiple of the 1024 K
        # step the batch 2 and 4 tuples take, so it also hits the ragged tail.
        (1, 4096, 256, 128),
        (2, 2560, 256, 32),
        (3, 4096, 256, 64),
        (4, 2560, 256, 128),
        (5, 4096, 256, 32),
        # On gfx115x, batch 1 with K == 4096, or symmetric with K a multiple of
        # 8192, takes a tuple with A_CHUNK = 32 instead of 16. G = 32 is the
        # tight case there: each lane's A_CHUNK spans exactly one scale group.
        (1, 4096, 256, 32),
        (1, 8192, 256, 128),
        (1, 16384, 256, 64),
        # K * batch beyond what LDS holds (32768 fp16 elements), so gfx115x
        # walks K in chunks; elsewhere the op rejects these, as the layer
        # routes them to Triton. The per-row window shrinks as the batch grows,
        # so these cover a single reload and several, with and without a short
        # final chunk. 20992 is not a multiple of the K step either, so it
        # crosses a chunk boundary *and* has a ragged K tail.
        (2, 20480, 512, 64),
        (3, 16384, 512, 32),
        (4, 16384, 512, 128),
        (5, 21504, 512, 32),
        (5, 20992, 512, 128),
    ],
)
def test_hip_skinny_wvSplitK_int4_g(dtype, M, K, N, G, has_zp):
    """Test HIP wvSplitK_int4_g kernel directly via _custom_ops."""
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    if K * M > LDS_CAPACITY_ELEMENTS and not on_gfx115x():
        pytest.skip("deep-K HIP path is gfx115x only")

    set_random_seed(0)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)

    b_packed_i32 = pack_int4_exllama_shuffle(w_int4_nk)
    b_packed_i8 = b_packed_i32.view(torch.int8)

    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )

    zp_nkg = None
    zp_packed = None
    if has_zp:
        zp_nkg = torch.randint(0, 16, (N, K // G), device=device, dtype=torch.int32)
        zp_packed = _pack_zp_rows_for_kernel(zp_nkg)

    cu_count = num_compute_units()
    out = ops.wvSplitK_int4_g(b_packed_i8, a, scales, cu_count, G, zp_packed)

    ref = _hip_skinny_reference(a, w_int4_nk, scales, group_size=G, zp_nkg=zp_nkg)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("G", [32, 128])
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize("M", [2, 64], ids=["M=2_decode", "M=64_prefill"])
def test_hip_skinny_padded_group_stride(dtype, G, has_zp, M):
    """Scale and zero-point rows may be padded past K // group_size.

    Both paths index them by their own row stride, so a padded row is read
    correctly rather than bleeding into the next row. M picks the path: the
    HIP skinny kernel at 2, the Triton prefill kernel at 64.
    """
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(0)
    K, N = 2048, 256
    num_groups = K // G
    pad = 7

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    b_packed_i8 = pack_int4_exllama_shuffle(w_int4_nk).view(torch.int8)

    scales = (0.05 * torch.rand((N, num_groups), device=device)).to(dtype)
    scales_padded = torch.zeros((N, num_groups + pad), device=device, dtype=dtype)
    scales_padded[:, :num_groups] = scales
    scales_view = scales_padded[:, :num_groups]
    assert scales_view.stride(0) == num_groups + pad

    zp_nkg = zp_view = None
    if has_zp:
        zp_nkg = torch.randint(0, 16, (N, num_groups), device=device, dtype=torch.int32)
        zp_packed = _pack_zp_rows_for_kernel(zp_nkg)
        zp_padded = torch.zeros(
            (N // 8, num_groups + pad), device=device, dtype=torch.int32
        )
        zp_padded[:, :num_groups] = zp_packed
        zp_view = zp_padded[:, :num_groups]

    cu_count = num_compute_units()
    out = torch.ops.vllm.rdna_hybrid_w4a16_apply(
        a, b_packed_i8, scales_view, zp_view, None, cu_count, G
    )
    ref = _hip_skinny_reference(a, w_int4_nk, scales, group_size=G, zp_nkg=zp_nkg)

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
        # Past LDS: HIP on gfx115x, Triton elsewhere.
        (2, 20480, 256, 64),
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

    ref = _hip_skinny_reference(a, w_int4_nk, scales, group_size=G, zp_nkg=None)

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
    "M,K,gfx115x,expected_path",
    [
        (4, 8192, False, "hip"),
        (4, 9728, False, "hip"),
        (4, 9856, False, "triton"),
        (4, 9856, True, "hip"),
        (MAX_SKINNY_BATCH_SIZE + 1, 1024, True, "triton"),
    ],
    ids=[
        "regular_limit",
        "medium_range",
        "above_medium",
        "above_medium_gfx115x",
        "batch_too_large",
    ],
)
def test_rdna_hybrid_w4a16_dispatch_boundaries(
    M, K, gfx115x, expected_path, monkeypatch
):
    """Route only supported regular and medium skinny shapes to HIP.

    gfx115x drops the K*M bound, so a shape past the medium limit still goes to
    HIP there; the batch bound holds on every target.
    """
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
    monkeypatch.setattr(hybrid_module, "_on_gfx115x", lambda: gfx115x)

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
