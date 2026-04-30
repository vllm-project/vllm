#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCm Triton W4A16 GEMM kernel.

Run `pytest tests/kernels/quantization/test_triton_w4a16.py`.
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# This test module is ROCm/Triton specific. Avoid import-time failures on
# non-ROCm or environments without Triton by skipping early.
if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

pytest.importorskip("triton")

device = "cuda"

triton_w4a16_module = importlib.import_module(
    "vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16"
)
triton_w4a16_gemm = triton_w4a16_module.triton_w4a16_gemm
TritonW4A16LinearKernel = triton_w4a16_module.TritonW4A16LinearKernel


def _pack_int4_along_n(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """Pack int4 values along N: [K, N] -> [K, N//8] int32."""
    assert w_int4_kn.dtype == torch.int32
    K, N = w_int4_kn.shape
    assert N % 8 == 0
    shifts = torch.arange(8, device=w_int4_kn.device, dtype=torch.int32) * 4
    return torch.sum(
        (w_int4_kn.view(K, N // 8, 8) & 0xF) << shifts,
        dim=2,
        dtype=torch.int32,
    ).contiguous()


def _unpack_int4_along_n(w_packed_kn8: torch.Tensor) -> torch.Tensor:
    """Unpack int4 values along N: [K, N//8] -> [K, N] int32."""
    assert w_packed_kn8.dtype == torch.int32
    K, N8 = w_packed_kn8.shape
    shifts = torch.arange(8, device=w_packed_kn8.device, dtype=torch.int32) * 4
    nibbles = (w_packed_kn8.unsqueeze(-1) >> shifts) & 0xF
    return nibbles.reshape(K, N8 * 8)


def _pack_int4_along_k_to_ckpt(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """Pack int4 values along K into CT checkpoint layout: [K,N] -> [N, K//8]."""
    assert w_int4_kn.dtype == torch.int32
    K, N = w_int4_kn.shape
    assert K % 8 == 0
    out = torch.zeros((N, K // 8), dtype=torch.int32, device=w_int4_kn.device)
    for i in range(8):
        out |= (w_int4_kn[i::8, :].t() & 0xF) << (i * 4)
    return out.contiguous()


def _w4a16_reference(
    a_mk: torch.Tensor,
    b_packed_kn8: torch.Tensor,
    scales_gn: torch.Tensor,
    *,
    group_size: int,
    qzeros_gn8: torch.Tensor | None,
    zp_bias: int,
) -> torch.Tensor:
    """Reference implementation for W4A16.

    a_mk: [M,K] fp16/bf16
    b_packed_kn8: [K, N//8] int32, N-packed int4 weights
    scales_gn: [K//G, N] fp16/bf16
    qzeros_gn8: [K//G, N//8] int32, N-packed int4 zeros, or None
    """
    assert a_mk.dtype in (torch.float16, torch.bfloat16)
    assert b_packed_kn8.dtype == torch.int32
    assert scales_gn.dtype == a_mk.dtype

    M, K = a_mk.shape
    N = b_packed_kn8.shape[1] * 8
    assert b_packed_kn8.shape[0] == K

    assert group_size > 0 and K % group_size == 0
    G = group_size
    num_groups = K // G
    assert scales_gn.shape == (num_groups, N)

    w_int4 = _unpack_int4_along_n(b_packed_kn8)  # [K,N]
    if qzeros_gn8 is None:
        z_full = torch.full((K, N), zp_bias, dtype=torch.int32, device=a_mk.device)
    else:
        assert qzeros_gn8.shape == (num_groups, N // 8)
        z_gn = _unpack_int4_along_n(qzeros_gn8)  # [G,N] in groups
        z_full = z_gn.repeat_interleave(G, dim=0)  # [K,N]

    s_full = scales_gn.repeat_interleave(G, dim=0).to(torch.float32)  # [K,N]
    w_fp = (w_int4 - z_full).to(torch.float32) * s_full  # [K,N]

    out = a_mk.to(torch.float32) @ w_fp  # [M,N]
    return out.to(a_mk.dtype)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G,has_zp",
    [
        (1, 256, 256, 32, False),
        (17, 256, 512, 32, False),
        (32, 512, 256, 64, False),
        (33, 512, 512, 128, False),
        (64, 1024, 256, 256, False),
        (128, 256, 1024, 32, True),
        (64, 512, 512, 64, True),
    ],
)
def test_triton_w4a16_gemm_matches_reference(dtype, M, K, N, G, has_zp):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    if N % 8 != 0 or K % G != 0:
        pytest.skip("Invalid test shape")

    set_random_seed(0)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4 = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    b_packed = _pack_int4_along_n(w_int4)

    scales = (0.05 * torch.rand((K // G, N), device=device, dtype=torch.float32)).to(
        dtype
    )

    qzeros = None
    if has_zp:
        zeros_int4 = torch.randint(0, 16, (K // G, N), device=device, dtype=torch.int32)
        qzeros = _pack_int4_along_n(zeros_int4)

    out = triton_w4a16_gemm(
        a=a,
        b_q=b_packed,
        scales=scales,
        qzeros=qzeros,
        group_size=G,
        zp_bias=8,
    )
    ref = _w4a16_reference(
        a,
        b_packed,
        scales,
        group_size=G,
        qzeros_gn8=qzeros,
        zp_bias=8,
    )

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_triton_w4a16_gemm_requires_contiguous_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    set_random_seed(0)
    M, K, N, G = 32, 256, 256, 32
    a = torch.randn((K, M), device=device, dtype=torch.float16).t()  # non-contiguous
    w_int4 = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    b_packed = _pack_int4_along_n(w_int4)
    scales = torch.rand((K // G, N), device=device, dtype=torch.float16)

    with pytest.raises(AssertionError):
        triton_w4a16_gemm(
            a=a,
            b_q=b_packed,
            scales=scales,
            qzeros=None,
            group_size=G,
            zp_bias=8,
        )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_triton_w4a16_process_weights_after_loading_repacks_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedColumnParameter,
        PackedvLLMParameter,
    )
    from vllm.scalar_type import scalar_types

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:0",
            local_rank=0,
        )
        ensure_model_parallel_initialized(1, 1)

    set_random_seed(0)

    # Small-but-nontrivial shapes.
    K, N = 256, 256
    G = 32
    assert K % 8 == 0 and N % 8 == 0 and K % G == 0

    # Build a canonical int4 weight grid then pack into the CT checkpoint layout.
    w_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    w_ckpt_nk8 = _pack_int4_along_k_to_ckpt(w_int4_kn)  # [N, K//8]

    # Scales in CT checkpoint layout for WNA16: [N, K//G]
    scales_ckpt_nkg = 0.05 * torch.rand((N, K // G), device=device, dtype=torch.float16)

    # Asymmetric case: zero points in CT checkpoint layout [N//8, K//G] (N-packed)
    zeros_int4_gn = torch.randint(0, 16, (K // G, N), device=device, dtype=torch.int32)
    zeros_packed_gn8 = _pack_int4_along_n(zeros_int4_gn)  # [K//G, N//8]
    zeros_ckpt_n8kg = zeros_packed_gn8.t().contiguous()  # [N//8, K//G]

    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4,  # asymmetric
        act_type=torch.float16,
        group_size=G,
        zero_points=True,
        has_g_idx=False,
    )
    kernel = TritonW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point",
        w_gidx_param_name=None,
    )

    # Build dummy layer with vLLM parameter wrappers.
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
    layer.register_parameter(
        "weight_zero_point",
        PackedColumnParameter(
            data=zeros_ckpt_n8kg,
            weight_loader=weight_loader,
            output_dim=0,
            packed_factor=8,
            packed_dim=0,
        ),
    )

    kernel.process_weights_after_loading(layer)

    # Expected transformed layouts.
    expected_w_kn8 = _pack_int4_along_n(w_int4_kn)  # [K, N//8]
    expected_scales_gn = scales_ckpt_nkg.t().contiguous()  # [K//G, N]
    expected_zeros_gn8 = zeros_ckpt_n8kg.t().contiguous()  # [K//G, N//8]

    assert tuple(layer.weight_packed.shape) == (K, N // 8)
    assert tuple(layer.weight_scale.shape) == (K // G, N)
    assert tuple(layer.weight_zero_point.shape) == (K // G, N // 8)

    torch.testing.assert_close(layer.weight_packed, expected_w_kn8)
    torch.testing.assert_close(layer.weight_scale, expected_scales_gn)
    torch.testing.assert_close(layer.weight_zero_point, expected_zeros_gn8)
