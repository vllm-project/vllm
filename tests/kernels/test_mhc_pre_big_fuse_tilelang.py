# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the TileLang mHC-pre fused backend."""

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm, has_tilelang

HC_MULT = 4
HC_MULT3 = HC_MULT * (2 + HC_MULT)
DEEPSEEK_V4_HIDDEN_SIZE = 4096
DEEPSEEK_V4_SINKHORN_ITERS = 20
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only"),
    pytest.mark.skipif(not has_tilelang(), reason="TileLang is required"),
]

if current_platform.is_cuda() and has_tilelang():
    import tilelang
    import tilelang.language as T
else:
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]


def _make_fused_inputs(
    num_tokens: int,
    hidden_size: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2026 + num_tokens * 17 + hidden_size + n_splits)
    gemm_out_mul = torch.randn(
        n_splits,
        num_tokens,
        HC_MULT3,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    gemm_out_sqrsum = (
        torch.rand(
            n_splits,
            num_tokens,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * (HC_MULT * hidden_size)
        + 1.0
    )
    hc_scale = torch.randn(3, dtype=torch.float32, device="cuda", generator=generator)
    hc_base = torch.randn(
        HC_MULT3, dtype=torch.float32, device="cuda", generator=generator
    )
    residual = torch.randn(
        num_tokens,
        HC_MULT,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    return gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual


def _alloc_outputs(
    num_tokens: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(num_tokens, HC_MULT, dtype=torch.float32, device="cuda"),
        torch.empty(num_tokens, HC_MULT * HC_MULT, dtype=torch.float32, device="cuda"),
        torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"),
    )


if tilelang is not None:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        },
    )
    def _reference_mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size: int,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 16,
        hc_mult: int = 4,
    ):
        """TileLang reference with the old float32 residual staging buffer."""
        num_tokens = T.dynamic("num_tokens")
        hc_mult3 = hc_mult * (2 + hc_mult)
        hidden_block = math.gcd(512, hidden_size)

        gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
        gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
        hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
        hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
        residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
        post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
        comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
        layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

        with T.Kernel(num_tokens, threads=96) as i:
            T.pdl_sync()
            ##################################################################
            # _pre_norm_fn_fwd_norm
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(hc_mult3, T.float32)
            T.clear(mixes)
            rms[0] = 0
            for i_split in T.serial(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, i]
            rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
            for j in T.Parallel(hc_mult3):
                mixes[j] = 0
                for i_split in T.serial(n_splits):
                    mixes[j] += gemm_out_mul[i_split, i, j]
                mixes[j] *= rms[0]
            mixes_shared = T.alloc_shared(hc_mult3, T.float32)
            T.copy(mixes, mixes_shared)

            if T.get_thread_binding() < 32:
                ##################################################################
                # _pre_split_mixes_fwd (post & comb)
                cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
                for j in T.Parallel(hc_mult):
                    post_mix[i, j] = (
                        T.sigmoid(
                            mixes_shared[j + hc_mult] * hc_scale[1]
                            + hc_base[j + hc_mult]
                        )
                        * hc_post_mult_value
                    )
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = (
                        mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                        + hc_base[j * hc_mult + k + hc_mult * 2]
                    )

                ##################################################################
                # _sinkhorn_fwd
                row_sum = T.alloc_fragment(hc_mult, T.float32)
                col_sum = T.alloc_fragment(hc_mult, T.float32)

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(hc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                # save comb_mix to global memory
                for j, k in T.Parallel(hc_mult, hc_mult):
                    comb_mix[i, j * hc_mult + k] = cm[j, k]
            else:
                ##################################################################
                # _pre_split_mixes_fwd (pre)
                pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
                for j in T.Parallel(hc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(
                            mixes_shared[j] * hc_scale[0] + hc_base[j],
                        )
                        + hc_pre_eps
                    )
                ###################################################################
                # _pre_apply_mix_fwd
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                    xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                    T.copy(residual[i, 0, i0_h * hidden_block], xs)
                    T.copy(xs, xl)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)

                    for i_hc in T.serial(hc_mult):
                        pre = pre_mix_shared[i_hc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_hc, i1_h]

                    T.copy(ol, layer_input[i, i0_h * hidden_block])
            T.pdl_trigger()

else:

    def _reference_mhc_pre_big_fuse_tilelang(*args, **kwargs):
        raise RuntimeError("TileLang is required for this test.")


def _reference_mhc_pre_big_fuse(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    hidden_size: int,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_out = _alloc_outputs(residual.shape[0], hidden_size)
    _reference_mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        ref_out[0],
        ref_out[1],
        ref_out[2],
        hidden_size,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        sinkhorn_repeat,
        gemm_out_mul.shape[0],
        HC_MULT,
    )
    torch.cuda.synchronize()
    return (
        ref_out[0],
        ref_out[1].view(residual.shape[0], HC_MULT, HC_MULT),
        ref_out[2],
    )


@pytest.mark.parametrize(
    ("num_tokens", "hidden_size", "n_splits", "sinkhorn_repeat"),
    [
        (1, DEEPSEEK_V4_HIDDEN_SIZE, 64, 1),
        (1, DEEPSEEK_V4_HIDDEN_SIZE, 64, DEEPSEEK_V4_SINKHORN_ITERS),
        (7, DEEPSEEK_V4_HIDDEN_SIZE, 64, DEEPSEEK_V4_SINKHORN_ITERS),
        (128, DEEPSEEK_V4_HIDDEN_SIZE, 64, DEEPSEEK_V4_SINKHORN_ITERS),
        (256, DEEPSEEK_V4_HIDDEN_SIZE, 37, DEEPSEEK_V4_SINKHORN_ITERS),
        (512, DEEPSEEK_V4_HIDDEN_SIZE, 18, DEEPSEEK_V4_SINKHORN_ITERS),
        (1024, DEEPSEEK_V4_HIDDEN_SIZE, 9, DEEPSEEK_V4_SINKHORN_ITERS),
    ],
)
def test_mhc_pre_big_fuse_tilelang_matches_reference(
    num_tokens: int,
    hidden_size: int,
    n_splits: int,
    sinkhorn_repeat: int,
):
    from vllm._tilelang_ops import mhc_pre_big_fuse_tilelang

    gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual = _make_fused_inputs(
        num_tokens, hidden_size, n_splits
    )
    tilelang_out = _alloc_outputs(num_tokens, hidden_size)
    mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        tilelang_out[0],
        tilelang_out[1],
        tilelang_out[2],
        hidden_size,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        sinkhorn_repeat,
        n_splits,
        HC_MULT,
    )
    torch.cuda.synchronize()

    ref_post, ref_comb, ref_layer = _reference_mhc_pre_big_fuse(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        hidden_size,
        sinkhorn_repeat,
    )
    torch.testing.assert_close(tilelang_out[0], ref_post, atol=0, rtol=0)
    torch.testing.assert_close(
        tilelang_out[1].view(num_tokens, HC_MULT, HC_MULT),
        ref_comb,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        tilelang_out[2],
        ref_layer,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("num_tokens", "n_splits"),
    [
        (1, 64),
        (128, 64),
        (256, 37),
        (2048, 4),
    ],
)
def test_mhc_pre_big_fuse_tilelang_is_deterministic(
    num_tokens: int,
    n_splits: int,
):
    from vllm._tilelang_ops import mhc_pre_big_fuse_tilelang

    hidden_size = DEEPSEEK_V4_HIDDEN_SIZE
    gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual = _make_fused_inputs(
        num_tokens, hidden_size, n_splits
    )
    actual_out = _alloc_outputs(num_tokens, hidden_size)
    expected_out = _alloc_outputs(num_tokens, hidden_size)
    params = (
        hidden_size,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        DEEPSEEK_V4_SINKHORN_ITERS,
        n_splits,
        HC_MULT,
    )

    mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        actual_out[0],
        actual_out[1],
        actual_out[2],
        *params,
    )
    mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        expected_out[0],
        expected_out[1],
        expected_out[2],
        *params,
    )
    torch.cuda.synchronize()

    for actual, expected in zip(actual_out, expected_out):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM is required for mhc_pre")
@pytest.mark.parametrize(
    ("outer_shape", "sinkhorn_repeat"),
    [
        ((1,), DEEPSEEK_V4_SINKHORN_ITERS),
        ((3,), DEEPSEEK_V4_SINKHORN_ITERS),
        ((2, 2), DEEPSEEK_V4_SINKHORN_ITERS),
    ],
)
def test_registered_mhc_pre_uses_tilelang(
    outer_shape: tuple[int, ...],
    sinkhorn_repeat: int,
    default_vllm_config,
):
    from vllm.model_executor.layers.mhc import MHCPreOp

    assert default_vllm_config is not None
    hidden_size = DEEPSEEK_V4_HIDDEN_SIZE
    generator = torch.Generator(device="cuda")
    generator.manual_seed(12345 + hidden_size + sinkhorn_repeat + sum(outer_shape))
    residual = torch.randn(
        *outer_shape,
        HC_MULT,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    fn = torch.randn(
        HC_MULT3,
        HC_MULT * hidden_size,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    hc_scale = torch.randn(3, dtype=torch.float32, device="cuda", generator=generator)
    hc_base = torch.randn(
        HC_MULT3, dtype=torch.float32, device="cuda", generator=generator
    )
    args = (
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        sinkhorn_repeat,
    )

    mhc_pre_op = MHCPreOp()
    default_out = mhc_pre_op(*args)
    second_out = mhc_pre_op(*args)
    torch.cuda.synchronize()

    for actual, expected in zip(default_out, second_out):
        torch.testing.assert_close(
            actual.to(torch.float32),
            expected.to(torch.float32),
            atol=0,
            rtol=0,
        )
