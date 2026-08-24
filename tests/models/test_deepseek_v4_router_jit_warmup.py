# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate DSv4 router JIT dispatch."""

import pytest

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

if not current_platform.is_cuda_alike():
    pytest.skip("NVIDIA dispatch tests require CUDA", allow_module_level=True)

requires_cutedsl = pytest.mark.skipif(
    not has_cutedsl(),
    reason="CuTeDSL is not installed",
)

if has_cutedsl():
    import vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 as ll_bf16_module
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import LLBf16Gemm
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (
        BF16x3RouterGemmKernel,
        BF16x3SplitKReduceKernel,
    )


@pytest.mark.parametrize(
    ("split_k", "expected_bn", "expected_bm", "expected_bs"),
    [(1, 16, 32, 1), (2, 16, 32, 2), (8, 1, 256, 8), (64, 1, 32, 64)],
)
def test_bf16x3_splitk_reduce_dispatch_matches_legacy_config(
    split_k: int,
    expected_bn: int,
    expected_bm: int,
    expected_bs: int,
) -> None:
    kernel = BF16x3SplitKReduceKernel()

    assert kernel.dispatch(M=256, split_k=split_k, USE_PDL=True) == kernel.CompileKey(
        m=256,
        bn=expected_bn,
        bm=expected_bm,
        bs=expected_bs,
        use_pdl=True,
    )


@requires_cutedsl
@pytest.mark.parametrize(
    ("M", "K", "N"),
    [
        (4, 7168, 256),
        (6, 7168, 256),
        (7, 7168, 256),
        (5, 7168, 384),
        (8, 7168, 384),
        (16, 1024, 256),
    ],
)
def test_ll_bf16_dispatch_matches_legacy_config(
    M: int,
    K: int,
    N: int,
) -> None:
    kernel = LLBf16Gemm()
    tuned_bs, tuned_splitk = ll_bf16_module._arch_tuned_configs()
    if M <= ll_bf16_module._DEFAULT_DOTPROD_MAX_M or K < 2048:
        bs = tuned_bs.get((K, N), {}).get(M, ll_bf16_module._DEFAULT_DOTPROD_BS)
        expected = kernel.CompileKey(backend="dotprod", m=M, k=K, bs=bs)
    else:
        split_k, num_stages = tuned_splitk.get((K, N), {}).get(
            M, ll_bf16_module._DEFAULT_SPLITK_CONFIG
        )
        expected = kernel.CompileKey(
            backend="splitk",
            split_k=split_k,
            num_stages=num_stages,
        )

    assert kernel.dispatch(M=M, K=K, N=N) == expected


@requires_cutedsl
@pytest.mark.parametrize(
    ("num_tokens", "expected_bn"),
    [(1, 8), (8, 8), (9, 16), (128, 128), (129, 128)],
)
def test_bf16x3_dispatch_matches_legacy_bn(
    num_tokens: int,
    expected_bn: int,
) -> None:
    kernel = BF16x3RouterGemmKernel()

    assert kernel.dispatch(num_tokens=num_tokens, K=6144) == kernel.CompileKey(
        bn=expected_bn,
        k=6144,
    )
