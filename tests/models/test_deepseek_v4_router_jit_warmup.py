# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate DSv4 router JIT dispatch."""

import pytest

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

if not current_platform.is_cuda():
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
    )


@requires_cutedsl
def test_bf16x3_router_warmup_covers_pdl_variants() -> None:
    kernel = BF16x3RouterGemmKernel()
    warmed_keys = kernel.get_warmup_keys(
        K=6144,
        M=256,
        num_sms=160,
        max_tokens=128,
    )

    assert warmed_keys
    assert {key.use_pdl for key in warmed_keys} == {
        current_platform.is_arch_support_pdl()
    }


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
@pytest.mark.parametrize("use_pdl", [False, True])
def test_bf16x3_dispatch_matches_legacy_bn(
    num_tokens: int,
    expected_bn: int,
    use_pdl: bool,
) -> None:
    kernel = BF16x3RouterGemmKernel()

    assert kernel.dispatch(
        num_tokens=num_tokens,
        K=6144,
        M=256,
        num_sms=160,
        use_pdl=use_pdl,
    ) == kernel.CompileKey(
        bn=expected_bn,
        k=6144,
        use_pdl=use_pdl,
    )
