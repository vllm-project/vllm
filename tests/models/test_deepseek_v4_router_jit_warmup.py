# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate DSv4 router JIT dispatch."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

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
        BF16x3SplitKReduceKernel,
    )


@requires_cutedsl
def test_bf16x3_router_warmup_covers_pdl_variants() -> None:
    kernel = BF16x3RouterGemmKernel()
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(enable_bf16x3_router_gemm=True),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=6144)),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
    )

    warmed_keys = kernel.get_warmup_keys(config)

    assert warmed_keys
    assert {key.use_pdl for key in warmed_keys} == {False, True}


@requires_cutedsl
@pytest.mark.parametrize(
    (
        "N",
        "split_k",
        "expected_n",
        "expected_k_splits",
        "expected_bn",
        "expected_bm",
        "expected_bs",
    ),
    [
        (1, 1, 1, 1, 16, 32, 1),
        (2, 2, 2, 2, 16, 32, 2),
        (8, 8, 2, 5, 1, 256, 8),
        (16, 64, 16, 48, 1, 32, 64),
        (3, 74, 2, 65, 1, 32, 128),
        (3, 80, 2, 80, 1, 32, 128),
    ],
)
def test_bf16x3_splitk_reduce_dispatch_matches_legacy_config(
    N: int,
    split_k: int,
    expected_n: int,
    expected_k_splits: int,
    expected_bn: int,
    expected_bm: int,
    expected_bs: int,
) -> None:
    kernel = BF16x3SplitKReduceKernel()

    assert kernel.dispatch(
        N=N, M=256, split_k=split_k, launch_pdl=True
    ) == kernel.CompileKey(
        m=256,
        n=expected_n,
        split_stride=16,
        k_splits=expected_k_splits,
        bn=expected_bn,
        bm=expected_bm,
        bs=expected_bs,
        launch_pdl=True,
    )


@requires_cutedsl
def test_bf16x3_splitk_reduce_warmup_covers_runtime_keys() -> None:
    kernel = BF16x3SplitKReduceKernel()
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(enable_bf16x3_router_gemm=True),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(n_routed_experts=256, hidden_size=6144)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
    )
    warmed_keys = set(kernel.get_warmup_keys(config))

    assert {key.launch_pdl for key in warmed_keys} == {False, True}
    for num_tokens in range(1, 129):
        for split_k in range(1, 97):
            assert (
                kernel.dispatch(
                    N=num_tokens,
                    M=256,
                    split_k=split_k,
                    launch_pdl=current_platform.is_arch_support_pdl(),
                )
                in warmed_keys
            )


@requires_cutedsl
def test_bf16x3_splitk_reduce_checks_pdl_at_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = BF16x3SplitKReduceKernel()
    launch = Mock()
    monkeypatch.setattr(current_platform, "is_arch_support_pdl", lambda: False)
    monkeypatch.setattr(kernel, "launch", launch)

    kernel(torch.empty(2, 2, 256), torch.empty(2, 256))

    assert launch.call_args.kwargs["launch_pdl"] is False


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
        num_tokens=num_tokens, K=6144, use_pdl=use_pdl
    ) == kernel.CompileKey(
        bn=expected_bn,
        k=6144,
        use_pdl=use_pdl,
    )
