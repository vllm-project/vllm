# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate JIT dispatch against pre-contract behavior."""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("NVIDIA dispatch tests require CUDA", allow_module_level=True)

from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    HcHeadFusedTileLangKernel,
    HcPrenormGemmTileLangKernel,
    MhcFusedTileLangKernel,
    MhcPostTileLangKernel,
    MhcPreBigFuseTileLangKernel,
    mhc_fused_post_pre_split_config,
    require_fused_post_pre_config,
)
from vllm.model_executor.warmup import jit_warmup_tilelang_helper


@pytest.mark.parametrize("num_sms", [78, 132, 148])
@pytest.mark.parametrize("max_tokens", [1, 64, 65, 129, 8192, 16384])
def test_glm_deep_gemm_warmup_covers_runtime_splits(
    monkeypatch, num_sms, max_tokens
) -> None:
    import torch

    from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split
    from vllm.model_executor.kernels.mhc.warmup import HcPrenormGemmDeepGemmKernel

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=num_sms),
    )
    compute_num_split.cache_clear()
    try:
        kernel = HcPrenormGemmDeepGemmKernel()
        keys = kernel.get_warmup_keys(n=24, k=16384, max_tokens=max_tokens)
        expected = {
            kernel.dispatch(n=24, k=16384, num_tokens=m)
            for m in range(1, max_tokens + 1)
        }
        assert len(keys) == len(set(keys))
        assert set(keys) == expected
    finally:
        compute_num_split.cache_clear()


def test_glm_deep_gemm_warmup_compiles_without_tensor_inputs(monkeypatch) -> None:
    from vllm.model_executor.kernels.mhc import warmup

    calls = []
    monkeypatch.setattr(
        warmup, "compile_tf32_hc_prenorm_gemm", lambda **kwargs: calls.append(kwargs)
    )
    kernel = warmup.HcPrenormGemmDeepGemmKernel()
    kernel.compile(kernel.CompileKey(n=24, k=16384, num_splits=26))

    assert calls == [dict(n=24, k=16384, num_splits=26)]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                num_tokens=64,
                hc_hidden_size=4096,
                hidden_size=2048,
                hc_mult=2,
                n_out=128,
            ),
            (2048, 2, 128, 1024, 4, 1, False, 1),
        ),
        (
            dict(
                num_tokens=1024,
                hc_hidden_size=4096,
                hidden_size=2048,
                hc_mult=2,
                n_out=128,
            ),
            (2048, 2, 128, 512, 12, 1, True, 2),
        ),
        (
            dict(
                num_tokens=64,
                hc_hidden_size=4096,
                hidden_size=2048,
                hc_mult=2,
                n_out=128,
                n_thr=256,
                tile_n=8,
                n_splits=4,
            ),
            (2048, 2, 128, 256, 8, 4, False, 1),
        ),
    ],
)
def test_hc_prenorm_gemm_dispatch_matches_legacy_runtime_config(
    kwargs: dict[str, Any],
    expected: tuple[int, int, int, int, int, int, bool, int],
) -> None:
    kernel = HcPrenormGemmTileLangKernel()

    assert kernel.dispatch(**kwargs) == kernel.CompileKey(*expected)


@pytest.mark.parametrize(
    ("is_broadcast", "use_norm_weight", "expected_use_norm", "expected_eps"),
    [
        (False, False, False, 0.0),
        (False, True, True, 1.0e-5),
        (True, False, True, 2.0e-5),
    ],
)
def test_mhc_pre_big_fuse_dispatch_matches_legacy_runtime_config(
    is_broadcast: bool,
    use_norm_weight: bool,
    expected_use_norm: bool,
    expected_eps: float,
) -> None:
    kernel = MhcPreBigFuseTileLangKernel()

    assert kernel.dispatch(
        hidden_size=4096,
        hc_mult=4,
        n_splits=2,
        is_broadcast=is_broadcast,
        use_norm_weight=use_norm_weight,
        rms_eps=1.0e-6,
        hc_pre_eps=2.0e-6,
        hc_sinkhorn_eps=3.0e-6,
        hc_post_mult_value=0.5,
        sinkhorn_repeat=3,
        norm_eps=1.0e-5,
        broadcast_norm_eps=2.0e-5,
    ) == kernel.CompileKey(
        hidden_size=4096,
        hc_mult=4,
        n_splits=2,
        use_norm_weight=expected_use_norm,
        is_broadcast=is_broadcast,
        rms_eps=1.0e-6,
        hc_pre_eps=2.0e-6,
        hc_sinkhorn_eps=3.0e-6,
        hc_post_mult_value=0.5,
        sinkhorn_repeat=3,
        norm_eps=expected_eps,
    )


@pytest.mark.parametrize(
    ("num_tokens", "hidden_size", "expected_tile_n", "expected_n_splits"),
    [
        (1, 4096, 2, 8),
        (4, 8192, 2, 8),
        (8, 5120, 2, 8),
        (16, 5120, 6, 8),
        (32, 7168, 6, 8),
    ],
)
def test_mhc_fused_dispatch_matches_the_launch_config(
    num_tokens: int,
    hidden_size: int,
    expected_tile_n: int,
    expected_n_splits: int,
) -> None:
    """The compile key must be the config the launch will actually use.

    Both read mhc_fused_post_pre_split_config, so this pins the tuned bands
    and guards against the two drifting apart again.
    """
    kernel = MhcFusedTileLangKernel()
    tile_n, n_splits, n_thr = require_fused_post_pre_config(
        num_tokens, hidden_size, hc_mult=4
    )

    assert (tile_n, n_splits) == (expected_tile_n, expected_n_splits)
    assert kernel.dispatch(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        hc_mult=4,
    ) == kernel.CompileKey(
        hidden_size=hidden_size,
        hc_mult=4,
        n_splits=n_splits,
        tile_n=tile_n,
        n_thr=n_thr,
    )


def test_mhc_fused_declines_shapes_it_cannot_tile() -> None:
    """Above the token cutoff, and for a hidden size the block cannot split."""
    assert mhc_fused_post_pre_split_config(33, 5120, 4) is None
    assert mhc_fused_post_pre_split_config(1, 5137, 4) is None
    with pytest.raises(ValueError, match="does not cover num_tokens"):
        require_fused_post_pre_config(33, 5120, 4)


@pytest.mark.parametrize(
    ("kernel", "compile_key"),
    [
        (
            HcPrenormGemmTileLangKernel(),
            HcPrenormGemmTileLangKernel.CompileKey(
                hidden_size=2048,
                hc_mult=2,
                n_out=128,
                n_thr=1024,
                tile_n=4,
                n_splits=1,
                use_block_m=False,
                block_m=1,
            ),
        ),
        (
            HcPrenormGemmTileLangKernel(),
            HcPrenormGemmTileLangKernel.CompileKey(
                hidden_size=2048,
                hc_mult=2,
                n_out=128,
                n_thr=512,
                tile_n=12,
                n_splits=1,
                use_block_m=False,
                block_m=1,
            ),
        ),
        (
            HcPrenormGemmTileLangKernel(),
            HcPrenormGemmTileLangKernel.CompileKey(
                hidden_size=2048,
                hc_mult=2,
                n_out=128,
                n_thr=512,
                tile_n=12,
                n_splits=1,
                use_block_m=True,
                block_m=2,
            ),
        ),
        (
            MhcPreBigFuseTileLangKernel(),
            MhcPreBigFuseTileLangKernel.CompileKey(
                hidden_size=4096,
                hc_mult=4,
                n_splits=2,
                use_norm_weight=True,
                is_broadcast=False,
                rms_eps=1.0e-6,
                hc_pre_eps=2.0e-6,
                hc_sinkhorn_eps=3.0e-6,
                hc_post_mult_value=0.5,
                sinkhorn_repeat=3,
                norm_eps=1.0e-5,
            ),
        ),
        (
            MhcPreBigFuseTileLangKernel(),
            MhcPreBigFuseTileLangKernel.CompileKey(
                hidden_size=4096,
                hc_mult=4,
                n_splits=1,
                use_norm_weight=False,
                is_broadcast=False,
                rms_eps=1.0e-6,
                hc_pre_eps=2.0e-6,
                hc_sinkhorn_eps=3.0e-6,
                hc_post_mult_value=0.5,
                sinkhorn_repeat=3,
                norm_eps=0.0,
            ),
        ),
        (
            MhcPreBigFuseTileLangKernel(),
            MhcPreBigFuseTileLangKernel.CompileKey(
                hidden_size=4096,
                hc_mult=4,
                n_splits=2,
                use_norm_weight=True,
                is_broadcast=True,
                rms_eps=1.0e-6,
                hc_pre_eps=2.0e-6,
                hc_sinkhorn_eps=3.0e-6,
                hc_post_mult_value=0.5,
                sinkhorn_repeat=3,
                norm_eps=1.0e-5,
            ),
        ),
        (
            MhcPostTileLangKernel(),
            MhcPostTileLangKernel.CompileKey(hidden_size=4096, hc_mult=4),
        ),
        (
            MhcFusedTileLangKernel(),
            MhcFusedTileLangKernel.CompileKey(
                hidden_size=4096,
                hc_mult=4,
                n_splits=8,
                tile_n=6,
                n_thr=128,
            ),
        ),
        (
            MhcFusedTileLangKernel(),
            MhcFusedTileLangKernel.CompileKey(
                hidden_size=4096,
                hc_mult=4,
                n_splits=8,
                tile_n=2,
                n_thr=128,
            ),
        ),
        (
            MhcFusedTileLangKernel(),
            MhcFusedTileLangKernel.CompileKey(
                hidden_size=8192,
                hc_mult=4,
                n_splits=8,
                tile_n=2,
                n_thr=128,
            ),
        ),
        (
            HcHeadFusedTileLangKernel(),
            HcHeadFusedTileLangKernel.CompileKey(
                hidden_size=4096,
                hc_mult=4,
                rms_eps=1.0e-6,
                hc_eps=2.0e-6,
            ),
        ),
    ],
)
def test_tilelang_warmup_inputs_reproduce_compile_key(
    monkeypatch: pytest.MonkeyPatch,
    kernel: Any,
    compile_key: Any,
) -> None:
    compiled: list[Any] = []
    single_kernel = isinstance(
        kernel,
        (MhcPostTileLangKernel, MhcFusedTileLangKernel, HcHeadFusedTileLangKernel),
    )
    expected_kernel = kernel.kernel() if single_kernel else kernel.kernel(compile_key)
    if single_kernel:
        monkeypatch.setattr(
            kernel,
            "dispatch",
            lambda **kwargs: pytest.fail("single-kernel launch must not dispatch"),
        )
    monkeypatch.setattr(
        jit_warmup_tilelang_helper,
        "compile_tilelang",
        lambda jit_impl, *args, **kwargs: compiled.append(jit_impl),
    )

    kernel.compile(compile_key)

    assert compiled == [expected_kernel]
