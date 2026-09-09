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
)


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
            (2048, 2, 128, 1024, 4, 1, False, 1, False),
        ),
        (
            dict(
                num_tokens=1024,
                hc_hidden_size=4096,
                hidden_size=2048,
                hc_mult=2,
                n_out=128,
            ),
            (2048, 2, 128, 512, 12, 1, True, 2, False),
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
            (2048, 2, 128, 256, 8, 4, False, 1, False),
        ),
    ],
)
def test_hc_prenorm_gemm_dispatch_matches_legacy_runtime_config(
    kwargs: dict[str, Any],
    expected: tuple[int, int, int, int, int, int, bool, int, bool],
) -> None:
    kernel = HcPrenormGemmTileLangKernel()

    assert kernel.dispatch(**kwargs, launch_pdl=False) == kernel.CompileKey(*expected)


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
        launch_pdl=False,
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
        launch_pdl=False,
    )


@pytest.mark.parametrize(
    ("num_tokens", "hidden_size", "expected_n_splits", "expected_tile_n"),
    [(4, 4096, 8, 2), (4, 8192, 4, 2), (8, 4096, 4, 3)],
)
def test_mhc_fused_dispatch_matches_legacy_runtime_config(
    num_tokens: int,
    hidden_size: int,
    expected_n_splits: int,
    expected_tile_n: int,
) -> None:
    kernel = MhcFusedTileLangKernel()

    assert kernel.dispatch(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        hc_mult=4,
        launch_pdl=False,
    ) == kernel.CompileKey(
        hidden_size=hidden_size,
        hc_mult=4,
        n_splits=expected_n_splits,
        tile_n=expected_tile_n,
        launch_pdl=False,
    )


def test_mhc_tilelang_warmup_covers_pdl_variants() -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16)
    )
    owners_and_keys = (
        (
            HcPrenormGemmTileLangKernel(),
            dict(
                vllm_config=config,
                hidden_size=4096,
                hc_mult=4,
                n_out=24,
            ),
        ),
        (
            MhcPreBigFuseTileLangKernel(),
            dict(
                vllm_config=config,
                hidden_size=4096,
                hc_mult=4,
                use_norm_weight=True,
                rms_eps=1.0e-6,
                hc_pre_eps=2.0e-6,
                hc_sinkhorn_eps=3.0e-6,
                hc_post_mult_value=0.5,
                sinkhorn_repeat=3,
                norm_eps=1.0e-5,
            ),
        ),
        (MhcPostTileLangKernel(), dict(hidden_size=4096, hc_mult=4)),
        (
            MhcFusedTileLangKernel(),
            dict(vllm_config=config, hidden_size=4096, hc_mult=4),
        ),
        (
            HcHeadFusedTileLangKernel(),
            dict(
                hidden_size=4096,
                hc_mult=4,
                rms_eps=1.0e-6,
                hc_eps=2.0e-6,
            ),
        ),
    )

    for owner, kwargs in owners_and_keys:
        keys = owner.get_warmup_keys(**kwargs)
        assert keys
        assert {key.launch_pdl for key in keys} == {False, True}
