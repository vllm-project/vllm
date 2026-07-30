# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

_PrenormGemmImpl = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], None
]


def _can_use_cutedsl_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    n_splits: int,
) -> bool:
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_cutedsl()
    ):
        return False

    from vllm.model_executor.kernels.mhc.cutedsl import can_use_hc_prenorm_gemm

    return can_use_hc_prenorm_gemm(x, fn, n_splits)


def _get_deep_gemm_impl() -> _PrenormGemmImpl:
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    return tf32_hc_prenorm_gemm


def _get_cutedsl_gemm_impl() -> _PrenormGemmImpl:
    from vllm.model_executor.kernels.mhc.cutedsl import run_hc_prenorm_gemm

    return run_hc_prenorm_gemm


def _tilelang_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    hidden_size: int,
    hc_mult: int,
    tile_n: int = 12,
    n_thr: int = 512,
    n_splits: int = 1,
) -> None:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        hc_prenorm_gemm_block_m_tilelang,
        hc_prenorm_gemm_tilelang,
    )

    assert out.shape[0] == n_splits
    assert sqrsum.shape[0] == n_splits
    assert x.shape[1] == hc_mult * hidden_size
    assert x.shape[1] % n_splits == 0
    assert (x.shape[1] // n_splits) % n_thr == 0
    use_default_config = tile_n == 12 and n_thr == 512
    if n_splits == 1 and use_default_config and x.shape[0] >= 1024:
        hc_prenorm_gemm_block_m_tilelang(
            x,
            fn,
            out,
            sqrsum,
            hidden_size,
            hc_mult,
            fn.shape[0],
            n_thr,
            tile_n,
            2,
        )
        return
    if (
        n_splits == 1
        and use_default_config
        and x.shape[0] < 128
        and x.shape[1] % 1024 == 0
    ):
        hc_prenorm_gemm_tilelang(
            x,
            fn,
            out,
            sqrsum,
            hidden_size,
            hc_mult,
            fn.shape[0],
            1024,
            4,
            n_splits,
        )
        return
    hc_prenorm_gemm_tilelang(
        x,
        fn,
        out,
        sqrsum,
        hidden_size,
        hc_mult,
        fn.shape[0],
        n_thr,
        tile_n,
        n_splits,
    )


def _get_tilelang_impl(hidden_size: int, hc_mult: int) -> _PrenormGemmImpl:
    def run(
        x: torch.Tensor,
        fn: torch.Tensor,
        out: torch.Tensor,
        sqrsum: torch.Tensor,
        n_splits: int,
    ) -> None:
        _tilelang_hc_prenorm_gemm(
            x,
            fn,
            out,
            sqrsum,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            n_splits=n_splits,
        )

    return run


class HCPrenormGemm:
    def __init__(
        self,
        x: torch.Tensor,
        fn: torch.Tensor,
        hidden_size: int,
        hc_mult: int,
        preferred_n_splits: int,
    ) -> None:
        from vllm.utils.deep_gemm import is_deep_gemm_supported

        if is_deep_gemm_supported():
            self.n_splits = preferred_n_splits
            self._impl = _get_deep_gemm_impl()
        elif _can_use_cutedsl_hc_prenorm_gemm(x, fn, preferred_n_splits):
            self.n_splits = preferred_n_splits
            self._impl = _get_cutedsl_gemm_impl()
        else:
            self.n_splits = 1
            self._impl = _get_tilelang_impl(hidden_size, hc_mult)

    def __call__(
        self,
        x: torch.Tensor,
        fn: torch.Tensor,
        out: torch.Tensor,
        sqrsum: torch.Tensor,
    ) -> None:
        self._impl(x, fn, out, sqrsum, self.n_splits)
