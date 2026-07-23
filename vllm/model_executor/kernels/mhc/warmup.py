# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mHC TileLang kernel warmup wrappers (VllmJitKernel contract).

Each wrapper mirrors the runtime dispatch logic of one mHC TileLang op and
exposes the compile-key space that should be pre-compiled at engine startup.
Warmup logic stays next to the kernel definitions, following the kernel-owned
principle of RFC #47456 / PR #47451.

The TileLang kernels treat ``num_tokens`` as a dynamic dimension, so the same
compiled specialization covers many token sizes. What triggers re-compilation
is the static parameters derived from ``num_tokens`` via the runtime dispatch
heuristics (``n_splits``, ``tile_n``, ``use_small_fma``, ``use_norm_weight``,
``use_deep_gemm``). The wrappers let the AST tracer in
:mod:`vllm.model_executor.warmup.jit_warmup` expand ``WarmupIntRange`` and
deduplicate to the actual compile-key set.

``compile()`` calls ``.compile()`` on the underlying ``@tilelang.jit`` kernels
(not ``torch.ops.vllm.*`` ops and not direct ``__call__``): the op wrappers
recompute ``n_splits`` / ``tile_n`` from ``num_tokens`` and would override the
key's static params, and ``__call__`` would launch the kernel. ``.compile()``
is compile-only — it inspects just tensor metadata via the TIR builder — so we
pass :class:`TileLangWarmupTensor` fake tensors and allocate no GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel, WarmupIntRange
from vllm.model_executor.warmup.jit_warmup_tilelang_helper import (
    TileLangWarmupTensor,
)
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _is_deep_gemm_supported() -> bool:
    """Lazy import to avoid CUDA init at module load time."""
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    return is_deep_gemm_supported()


def _compute_n_splits(num_tokens: int, hc_hidden_size: int) -> int:
    """Mirror of mhc_fused_post_pre_tilelang's deep_gemm split heuristic.

    Wraps ``compute_num_split`` so the AST tracer can call it during dispatch
    expansion.
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split

    block_k = 64
    block_m = 64
    return compute_num_split(block_k, hc_hidden_size, cdiv(num_tokens, block_m))


def _fake(
    dtype: torch.dtype,
    *shape: int,
) -> TileLangWarmupTensor:
    """Build a compile-only fake tensor (no GPU memory allocated)."""
    return TileLangWarmupTensor(dtype=dtype, shape=tuple(shape))


# =============================================================================
# 1. MhcPreKernel — first-layer path (mhc_pre + mhc_post)
# =============================================================================


class MhcPreKernel(VllmJitKernel["MhcPreKernel.CompileKey"]):
    """Warmup for the first-layer mHC path.

    Dispatch:
    - ``use_deep_gemm``: True → ``n_splits = compute_num_split(...)``; else 1
    - ``use_norm_weight``: True (NVIDIA) → ``mhc_pre_big_fuse_with_norm_tilelang``;
      False (AMD/XPU) → ``mhc_pre_big_fuse_tilelang``
    """

    @dataclass(frozen=True)
    class CompileKey:
        hidden_size: int
        hc_mult: int
        n_splits: int
        use_norm_weight: bool
        use_deep_gemm: bool

    def dispatch(  # type: ignore[override]
        self,
        *,
        num_tokens: int,
        hidden_size: int,
        hc_mult: int,
        use_norm_weight: bool,
        use_deep_gemm: bool,
    ) -> CompileKey:
        # Ternary form required by the AST tracer (no if/else stmt).
        hc_hidden_size = hc_mult * hidden_size
        n_splits = _compute_n_splits(num_tokens, hc_hidden_size) if use_deep_gemm else 1
        return self.CompileKey(
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            n_splits=n_splits,
            use_norm_weight=use_norm_weight,
            use_deep_gemm=use_deep_gemm,
        )

    def get_warmup_keys(
        self,
        vllm_config: VllmConfig,
        *,
        hidden_size: int,
        hc_mult: int,
        use_norm_weight: bool,
    ) -> list[CompileKey]:
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if max_tokens <= 0:
            return []
        use_deep_gemm = _is_deep_gemm_supported()
        return self._trace_dispatch(self.dispatch)(
            num_tokens=WarmupIntRange(1, max_tokens + 1),
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            use_norm_weight=use_norm_weight,
            use_deep_gemm=use_deep_gemm,
        )

    def compile(self, compile_key: CompileKey) -> None:
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            mhc_post_tilelang as _mhc_post_kernel,
        )
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            mhc_pre_big_fuse_tilelang,
            mhc_pre_big_fuse_with_norm_tilelang,
        )

        hidden_size = compile_key.hidden_size
        hc_mult = compile_key.hc_mult
        n_splits = compile_key.n_splits
        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
        num_tokens = 1  # dynamic dim; smallest valid value

        gemm_out_mul = _fake(torch.float32, n_splits, num_tokens, hc_mult3)
        gemm_out_sqrsum = _fake(torch.float32, n_splits, num_tokens)
        hc_scale = _fake(torch.float32, 3)
        hc_base = _fake(torch.float32, hc_mult3)
        residual = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
        post_mix = _fake(torch.float32, num_tokens, hc_mult)
        # comb_mix for mhc_pre_big_fuse is 2D; mhc_fused/mhc_post expect 3D.
        comb_mix = _fake(torch.float32, num_tokens, hc_mult * hc_mult)
        layer_input = _fake(torch.bfloat16, num_tokens, hidden_size)

        if compile_key.use_norm_weight:
            norm_weight = _fake(torch.bfloat16, hidden_size)
            mhc_pre_big_fuse_with_norm_tilelang.compile(
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                residual,
                post_mix,
                comb_mix,
                layer_input,
                norm_weight,
                hidden_size,
                1e-6,
                1e-6,
                1e-6,
                1.0,
                1,
                1e-6,
                n_splits,
                hc_mult,
            )
        else:
            mhc_pre_big_fuse_tilelang.compile(
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                residual,
                post_mix,
                comb_mix,
                layer_input,
                hidden_size,
                1e-6,
                1e-6,
                1e-6,
                1.0,
                1,
                n_splits,
                hc_mult,
            )

        # mhc_post: dispatch independent of n_splits; one compile per
        # (hc_mult, hidden_size) suffices but calling per key is cheap.
        a = _fake(torch.float32, num_tokens, hc_mult, hc_mult)
        b = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
        c = _fake(torch.float32, num_tokens, hc_mult)
        d = _fake(torch.bfloat16, num_tokens, hidden_size)
        x = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
        _mhc_post_kernel.compile(a, b, c, d, x, hc_mult, hidden_size)


# =============================================================================
# 2. MhcFusedPostPreKernel — second-layer-and-after path
# =============================================================================


class MhcFusedPostPreKernel(VllmJitKernel["MhcFusedPostPreKernel.CompileKey"]):
    """Warmup for ``torch.ops.vllm.mhc_fused_post_pre_tilelang``.

    Runtime dispatch (from ``vllm/model_executor/kernels/mhc/tilelang.py``):

    - ``use_small_fma`` (num_tokens <= 16):
      - ``tile_n = 2 if num_tokens < 8 else 3``
      - ``n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4``
      - calls ``mhc_fused_tilelang`` (single fused kernel)
    - else:
      - ``n_splits = compute_num_split(...)`` if ``use_deep_gemm`` else 1
      - calls ``mhc_post_tilelang`` + GEMM + ``mhc_pre_big_fuse[_with_norm]_tilelang``
    - ``use_norm_weight`` selects the norm-fused (NVIDIA) variant of pre
    """

    @dataclass(frozen=True)
    class CompileKey:
        hidden_size: int
        hc_mult: int
        n_splits: int
        tile_n: int  # only meaningful when use_small_fma
        use_small_fma: bool
        use_norm_weight: bool
        use_deep_gemm: bool

    def dispatch(  # type: ignore[override]
        self,
        *,
        num_tokens: int,
        hidden_size: int,
        hc_mult: int,
        use_norm_weight: bool,
        use_deep_gemm: bool,
    ) -> CompileKey:
        # Ternary form required by the AST tracer.
        use_small_fma = num_tokens <= 16
        tile_n = 2 if num_tokens < 8 else 3
        hc_hidden_size = hc_mult * hidden_size
        n_splits_small = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
        n_splits_big = (
            _compute_n_splits(num_tokens, hc_hidden_size) if use_deep_gemm else 1
        )
        n_splits = n_splits_small if use_small_fma else n_splits_big
        return self.CompileKey(
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            n_splits=n_splits,
            tile_n=tile_n,
            use_small_fma=use_small_fma,
            use_norm_weight=use_norm_weight,
            use_deep_gemm=use_deep_gemm,
        )

    def get_warmup_keys(
        self,
        vllm_config: VllmConfig,
        *,
        hidden_size: int,
        hc_mult: int,
        use_norm_weight: bool,
    ) -> list[CompileKey]:
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if max_tokens <= 0:
            return []
        use_deep_gemm = _is_deep_gemm_supported()
        return self._trace_dispatch(self.dispatch)(
            num_tokens=WarmupIntRange(1, max_tokens + 1),
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            use_norm_weight=use_norm_weight,
            use_deep_gemm=use_deep_gemm,
        )

    def compile(self, compile_key: CompileKey) -> None:
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            mhc_fused_tilelang,
            mhc_pre_big_fuse_tilelang,
            mhc_pre_big_fuse_with_norm_tilelang,
        )
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            mhc_post_tilelang as _mhc_post_kernel,
        )

        hidden_size = compile_key.hidden_size
        hc_mult = compile_key.hc_mult
        n_splits = compile_key.n_splits
        tile_n = compile_key.tile_n
        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
        num_tokens = 1  # dynamic dim; smallest valid value

        if compile_key.use_small_fma:
            comb_mix = _fake(torch.float32, num_tokens, hc_mult, hc_mult)
            residual_in = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
            post_mix = _fake(torch.float32, num_tokens, hc_mult)
            x_in = _fake(torch.bfloat16, num_tokens, hidden_size)
            weight_t = _fake(torch.float32, hc_mult3, hc_mult, hidden_size)
            yp_out = _fake(torch.float32, n_splits, num_tokens, hc_mult3)
            rp_out = _fake(torch.float32, n_splits, num_tokens)
            residual_out = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
            # NOTE: the op-level caller passes n_splits=..., but the underlying
            # kernel's parameter is split_k. May warrant an upstream fix.
            mhc_fused_tilelang.compile(
                comb_mix,
                residual_in,
                post_mix,
                x_in,
                weight_t,
                yp_out,
                rp_out,
                residual_out,
                hc_mult,
                hidden_size,
                hc_mult3,
                tile_n=tile_n,
                split_k=n_splits,
            )
        else:
            # mhc_post
            a = _fake(torch.float32, num_tokens, hc_mult, hc_mult)
            b = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
            c = _fake(torch.float32, num_tokens, hc_mult)
            d = _fake(torch.bfloat16, num_tokens, hidden_size)
            x = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
            _mhc_post_kernel.compile(a, b, c, d, x, hc_mult, hidden_size)

            # mhc_pre_big_fuse[_with_norm]
            gemm_out_mul = _fake(torch.float32, n_splits, num_tokens, hc_mult3)
            gemm_out_sqrsum = _fake(torch.float32, n_splits, num_tokens)
            hc_scale = _fake(torch.float32, 3)
            hc_base = _fake(torch.float32, hc_mult3)
            residual = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
            post_mix = _fake(torch.float32, num_tokens, hc_mult)
            comb_mix = _fake(torch.float32, num_tokens, hc_mult * hc_mult)
            layer_input = _fake(torch.bfloat16, num_tokens, hidden_size)

            if compile_key.use_norm_weight:
                norm_weight = _fake(torch.bfloat16, hidden_size)
                mhc_pre_big_fuse_with_norm_tilelang.compile(
                    gemm_out_mul,
                    gemm_out_sqrsum,
                    hc_scale,
                    hc_base,
                    residual,
                    post_mix,
                    comb_mix,
                    layer_input,
                    norm_weight,
                    hidden_size,
                    1e-6,
                    1e-6,
                    1e-6,
                    1.0,
                    1,
                    1e-6,
                    n_splits,
                    hc_mult,
                )
            else:
                mhc_pre_big_fuse_tilelang.compile(
                    gemm_out_mul,
                    gemm_out_sqrsum,
                    hc_scale,
                    hc_base,
                    residual,
                    post_mix,
                    comb_mix,
                    layer_input,
                    hidden_size,
                    1e-6,
                    1e-6,
                    1e-6,
                    1.0,
                    1,
                    n_splits,
                    hc_mult,
                )

            # NOTE: GEMM step (tf32_hc_prenorm_gemm / _tilelang_hc_prenorm_gemm)
            # is not warmed here — its compile key is independent of num_tokens.


# =============================================================================
# 3. HcHeadFusedKernel — hc_head_fused_kernel_tilelang op
# =============================================================================


class HcHeadFusedKernel(VllmJitKernel["HcHeadFusedKernel.CompileKey"]):
    """Warmup for ``torch.ops.vllm.hc_head_fused_kernel_tilelang``.

    The underlying kernel has no num_tokens-driven branches, so the entire
    token range deduplicates to a single CompileKey per
    (hidden_size, hc_mult, use_norm_weight).
    """

    @dataclass(frozen=True)
    class CompileKey:
        hidden_size: int
        hc_mult: int
        use_norm_weight: bool

    def dispatch(  # type: ignore[override]
        self,
        *,
        num_tokens: int,
        hidden_size: int,
        hc_mult: int,
        use_norm_weight: bool,
    ) -> CompileKey:
        # num_tokens is intentionally absent: it's a dynamic dim and does
        # not trigger re-compilation.
        return self.CompileKey(
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            use_norm_weight=use_norm_weight,
        )

    def get_warmup_keys(
        self,
        vllm_config: VllmConfig,
        *,
        hidden_size: int,
        hc_mult: int,
        use_norm_weight: bool,
    ) -> list[CompileKey]:
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if max_tokens <= 0:
            return []
        # WarmupIntRange collapses to 1 key since dispatch ignores num_tokens.
        return self._trace_dispatch(self.dispatch)(
            num_tokens=WarmupIntRange(1, max_tokens + 1),
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            use_norm_weight=use_norm_weight,
        )

    def compile(self, compile_key: CompileKey) -> None:
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            hc_head_fuse_tilelang,
        )

        hidden_size = compile_key.hidden_size
        hc_mult = compile_key.hc_mult
        num_tokens = 1  # dynamic dim; smallest valid value

        hs = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
        fn = _fake(torch.float32, hc_mult, hc_mult * hidden_size)
        # hc_head_fuse_tilelang expects hc_scale shape (1,) and hc_base (hc_mult,)
        hc_scale = _fake(torch.float32, 1)
        hc_base = _fake(torch.float32, hc_mult)
        out = _fake(torch.bfloat16, num_tokens, hidden_size)

        hc_head_fuse_tilelang.compile(
            hs,
            fn,
            hc_scale,
            hc_base,
            out,
            hidden_size,
            1e-6,
            1e-6,
            hc_mult,
        )


MHC_PRE_KERNEL = MhcPreKernel()
MHC_FUSED_POST_PRE_KERNEL = MhcFusedPostPreKernel()
HC_HEAD_FUSED_KERNEL = HcHeadFusedKernel()
