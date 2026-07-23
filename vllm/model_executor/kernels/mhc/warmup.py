# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mHC TileLang kernel warmup wrappers (VllmJitKernel contract).

Each wrapper exposes the compile-key space of one mHC TileLang op so that
all specializations the runtime may invoke are pre-compiled at engine
startup.  Warmup logic stays next to the kernel definitions, following the
kernel-owned principle of RFC #47456 / PR #47451.

The TileLang kernels treat ``num_tokens`` as a dynamic dimension, so the
same compiled specialization covers many token sizes.  What triggers
re-compilation are the static parameters derived from ``num_tokens`` via
:func:`~vllm.model_executor.kernels.mhc.tilelang_kernels.compute_mhc_dispatch`
(``n_splits``, ``tile_n``, ``use_small_fma``).  The wrappers let the AST
tracer in :mod:`vllm.model_executor.warmup.jit_warmup` expand
``WarmupIntRange`` and deduplicate to the actual compile-key set.

Model-level constants that are also part of the cache_key (``hc_post_alpha``,
``hc_sinkhorn_iters``, various epsilons) are collected once into
:class:`MhcKernelConstants` so the warmup key matches the runtime key
exactly.

``compile()`` calls ``.compile()`` on the underlying ``@tilelang.jit``
kernels — not ``torch.ops.vllm.*`` ops and not direct ``__call__`` — so it
is compile-only (no kernel launch, no GPU memory) using
:class:`TileLangWarmupTensor` fake tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel, WarmupIntRange
from vllm.model_executor.warmup.jit_warmup_tilelang_helper import (
    TileLangWarmupTensor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class MhcKernelConstants:
    """Model-level constants that are part of the TileLang cache_key.

    These values do not vary with ``num_tokens`` (unlike ``n_splits``), so
    they are not part of the dispatch / ``WarmupIntRange`` expansion.  They
    are read once from the model layer and threaded into every ``compile()``
    call so the warmup cache_key matches the runtime cache_key exactly.

    Mapping to runtime sources (DeepseekV4DecoderLayer):
        hc_post_mult_value ← layer.hc_post_alpha          (hardcoded 2.0)
        sinkhorn_repeat    ← layer.hc_sinkhorn_iters      (from config)
        rms_eps            ← layer.rms_norm_eps            (from config)
        hc_pre_eps         ← layer.hc_eps                  (from config)
        hc_sinkhorn_eps    ← layer.hc_eps                  (same as hc_pre_eps)
        norm_eps           ← layer.attn_norm.variance_epsilon  (= rms_norm_eps)
    """

    hc_post_mult_value: float
    sinkhorn_repeat: int
    rms_eps: float
    hc_pre_eps: float
    hc_sinkhorn_eps: float
    norm_eps: float


def _is_deep_gemm_supported() -> bool:
    """Lazy import to avoid CUDA init at module load time."""
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    return is_deep_gemm_supported()


def _compute_mhc_dispatch(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    use_deep_gemm: bool,
    *,
    is_broadcast: bool = False,
    is_fused: bool = False,
):
    """Thin wrapper around ``compute_mhc_dispatch`` for the AST tracer.

    The AST tracer in ``jit_warmup.py`` can call free functions referenced
    in dispatch bodies via ``__globals__``, but the function must be visible
    in the dispatch function's module-level scope.  This wrapper re-exports
    the shared dispatch logic from ``tilelang_kernels`` so both the runtime
    ops and the warmup wrappers use exactly the same heuristic.
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_mhc_dispatch,
    )

    return compute_mhc_dispatch(
        num_tokens,
        hidden_size,
        hc_mult,
        use_deep_gemm,
        is_broadcast=is_broadcast,
        is_fused=is_fused,
    )


def _fake(
    dtype: torch.dtype,
    *shape: int,
) -> TileLangWarmupTensor:
    """Build a compile-only fake tensor (no GPU memory allocated)."""
    return TileLangWarmupTensor(dtype=dtype, shape=tuple(shape))


def _compile_and_cache(jit_impl, *args, **kwargs) -> None:
    """Compile a TileLang kernel and populate its per-instance ``_kernel_cache``.

    ``JITImpl.compile()`` populates the global ``KernelCache`` but NOT the
    per-instance ``_kernel_cache``.  At runtime, ``JITImpl.__call__`` checks
    ``_kernel_cache`` first and logs a JIT warning on miss — even though the
    global cache has the kernel.  This helper closes that gap by calling
    ``compile()`` and then storing the result in ``_kernel_cache`` using the
    same key that ``__call__`` would compute via ``func.parse_args()``.
    """
    kernel = jit_impl.compile(*args, **kwargs)
    key, _ = jit_impl.func.parse_args(*args, **kwargs)
    jit_impl._kernel_cache[key] = kernel


def _compile_mhc_post(hidden_size: int, hc_mult: int) -> None:
    """Compile ``mhc_post_tilelang`` — shared by MhcPreKernel and MhcFusedPostPreKernel.

    Its cache_key depends only on ``(hc_mult, hidden_size)``, so repeated
    calls with the same pair are no-ops (the per-instance ``_kernel_cache``
    already has the entry).
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_post_tilelang as _mhc_post_kernel,
    )

    num_tokens = 1  # dynamic dim; smallest valid value
    a = _fake(torch.float32, num_tokens, hc_mult, hc_mult)
    b = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
    c_t = _fake(torch.float32, num_tokens, hc_mult)
    d = _fake(torch.bfloat16, num_tokens, hidden_size)
    x = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
    _compile_and_cache(_mhc_post_kernel, a, b, c_t, d, x, hc_mult, hidden_size)


# =============================================================================
# 1. MhcPreKernel — first-layer path (mhc_pre + mhc_post)
# =============================================================================


class MhcPreKernel(VllmJitKernel["MhcPreKernel.CompileKey"]):
    """Warmup for the first-layer mHC pre kernel.

    Covers both the standard 3D-residual path (``mhc_pre_big_fuse_with_norm``
    on NVIDIA / ``mhc_pre_big_fuse`` on AMD) and the 2D-residual broadcast
    path (``mhc_pre_big_fuse_broadcast_with_norm``) used when the first
    decoder layer receives a raw embedding (``x.dim() == 2``).

    Derived parameters (``n_splits``) are computed by the shared
    :func:`compute_mhc_dispatch`; model-level constants come from
    :class:`MhcKernelConstants`.
    """

    @dataclass(frozen=True)
    class CompileKey:
        hidden_size: int
        hc_mult: int
        n_splits: int
        use_norm_weight: bool
        use_deep_gemm: bool
        is_broadcast: bool = False

    def dispatch(  # type: ignore[override]
        self,
        *,
        num_tokens: int,
        hidden_size: int,
        hc_mult: int,
        use_norm_weight: bool,
        use_deep_gemm: bool,
        is_broadcast: bool,
    ) -> CompileKey:
        d = _compute_mhc_dispatch(
            num_tokens,
            hidden_size,
            hc_mult,
            use_deep_gemm,
            is_broadcast=is_broadcast,
        )
        return self.CompileKey(
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            n_splits=d.n_splits,
            use_norm_weight=use_norm_weight,
            use_deep_gemm=use_deep_gemm,
            is_broadcast=is_broadcast,
        )

    def get_warmup_keys(
        self,
        vllm_config: VllmConfig,
        *,
        hidden_size: int,
        hc_mult: int,
        use_norm_weight: bool,
        is_broadcast_values: list[bool],
        constants: MhcKernelConstants,
    ) -> list[CompileKey]:
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if max_tokens <= 0:
            return []
        use_deep_gemm = _is_deep_gemm_supported()
        self._constants = constants
        keys = self._trace_dispatch(self.dispatch)(
            num_tokens=WarmupIntRange(1, max_tokens + 1),
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            use_norm_weight=use_norm_weight,
            use_deep_gemm=use_deep_gemm,
            is_broadcast=is_broadcast_values,
        )
        logger.info(
            "MhcPreKernel: total=%d "
            "(use_norm_weight=%s, use_deep_gemm=%s, is_broadcast=%s, "
            "constants=%s)",
            len(keys),
            use_norm_weight,
            use_deep_gemm,
            is_broadcast_values,
            constants,
        )
        return keys

    def compile(self, compile_key: CompileKey) -> None:
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            mhc_pre_big_fuse_broadcast_with_norm_tilelang,
            mhc_pre_big_fuse_tilelang,
            mhc_pre_big_fuse_with_norm_tilelang,
        )

        hidden_size = compile_key.hidden_size
        hc_mult = compile_key.hc_mult
        n_splits = compile_key.n_splits
        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
        num_tokens = 1  # dynamic dim; smallest valid value
        c = self._constants

        gemm_out_mul = _fake(torch.float32, n_splits, num_tokens, hc_mult3)
        gemm_out_sqrsum = _fake(torch.float32, n_splits, num_tokens)
        hc_scale = _fake(torch.float32, 3)
        hc_base = _fake(torch.float32, hc_mult3)
        post_mix = _fake(torch.float32, num_tokens, hc_mult)
        # comb_mix for mhc_pre_big_fuse is 2D; mhc_fused/mhc_post expect 3D.
        comb_mix = _fake(torch.float32, num_tokens, hc_mult * hc_mult)
        layer_input = _fake(torch.bfloat16, num_tokens, hidden_size)

        if compile_key.is_broadcast:
            # Broadcast path: residual is 2D (num_tokens, hidden_size),
            # and the kernel takes an extra residual_out (3D) output tensor.
            residual = _fake(torch.bfloat16, num_tokens, hidden_size)
            residual_out = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
            norm_weight = _fake(torch.bfloat16, hidden_size)
            _compile_and_cache(
                mhc_pre_big_fuse_broadcast_with_norm_tilelang,
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                residual,
                residual_out,
                post_mix,
                comb_mix,
                layer_input,
                norm_weight,
                hidden_size,
                c.rms_eps,
                c.hc_pre_eps,
                c.hc_sinkhorn_eps,
                c.hc_post_mult_value,
                c.sinkhorn_repeat,
                c.norm_eps,
                n_splits,
                hc_mult,
            )
            # mhc_post is compiled by the non-broadcast keys; skip for broadcast.
            return

        # Non-broadcast path: residual is 3D (num_tokens, hc_mult, hidden_size)
        residual = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)

        if compile_key.use_norm_weight:
            norm_weight = _fake(torch.bfloat16, hidden_size)
            _compile_and_cache(
                mhc_pre_big_fuse_with_norm_tilelang,
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
                c.rms_eps,
                c.hc_pre_eps,
                c.hc_sinkhorn_eps,
                c.hc_post_mult_value,
                c.sinkhorn_repeat,
                c.norm_eps,
                n_splits,
                hc_mult,
            )
        else:
            _compile_and_cache(
                mhc_pre_big_fuse_tilelang,
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                residual,
                post_mix,
                comb_mix,
                layer_input,
                hidden_size,
                c.rms_eps,
                c.hc_pre_eps,
                c.hc_sinkhorn_eps,
                c.hc_post_mult_value,
                c.sinkhorn_repeat,
                n_splits,
                hc_mult,
            )

        # mhc_post: cache_key depends only on (hc_mult, hidden_size).
        _compile_mhc_post(hidden_size, hc_mult)


# =============================================================================
# 2. MhcFusedPostPreKernel — second-layer-and-after path
# =============================================================================


class MhcFusedPostPreKernel(VllmJitKernel["MhcFusedPostPreKernel.CompileKey"]):
    """Warmup for ``torch.ops.vllm.mhc_fused_post_pre_tilelang``.

    Used by every decoder layer after the first.  Derived parameters
    (``n_splits``, ``tile_n``, ``use_small_fma``) are computed by the shared
    :func:`compute_mhc_dispatch(is_fused=True)`; model-level constants come
    from :class:`MhcKernelConstants`.
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
        d = _compute_mhc_dispatch(
            num_tokens, hidden_size, hc_mult, use_deep_gemm, is_fused=True
        )
        return self.CompileKey(
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            n_splits=d.n_splits,
            tile_n=d.tile_n,
            use_small_fma=d.use_small_fma,
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
        constants: MhcKernelConstants,
    ) -> list[CompileKey]:
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if max_tokens <= 0:
            return []
        use_deep_gemm = _is_deep_gemm_supported()
        self._constants = constants
        keys = self._trace_dispatch(self.dispatch)(
            num_tokens=WarmupIntRange(1, max_tokens + 1),
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            use_norm_weight=use_norm_weight,
            use_deep_gemm=use_deep_gemm,
        )
        logger.info(
            "MhcFusedPostPreKernel: total=%d (use_norm_weight=%s, "
            "use_deep_gemm=%s, constants=%s)",
            len(keys),
            use_norm_weight,
            use_deep_gemm,
            constants,
        )
        return keys

    def compile(self, compile_key: CompileKey) -> None:
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            mhc_fused_tilelang,
            mhc_pre_big_fuse_tilelang,
            mhc_pre_big_fuse_with_norm_tilelang,
        )

        hidden_size = compile_key.hidden_size
        hc_mult = compile_key.hc_mult
        n_splits = compile_key.n_splits
        tile_n = compile_key.tile_n
        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
        num_tokens = 1  # dynamic dim; smallest valid value
        c = self._constants

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
            _compile_and_cache(
                mhc_fused_tilelang,
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
            _compile_mhc_post(hidden_size, hc_mult)

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
                _compile_and_cache(
                    mhc_pre_big_fuse_with_norm_tilelang,
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
                    c.rms_eps,
                    c.hc_pre_eps,
                    c.hc_sinkhorn_eps,
                    c.hc_post_mult_value,
                    c.sinkhorn_repeat,
                    c.norm_eps,
                    n_splits,
                    hc_mult,
                )
            else:
                _compile_and_cache(
                    mhc_pre_big_fuse_tilelang,
                    gemm_out_mul,
                    gemm_out_sqrsum,
                    hc_scale,
                    hc_base,
                    residual,
                    post_mix,
                    comb_mix,
                    layer_input,
                    hidden_size,
                    c.rms_eps,
                    c.hc_pre_eps,
                    c.hc_sinkhorn_eps,
                    c.hc_post_mult_value,
                    c.sinkhorn_repeat,
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
    (hidden_size, hc_mult).
    """

    @dataclass(frozen=True)
    class CompileKey:
        hidden_size: int
        hc_mult: int

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
        )

    def get_warmup_keys(
        self,
        vllm_config: VllmConfig,
        *,
        hidden_size: int,
        hc_mult: int,
        use_norm_weight: bool,
        constants: MhcKernelConstants,
    ) -> list[CompileKey]:
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if max_tokens <= 0:
            return []
        self._constants = constants
        # WarmupIntRange collapses to 1 key since dispatch ignores num_tokens.
        keys = self._trace_dispatch(self.dispatch)(
            num_tokens=WarmupIntRange(1, max_tokens + 1),
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            use_norm_weight=use_norm_weight,
        )
        logger.info(
            "HcHeadFusedKernel: total=%d (use_norm_weight=%s, constants=%s)",
            len(keys),
            use_norm_weight,
            constants,
        )
        return keys

    def compile(self, compile_key: CompileKey) -> None:
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            hc_head_fuse_tilelang,
        )

        hidden_size = compile_key.hidden_size
        hc_mult = compile_key.hc_mult
        num_tokens = 1  # dynamic dim; smallest valid value
        c = self._constants

        hs = _fake(torch.bfloat16, num_tokens, hc_mult, hidden_size)
        fn = _fake(torch.float32, hc_mult, hc_mult * hidden_size)
        # hc_head_fuse_tilelang expects hc_scale shape (1,) and hc_base (hc_mult,)
        hc_scale = _fake(torch.float32, 1)
        hc_base = _fake(torch.float32, hc_mult)
        out = _fake(torch.bfloat16, num_tokens, hidden_size)

        _compile_and_cache(
            hc_head_fuse_tilelang,
            hs,
            fn,
            hc_scale,
            hc_base,
            out,
            hidden_size,
            c.rms_eps,
            c.hc_pre_eps,
            hc_mult,
        )


MHC_PRE_KERNEL = MhcPreKernel()
MHC_FUSED_POST_PRE_KERNEL = MhcFusedPostPreKernel()
HC_HEAD_FUSED_KERNEL = HcHeadFusedKernel()
