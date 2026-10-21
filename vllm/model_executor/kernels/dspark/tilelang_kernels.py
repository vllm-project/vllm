# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw TileLang kernel for DSpark confidence head.

Computes ``sigmoid((x0 ; x1) @ W / temp)`` in a single kernel launch,
where ``(x0 ; x1)`` is the logical concatenation of ``x0`` (hidden) and
``x1`` (markov_embed) along the feature dim.

Unlike the DeepInfer reference implementation (`dflash.py:12-72`), weights
are FP32 (not BF16), matching vLLM's ``ReplicatedLinear(params_dtype=float32)``
for the confidence head in
``vllm/model_executor/models/qwen3_dspark.py:112-137``.

Layout follows ``vllm/model_executor/kernels/mhc/tilelang_kernels.py``: we
use TileLang's "implicit signature" style (top-level ``T.Kernel`` + variable
annotations) rather than ``@T.prim_func`` on a nested function.
"""

from __future__ import annotations

from math import gcd

import tilelang
import tilelang.language as T
import torch

from vllm.platforms import current_platform


def _enable_pdl() -> bool:
    """PDL (programmatic dependent launch) only makes sense on SM90+ CUDA."""
    if not torch.cuda.is_available():
        return False
    try:
        if not current_platform.is_cuda():
            return False
        fn = getattr(current_platform, "is_arch_support_pdl", None)
        return bool(fn and fn())
    except Exception:
        return False


_PASS_CONFIGS: dict = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


@tilelang.jit(out_idx=None, target="cuda", pass_configs=_PASS_CONFIGS)
def confidence_head_fused_tilelang(
    x0,
    x1,
    weight,
    out,
    temp: float,
    n_embed: int,
    markov_rank: int,
) -> None:
    """Fused confidence head kernel.

    ``n_embed`` and ``markov_rank`` are static specialization parameters.
    Tensor parameters are declared without annotations in the signature
    (TileLang's "implicit signature" style — see mhc/tilelang_kernels.py);
    their types are re-declared inside below.
    """
    n = T.dynamic("n")
    threads = 128
    B0 = gcd(1024, n_embed)
    B1 = max(markov_rank, 1)
    has_markov = markov_rank > 0
    enable_pdl = _enable_pdl()

    # Tensor type re-declarations (TileLang resolves these at parse time):
    x0: T.Tensor[[n, n_embed], T.bfloat16]  # type: ignore[no-redef, valid-type]
    x1: T.Tensor[[n, B1], T.bfloat16]  # type: ignore[no-redef, valid-type]
    weight: T.Tensor[[n_embed + markov_rank], T.float32]  # type: ignore[no-redef, valid-type]
    out: T.Tensor[[n], T.float32]  # type: ignore[no-redef, valid-type]

    with T.Kernel(n, threads=threads) as bx:
        sum_frag = T.alloc_fragment((1,), T.float32)
        sum_frag[0] = 0.0

        if enable_pdl and not has_markov:
            T.pdl_sync()

        # GEMV #1: hidden (bf16) x weight[0 : n_embed] (fp32) -> fp32
        for i in T.Pipelined(T.ceildiv(n_embed, B0)):
            w_frag = T.alloc_fragment((B0,), T.float32)
            x_frag = T.alloc_fragment((B0,), T.bfloat16)
            T.copy(weight[0 + i * B0], w_frag)
            T.copy(x0[bx, i * B0], x_frag)
            acc0 = T.alloc_fragment((B0,), T.float32)
            T.fill(acc0, 0.0)
            for k in T.Parallel(B0):
                contrib = T.cast(x_frag[k], T.float32) * w_frag[k]
                acc0[k] += T.Select(
                    k + i * B0 < n_embed, contrib, T.cast(0.0, T.float32)
                )
            T.reduce_sum(acc0, sum_frag, clear=False)

        # GEMV #2: markov (bf16) x weight[n_embed : n_embed + r] (fp32) -> fp32
        if has_markov:
            for i in T.Pipelined(T.ceildiv(markov_rank, B1)):
                w_frag = T.alloc_fragment((B1,), T.float32)
                x_frag = T.alloc_fragment((B1,), T.bfloat16)
                T.copy(weight[n_embed + i * B1], w_frag)
                T.copy(x1[bx, i * B1], x_frag)
                acc1 = T.alloc_fragment((B1,), T.float32)
                T.fill(acc1, 0.0)
                for k in T.Parallel(B1):
                    contrib = T.cast(x_frag[k], T.float32) * w_frag[k]
                    acc1[k] += T.Select(
                        k + i * B1 < markov_rank, contrib, T.cast(0.0, T.float32)
                    )
                T.reduce_sum(acc1, sum_frag, clear=False)

        # Sigmoid epilogue in single thread.
        if T.get_thread_binding() == 0:
            scaled = T.min(
                T.max(sum_frag[0] / temp, T.cast(-60.0, T.float32)),
                T.cast(60.0, T.float32),
            )
            out[bx] = 1.0 / (1.0 + T.exp(-scaled))
