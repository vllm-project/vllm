# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-batch GEMMs of the attention side projections.

Next to ``fused_wqa_wkv``, the kv-source layers project the attention input
through the compressor's ``fused_wkv_wgate`` (FP32 ``kv_score``) and the
index-source layers through the indexer's ``weights_proj``. At decode sizes
cuBLAS runs both as a split-K GEMM plus a reduce kernel, longer than
``fused_wqa_wkv`` itself. For low-batch FULL-graph steps they run the CuTe DSL
``ll_bf16`` kernels instead: FP32 ``x @ W.T`` that differs from cuBLAS only in
the K summation order. The indexer's ``wk`` (compressor latent to index key)
takes the same kernels.
"""

from collections.abc import Iterable

import torch

from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import LLBf16Gemm

from .. import LOW_BATCH_MAX_ROWS

# Below this N the split-K kernel has too few CTAs; the dot product runs.
_DOTPROD_MAX_N = 128
_DOTPROD_BS = 128

# Per (K, N) and row count, the fastest config on GB300 where it beats the
# default dispatch: ("dotprod", bs) or ("splitk", (split_k, num_stages)).
_TUNED: dict[tuple[int, int], dict[int, tuple[str, int | tuple[int, int]]]] = {
    # Compressor ratio 2 (kv + score).
    (5120, 1024): {m: ("splitk", (4, 4)) for m in (5, 6, 7, 8)},
    # Compressor ratio 1 (kv only).
    (5120, 512): {5: ("dotprod", 128), 6: ("dotprod", 64), 8: ("splitk", (4, 6))},
}


class AuxProjectionGemm(LLBf16Gemm):
    def dispatch(
        self, *, M: int, K: int, N: int, output_dtype: torch.dtype = torch.float32
    ) -> LLBf16Gemm.CompileKey:
        tuned = _TUNED.get((K, N), {}).get(M)
        if tuned is not None:
            backend, config = tuned
            if backend == "dotprod":
                assert isinstance(config, int)
                return self.CompileKey(
                    backend="dotprod", M=M, K=K, bs=config, output_dtype=output_dtype
                )
            assert isinstance(config, tuple)
            split_k, num_stages = config
            return self.CompileKey(
                backend="splitk",
                split_k=split_k,
                num_stages=num_stages,
                output_dtype=output_dtype,
            )
        if N < _DOTPROD_MAX_N:
            return self.CompileKey(
                backend="dotprod", M=M, K=K, bs=_DOTPROD_BS, output_dtype=output_dtype
            )
        return super().dispatch(M=M, K=K, N=N, output_dtype=output_dtype)


aux_projection_gemm = AuxProjectionGemm()


def aux_projection_gemm_accepts(weights: list[torch.Tensor]) -> bool:
    """Whether ``aux_projection_gemm`` takes every one of ``weights``."""
    return (
        bool(weights)
        and ll_bf16.is_available()
        and all(
            w.dtype == torch.bfloat16
            and w.dim() == 2
            and w.is_contiguous()
            and w.shape[1] % 8 == 0
            for w in weights
        )
    )


def warmup_aux_projection_gemms(weights: Iterable[torch.Tensor]) -> None:
    """Compile the kernels of every row count before graph capture."""
    shapes = sorted({(w.shape[1], w.shape[0]) for w in weights})
    aux_projection_gemm.warmup(shapes=shapes, m_values=range(1, LOW_BATCH_MAX_ROWS + 1))
