# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    compute_num_split,
    mhc_pre_big_fuse_with_norm_tilelang,
)
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.math_utils import cdiv


def compute_mhc_num_splits(input_size: int, num_tokens: int) -> int:
    return compute_num_split(64, input_size, cdiv(num_tokens, 64))


def compute_mhc_pre_num_splits(input_size: int, num_tokens: int) -> int:
    splits = compute_mhc_num_splits(input_size, num_tokens)
    # Bound both GEMM and fused-normalization specializations during startup.
    return 1 if splits == 1 else 4 if splits <= 4 else 16


class MHCPreNormKernel(VllmJitKernel["MHCPreNormKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        hidden_size: int
        rms_eps: float
        hc_pre_eps: float
        hc_sinkhorn_eps: float
        hc_post_mult_value: float
        sinkhorn_repeat: int
        norm_eps: float
        n_splits: int
        hc_mult: int
        use_pre_mix_in: bool
        rms_numel: int
        save_pre_mix: bool = True
        split_mode: str = "fused"
        write_aux: bool = False

    kernel: Any = staticmethod(mhc_pre_big_fuse_with_norm_tilelang)

    def dispatch(self, *, n_splits, **fields) -> CompileKey:
        return self.CompileKey(n_splits=n_splits, **fields)

    def get_warmup_keys(
        self,
        *,
        max_tokens: int,
        extra_splits: Iterable[int] = (),
        **fields,
    ) -> list[CompileKey]:
        # The split heuristic changes only at 64-token boundaries. Callers that
        # reach this kernel through another GEMM (the fused post + pre-norm
        # GEMM) pass the splits that GEMM picks in extra_splits.
        splits = sorted(
            set(extra_splits)
            | (
                {
                    compute_mhc_pre_num_splits(fields["rms_numel"], num_tokens)
                    for num_tokens in range(1, max_tokens + 1, 64)
                }
                if is_deep_gemm_supported()
                else {1}
            )
        )
        return self._trace_dispatch(self.dispatch)(n_splits=splits, **fields)

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key not in self._compiled_cache:
            self._compiled_cache[compile_key] = self.kernel.compile(
                **asdict(compile_key)
            )

    def compile_many(self, compile_keys: Iterable[CompileKey]) -> None:
        missing = list(
            dict.fromkeys(
                key for key in compile_keys if key not in self._compiled_cache
            )
        )
        if len(missing) < 2:
            return super().compile_many(missing)

        # TileLang elaborates serially before compiling independent TIR modules.
        compiled = self.kernel.par_compile(
            [asdict(key) for key in missing], num_workers=min(4, len(missing))
        )
        self._compiled_cache.update(zip(missing, compiled, strict=True))

    def __call__(self, *tensors, **fields):
        compile_key = self.dispatch(
            n_splits=tensors[0].shape[0],
            **fields,
        )
        if not current_platform.is_cuda():
            return self.kernel(*tensors, **asdict(compile_key))
        return self._get_or_compile(compile_key)(*tensors)


MHC_PRE_NORM_KERNEL = MHCPreNormKernel()


class HCPrenormGemmKernel(VllmJitKernel["HCPrenormGemmKernel.CompileKey"]):
    """DeepGEMM's `tf32_hc_prenorm_gemm`, which JITs once per (n, k, splits)."""

    @dataclass(frozen=True)
    class CompileKey:
        n: int
        k: int
        n_splits: int

    def get_warmup_keys(
        self,
        *,
        max_tokens: int,
        n: int,
        k: int,
        num_splits: Callable[[int, int], int],
    ) -> list[CompileKey]:
        # num_splits is the caller's runtime heuristic; it moves every 64 tokens.
        splits = {
            num_splits(k, num_tokens) for num_tokens in range(1, max_tokens + 1, 64)
        }
        return [self.CompileKey(n=n, k=k, n_splits=s) for s in sorted(splits)]

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key in self._compiled_cache:
            return
        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

        n, k, splits = compile_key.n, compile_key.k, compile_key.n_splits
        # M is a launch argument, not a template parameter, so any M works.
        num_tokens = 64
        device = current_platform.device_type
        x = torch.zeros((num_tokens, k), device=device, dtype=torch.bfloat16)
        fn = torch.zeros((n, k), device=device, dtype=torch.float32)
        out = torch.empty((splits, num_tokens, n), device=device, dtype=torch.float32)
        sqr_sum = torch.empty((splits, num_tokens), device=device, dtype=torch.float32)
        tf32_hc_prenorm_gemm(x, fn, out, sqr_sum, splits)
        self._compiled_cache[compile_key] = True


HC_PRENORM_GEMM_KERNEL = HCPrenormGemmKernel()
