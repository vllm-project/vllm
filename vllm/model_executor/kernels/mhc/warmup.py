# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict, dataclass
from typing import Any

from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    compute_num_split,
    mhc_pre_big_fuse_with_norm_tilelang,
)
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.math_utils import cdiv


def compute_mhc_pre_num_splits(input_size: int, num_tokens: int) -> int:
    splits = compute_num_split(64, input_size, cdiv(num_tokens, 64))
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

    kernel: Any = staticmethod(mhc_pre_big_fuse_with_norm_tilelang)

    def dispatch(self, *, n_splits, **fields) -> CompileKey:  # type: ignore[override]
        return self.CompileKey(n_splits=n_splits, **fields)

    def get_warmup_keys(self, *, max_tokens: int, **fields) -> list[CompileKey]:
        # The split heuristic changes only at 64-token boundaries.
        splits = (
            sorted(
                {
                    compute_mhc_pre_num_splits(fields["rms_numel"], num_tokens)
                    for num_tokens in range(1, max_tokens + 1, 64)
                }
            )
            if is_deep_gemm_supported()
            else [1]
        )
        return self._trace_dispatch(self.dispatch)(n_splits=splits, **fields)

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key not in self._compiled_cache:
            self._compiled_cache[compile_key] = self.kernel.compile(
                **asdict(compile_key)
            )

    def __call__(self, *tensors, **fields):
        compile_key = self.dispatch(
            n_splits=tensors[0].shape[0],
            **fields,
        )
        if not current_platform.is_cuda():
            return self.kernel(*tensors, **asdict(compile_key))
        return self._get_or_compile(compile_key)(*tensors)


MHC_PRE_NORM_KERNEL = MHCPreNormKernel()
