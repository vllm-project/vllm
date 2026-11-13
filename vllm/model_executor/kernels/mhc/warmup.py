# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
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
        split_mode: str = "fused"
        write_aux: bool = False
        h_tiles: int = 1

    kernel: Any = staticmethod(mhc_pre_big_fuse_with_norm_tilelang)

    def dispatch(  # type: ignore[override]
        self,
        *,
        n_splits: int,
        num_tokens: int = 0,
        input_is_disjoint: bool = False,
        hidden_size: int,
        hc_mult: int,
        rms_numel: int,
        save_pre_mix: bool = True,
        split_mode: str = "fused",
        write_aux: bool = False,
        **fields: Any,
    ) -> CompileKey:
        return self.CompileKey(
            n_splits=n_splits,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            rms_numel=rms_numel,
            save_pre_mix=save_pre_mix,
            split_mode=split_mode,
            write_aux=write_aux,
            h_tiles=5
            if current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and 0 < num_tokens <= 8
            and input_is_disjoint
            and hidden_size == 5120
            and hc_mult == 4
            and rms_numel == hc_mult * hidden_size
            and save_pre_mix
            and split_mode == "input"
            and not write_aux
            else 1,
            **fields,
        )

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
        return self._trace_dispatch(self.dispatch)(
            num_tokens=(1, min(max_tokens, 9)),
            input_is_disjoint=(False, True),
            n_splits=splits,
            **fields,
        )

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
        # Each spatial CTA rereads the full row, so output must not alias inputs.
        input_is_disjoint = False
        if fields.get("split_mode", "fused") == "input":
            output_storage = tensors[7].untyped_storage().data_ptr()
            inputs = [tensors[4], tensors[8]]
            if fields["use_pre_mix_in"]:
                inputs.append(tensors[9])
            input_is_disjoint = all(
                tensor.untyped_storage().data_ptr() != output_storage
                for tensor in inputs
            )
        compile_key = self.dispatch(
            n_splits=tensors[0].shape[0],
            num_tokens=tensors[4].shape[0],
            input_is_disjoint=input_is_disjoint,
            **fields,
        )
        if not current_platform.is_cuda():
            return self.kernel(*tensors, **asdict(compile_key))
        return self._get_or_compile(compile_key)(*tensors)


MHC_PRE_NORM_KERNEL = MHCPreNormKernel()
