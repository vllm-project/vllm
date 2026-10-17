# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Compile-only startup warmup for the composite Hadamard epilogue."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel


class CompositeHadamardEpilogueKernel(
    VllmJitKernel["CompositeHadamardEpilogueKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        k: int
        p: int
        inverse: bool
        dtype: torch.dtype

    def dispatch(  # type: ignore[override]
        self, k: int, p: int, inverse: bool, dtype: torch.dtype
    ) -> CompileKey:
        return self.CompileKey(k=k, p=p, inverse=inverse, dtype=dtype)

    def get_warmup_keys(
        self, *, k: int, p: int, inverse: bool, dtype: torch.dtype
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            k=k, p=p, inverse=inverse, dtype=dtype
        )

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key not in self._compiled_cache:
            from .tilelang_kernels import make_composite_hadamard_epilogue

            dtype = str(compile_key.dtype).removeprefix("torch.")
            compiled = make_composite_hadamard_epilogue.compile(
                compile_key.k, compile_key.p, compile_key.inverse, dtype
            )
            self._compiled_cache[compile_key] = compiled

    def __call__(self, k: int, p: int, inverse: bool, dtype: torch.dtype) -> Any:
        return self._get_or_compile(self.dispatch(k, p, inverse, dtype))


COMPOSITE_HADAMARD_EPILOGUE = CompositeHadamardEpilogueKernel()
