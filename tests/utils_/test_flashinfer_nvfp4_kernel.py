# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


def test_cutlass_kernel_limits_jit_parallelism(monkeypatch: pytest.MonkeyPatch):
    """Selecting the FlashInfer CUTLASS kernel clamps MAX_JOBS before the first
    FP4 GEMM, which is the call that triggers the JIT build."""
    nvfp4 = pytest.importorskip("vllm.model_executor.kernels.linear.nvfp4.flashinfer")
    kernel_cls = nvfp4.FlashInferCutlassNvFp4LinearKernel
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.setattr(
        kernel_cls, "is_supported", classmethod(lambda cls, cc=None: (True, None))
    )
    monkeypatch.setattr(
        kernel_cls, "can_implement", classmethod(lambda cls, config: (True, None))
    )
    calls: list[bool] = []
    monkeypatch.setattr(
        nvfp4, "limit_flashinfer_jit_parallelism", lambda: calls.append(True)
    )

    kernel_cls(nvfp4.NvFp4LinearLayerConfig())

    assert calls == [True]
