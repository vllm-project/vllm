# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.model_executor.warmup.kernel_warmup import KernelWarmup, kernel_warmup


@pytest.mark.parametrize(
    "process_local_only,expected",
    [(False, ["jit", "model_runs", "autotune"]), (True, ["jit"])],
)
def test_kernel_warmup_runs_the_model_warmup_phases(process_local_only, expected):
    """The model's own `KernelWarmup` runs; process-local warmup (elastic EP)
    stops before the phases that communicate with other ranks."""
    calls: list[str] = []

    class ModelWarmup(KernelWarmup):
        def warmup_jit_kernels(self, worker):
            calls.append("jit")

        def warmup_model_runs(self, worker):
            calls.append("model_runs")

        def autotune(self, worker):
            calls.append("autotune")

    model = SimpleNamespace(get_kernel_warmup_cls=lambda: ModelWarmup)
    runner = SimpleNamespace(vllm_config=None, get_model=lambda: model)

    kernel_warmup(
        SimpleNamespace(model_runner=runner), process_local_only=process_local_only
    )

    assert calls == expected
