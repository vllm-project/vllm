# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU startup must build the sampling kernels that profile_run leaves out."""

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform
from vllm.v1.worker.gpu import warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run")


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU worker only")
def test_cpu_worker_drives_jit_warmup_registry(monkeypatch):
    """CPUWorker skips kernel_warmup(), so it must run the registry itself."""
    from vllm.v1.worker import cpu_worker

    calls: list[str] = []
    monkeypatch.setattr(
        cpu_worker.current_platform,
        "uses_standalone_cpu_model_runner",
        lambda: False,
    )
    monkeypatch.setattr(
        warmup,
        "warmup_kernels",
        lambda runner, execute, sample: calls.append("kernels"),
    )
    monkeypatch.setattr(cpu_worker, "set_random_seed", lambda seed, dp_index: None)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            kv_caches=[object()],
            jit_warmup_registry=SimpleNamespace(
                warmup=lambda: calls.append("registry")
            ),
        ),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_jit_warmup=True)
        ),
        use_v2_model_runner=True,
        execute_model=_fail,
        sample_tokens=_fail,
        model_config=SimpleNamespace(seed=0),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        compilation_config=SimpleNamespace(
            compilation_time=0.0, encoder_compilation_time=0.0
        ),
    )

    cpu_worker.CPUWorker.compile_or_warm_up_model(worker)

    assert calls == ["registry", "kernels"]


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU worker only")
def test_async_tensor_h2d_does_not_alias_numpy():
    """A numpy view carries its own alignment, which Triton specializes on."""
    import numpy as np
    import torch

    from vllm.v1.worker.cpu import shm

    unaligned = np.arange(64, dtype=np.int32)[1:]
    out = shm.async_tensor_h2d(unaligned)

    assert out.data_ptr() != torch.from_numpy(unaligned).data_ptr()
    assert out.data_ptr() % 16 == 0
    assert torch.equal(out, torch.from_numpy(unaligned))
