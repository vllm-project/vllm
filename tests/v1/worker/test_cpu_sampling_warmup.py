# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU startup must build the sampling kernels that profile_run leaves out."""

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.gpu import warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run")


@pytest.mark.parametrize(
    ("is_cpu", "is_pooling_model", "expected_rounds"),
    [
        (True, False, [False, True]),
        (True, True, [False]),
        (False, False, [False]),
    ],
)
def test_greedy_warmup_round_runs_on_cpu(
    monkeypatch, is_cpu, is_pooling_model, expected_rounds
):
    """Without speculative decoding only CPU pays for the second, greedy pass."""
    rounds: list[bool] = []
    monkeypatch.setattr(
        warmup,
        "_warmup_kernels",
        lambda runner, execute, sample, greedy=False: rounds.append(greedy),
    )
    monkeypatch.setattr(
        warmup, "current_platform", SimpleNamespace(is_cpu=lambda: is_cpu)
    )
    runner = SimpleNamespace(
        adaptive_verification=None,
        speculator=None,
        speculative_config=None,
        rejection_sampler=None,
        is_pooling_model=is_pooling_model,
    )

    warmup.warmup_kernels(runner, _fail, _fail)

    assert rounds == expected_rounds


@pytest.mark.skipif(not HAS_TRITON, reason="sampling kernels need Triton")
def test_top_k_top_p_warmup_registered_on_cpu(monkeypatch):
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.v1.sample.ops import topk_topp_sampler

    monkeypatch.setattr(
        topk_topp_sampler,
        "current_platform",
        SimpleNamespace(is_cpu=lambda: True, is_cuda_alike=lambda: False),
    )
    registry = JitWarmupRegistry(vllm_config=None)
    with registry.activate():
        topk_topp_sampler.register_top_k_top_p_warmups()

    assert len(registry) == 1


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU worker only")
def test_cpu_worker_drives_jit_warmup_registry(monkeypatch):
    """CPUWorker skips kernel_warmup(), so it must run the registry itself."""
    from vllm.model_executor.warmup import kernel_warmup
    from vllm.v1.worker import cpu_worker

    calls: list[str] = []
    monkeypatch.setattr(
        cpu_worker.current_platform,
        "uses_standalone_cpu_model_runner",
        lambda: False,
    )
    monkeypatch.setattr(
        kernel_warmup,
        "run_jit_warmup_registry",
        lambda worker: calls.append("registry"),
    )
    monkeypatch.setattr(
        warmup,
        "warmup_kernels",
        lambda runner, execute, sample: calls.append("kernels"),
    )
    monkeypatch.setattr(cpu_worker, "set_random_seed", lambda seed, dp_index: None)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(kv_caches=[object()]),
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
