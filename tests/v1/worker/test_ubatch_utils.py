# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import ParallelConfig
from vllm.v1.worker import gpu_ubatch_wrapper, ubatch_utils


@pytest.mark.parametrize(
    ("all2all_backend", "num_tokens", "uniform_decode", "expected"),
    [
        ("deepep_high_throughput", 15, True, False),
        ("deepep_high_throughput", 16, True, True),
        ("deepep_high_throughput", 255, False, False),
        ("deepep_high_throughput", 256, False, True),
        ("deepep_low_latency", 16, True, True),
    ],
)
def test_check_ubatch_thresholds(
    all2all_backend: str,
    num_tokens: int,
    uniform_decode: bool,
    expected: bool,
):
    config = ParallelConfig(
        enable_dbo=True,
        all2all_backend=all2all_backend,
        dbo_decode_token_threshold=16,
        dbo_prefill_token_threshold=256,
    )

    assert (
        ubatch_utils.check_ubatch_thresholds(
            config,
            num_tokens=num_tokens,
            uniform_decode=uniform_decode,
        )
        is expected
    )


def test_rocm_deepep_ht_dbo_disables_sm_partition(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, int] = {}
    sentinel = object()

    def fake_sm_control_context_manager(
        *,
        comm_sms: int,
        set_comm_sms,
        set_compute_sms,
    ):
        captured["comm_sms"] = comm_sms
        return sentinel

    monkeypatch.setattr(gpu_ubatch_wrapper.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(gpu_ubatch_wrapper.envs, "VLLM_DBO_COMM_SMS", 20)
    monkeypatch.setattr(
        gpu_ubatch_wrapper,
        "SMControlContextManager",
        fake_sm_control_context_manager,
    )

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_dbo=True,
            all2all_backend="deepep_high_throughput",
            enable_expert_parallel=False,
        )
    )

    context = gpu_ubatch_wrapper.UBatchWrapper._create_sm_control_context(config)

    assert context is sentinel
    assert captured["comm_sms"] == 0
