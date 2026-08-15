# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.layers.mamba.ops import ssu_dispatch
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FLASHINFER_REPLAYSSM_AUTO_TACTIC,
    FlashInferReplaySSMBackend,
    FlashInferReplaySSMTactic,
    use_flashinfer_replayssm_tactic,
)
from vllm.model_executor.warmup.flashinfer_replayssm_warmup import (
    FLASHINFER_REPLAYSSM_TUNING_CANDIDATES,
    FlashInferReplaySSMAutotuneResult,
    _load_cache,
    _make_cache_key,
    _ReplaySSMBenchmark,
    _save_cache,
    _select_fastest,
    flashinfer_replayssm_autotune_warmup,
)

_STAGES_ENV = "FLASHINFER_SSU_MAIN_PIPELINE_STAGES"
_CTAS_ENV = "FLASHINFER_SSU_MAIN_CTA_PER_SM"


def _fake_backend() -> FlashInferReplaySSMBackend:
    backend = FlashInferReplaySSMBackend.__new__(FlashInferReplaySSMBackend)
    backend._mamba_config = MambaConfig(backend=MambaBackendEnum.FLASHINFER)
    backend._kernel = Mock(return_value=torch.empty(1))
    backend._algorithm = "auto"
    return backend


def test_replayssm_tuning_candidates_and_deterministic_selection():
    assert [tactic.name for tactic in FLASHINFER_REPLAYSSM_TUNING_CANDIDATES] == [
        "auto",
        "monolith",
        "two_kernel_s1_c1",
        "two_kernel_s1_c2",
        "two_kernel_s1_c4",
        "two_kernel_s1_c8",
        "two_kernel_s1_c16",
        "two_kernel_s2_c1",
        "two_kernel_s2_c2",
        "two_kernel_s2_c4",
        "two_kernel_s2_c8",
        "two_kernel_s2_c16",
    ]
    timings = [3.0, 2.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    assert _select_fastest(timings).name == "monolith"
    assert _select_fastest([float("inf")] * len(timings)) is None


def test_replayssm_tuning_key_distinguishes_batch_and_T():
    fingerprint = {"geometry": "h64_d64_n128"}
    assert _make_cache_key(fingerprint, 32, 8) != _make_cache_key(fingerprint, 256, 1)
    assert _make_cache_key(fingerprint, 32, 8) == _make_cache_key(fingerprint, 32, 8)


def test_replayssm_autotune_cache_round_trip(tmp_path):
    path = tmp_path / "replayssm.json"
    _save_cache(path, {"key": "two_kernel_s2_c16", "bad": "unknown"})
    assert _load_cache(path) == {"key": "two_kernel_s2_c16"}


@pytest.mark.parametrize("payload", ["null", "[]", "1"])
def test_replayssm_autotune_ignores_non_mapping_cache(tmp_path, payload):
    path = tmp_path / "replayssm.json"
    path.write_text(payload)
    assert _load_cache(path) == {}


def test_replayssm_benchmark_reserves_null_cache_slot():
    pytest.importorskip("flashinfer.mamba")
    cache_slots, nheads, headdim, dstate, ngroups = 5, 2, 4, 8, 1
    layer = SimpleNamespace(
        kv_cache=(
            torch.empty(0),
            torch.empty(cache_slots, nheads, headdim, dstate),
            torch.empty(cache_slots, nheads, 17, headdim),
            torch.empty(cache_slots, nheads, 17),
            torch.empty(cache_slots, ngroups, 17, dstate),
        ),
        A=torch.empty(nheads),
        D=torch.empty(nheads),
        dt_bias=torch.empty(nheads),
        mamba_config=SimpleNamespace(
            enable_stochastic_rounding=False,
            stochastic_rounding_philox_rounds=None,
        ),
    )

    benchmark = _ReplaySSMBenchmark(layer, 3)
    assert benchmark.indices.tolist() == [1, 2, 3]
    assert benchmark.ring_start.shape == (cache_slots,)
    assert benchmark.initial_ring_start.tolist() == [0, 0, 1, 2, 0]
    assert benchmark.initial_prev_num_accepted.tolist() == [0, 1, 2, 3, 0]
    _ReplaySSMBenchmark(layer, cache_slots - 1)
    with pytest.raises(ValueError, match="needs 6 cache slots"):
        _ReplaySSMBenchmark(layer, cache_slots)


@pytest.mark.parametrize(
    ("use_v2_model_runner", "use_ubatching"),
    [(True, False), (False, True)],
)
def test_replayssm_autotune_safely_skips_unsupported_runners(
    use_v2_model_runner, use_ubatching
):
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=use_ubatching)
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=True)
        ),
        model_config=SimpleNamespace(enforce_eager=False),
        use_v2_model_runner=use_v2_model_runner,
    )

    flashinfer_replayssm_autotune_warmup(worker)
    assert runner.flashinfer_replayssm_autotune_result is None


def test_replayssm_tactic_scope_restores_algorithm_and_environment(
    monkeypatch,
):
    backend = _fake_backend()
    monkeypatch.setattr(ssu_dispatch, "_replayssm_backend", backend)
    monkeypatch.setenv(_STAGES_ENV, "7")
    monkeypatch.setenv(_CTAS_ENV, "11")

    tactic = FlashInferReplaySSMTactic("two-kernel", 2, 16)
    with (
        pytest.raises(RuntimeError, match="sentinel"),
        use_flashinfer_replayssm_tactic(tactic),
    ):
        assert backend._algorithm == "two-kernel"
        assert ssu_dispatch.os.environ[_STAGES_ENV] == "2"
        assert ssu_dispatch.os.environ[_CTAS_ENV] == "16"
        raise RuntimeError("sentinel")

    assert backend._algorithm == "auto"
    assert ssu_dispatch.os.environ[_STAGES_ENV] == "7"
    assert ssu_dispatch.os.environ[_CTAS_ENV] == "11"

    with use_flashinfer_replayssm_tactic(FLASHINFER_REPLAYSSM_AUTO_TACTIC):
        assert _STAGES_ENV not in ssu_dispatch.os.environ
        assert _CTAS_ENV not in ssu_dispatch.os.environ


def test_replayssm_backend_uses_scoped_algorithm(monkeypatch):
    backend = _fake_backend()
    monkeypatch.setattr(ssu_dispatch, "_replayssm_backend", backend)
    tensor = torch.empty(1)

    with use_flashinfer_replayssm_tactic(FlashInferReplaySSMTactic("monolith")):
        backend(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
        )

    assert backend._kernel.call_args.kwargs["algorithm"] == "monolith"


def test_replayssm_capture_tactic_uses_request_batch_not_tokens():
    result = FlashInferReplaySSMAutotuneResult(
        spec_query_len=1,
        tactics={32: FlashInferReplaySSMTactic("two-kernel", 2, 8)},
    )
    assert (
        result.tactic_for(
            BatchDescriptor(num_tokens=32, num_reqs=32, uniform=True)
        ).name
        == "two_kernel_s2_c8"
    )
    assert (
        result.tactic_for(BatchDescriptor(num_tokens=256, num_reqs=32, uniform=True))
        is None
    )
    mtp_result = FlashInferReplaySSMAutotuneResult(
        spec_query_len=8,
        tactics={32: FlashInferReplaySSMTactic("two-kernel", 2, 8)},
    )
    assert (
        mtp_result.tactic_for(
            BatchDescriptor(num_tokens=256, num_reqs=32, uniform=True)
        ).name
        == "two_kernel_s2_c8"
    )
    assert (
        result.tactic_for(BatchDescriptor(num_tokens=32, num_reqs=None, uniform=False))
        is None
    )
