# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.config import VllmConfig

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _config():
    parallel = SimpleNamespace(
        enable_dp_execution_contract=True,
        enable_speculator_dp_sync_pipeline=False,
        enable_cached_dp_execution_contract=False,
        dp_execution_contract_stability_steps=2,
        data_parallel_size=2,
        enable_expert_parallel=True,
        all2all_backend="flashinfer_nvlink_one_sided",
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_dbo=False,
        enable_eplb=False,
        enable_elastic_ep=False,
    )
    model = SimpleNamespace(
        is_moe=True,
        runner_type="generate",
        architecture="MiniMaxM3SparseForCausalLM",
    )
    return SimpleNamespace(
        parallel_config=parallel,
        model_config=model,
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            prefill_schedule_interval=1,
        ),
        lora_config=None,
        speculative_config=None,
        use_v2_model_runner=True,
    )


def test_dp_execution_contract_accepts_validated_subset():
    VllmConfig._verify_dp_execution_contract(_config())


def test_dp_execution_contract_accepts_async_scheduling():
    config = _config()
    config.scheduler_config.async_scheduling = True

    VllmConfig._verify_dp_execution_contract(config)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("model_config", "architecture"), "OtherMoEForCausalLM", "architecture"),
        (("parallel_config", "all2all_backend"), "deepep_low_latency", "backend"),
    ],
)
def test_dp_execution_contract_rejects_unvalidated_combinations(path, value, match):
    config = _config()
    setattr(getattr(config, path[0]), path[1], value)

    with pytest.raises(ValueError, match=match):
        VllmConfig._verify_dp_execution_contract(config)


def test_dp_execution_contract_requires_padding_mask(monkeypatch):
    config = _config()
    monkeypatch.setattr(envs, "VLLM_MOE_SKIP_PADDING", False)

    with pytest.raises(ValueError, match="VLLM_MOE_SKIP_PADDING=0"):
        VllmConfig._verify_dp_execution_contract(config)


def test_speculator_dp_pipeline_requires_execution_contract():
    config = _config()
    config.parallel_config.enable_dp_execution_contract = False
    config.parallel_config.enable_speculator_dp_sync_pipeline = True

    with pytest.raises(ValueError, match="requires enable_dp_execution_contract"):
        VllmConfig._verify_dp_execution_contract(config)


def test_speculator_dp_pipeline_requires_speculative_decoding():
    config = _config()
    config.parallel_config.enable_speculator_dp_sync_pipeline = True

    with pytest.raises(ValueError, match="speculative decoding disabled"):
        VllmConfig._verify_dp_execution_contract(config)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_speculator_dp_pipeline_accepts_autoregressive_speculator(
    async_scheduling: bool,
):
    config = _config()
    config.parallel_config.enable_speculator_dp_sync_pipeline = True
    config.scheduler_config.async_scheduling = async_scheduling
    config.speculative_config = SimpleNamespace(
        method="mtp",
        use_eagle=lambda: True,
        use_multi_module_mtp=lambda: False,
    )
    config.num_speculative_tokens = 3

    VllmConfig._verify_dp_execution_contract(config)


def _enable_cached_contract(config):
    config.parallel_config.enable_speculator_dp_sync_pipeline = True
    config.parallel_config.enable_cached_dp_execution_contract = True
    config.speculative_config = SimpleNamespace(
        method="mtp",
        use_eagle=lambda: True,
        use_multi_module_mtp=lambda: False,
    )
    config.num_speculative_tokens = 3


def test_cached_contract_requires_speculator_pipeline():
    config = _config()
    config.parallel_config.enable_cached_dp_execution_contract = True

    with pytest.raises(ValueError, match="requires enable_speculator_dp_sync_pipeline"):
        VllmConfig._verify_dp_execution_contract(config)


def test_cached_contract_requires_nontrivial_refresh_cadence():
    config = _config()
    _enable_cached_contract(config)

    with pytest.raises(ValueError, match="prefill_schedule_interval <= 1"):
        VllmConfig._verify_dp_execution_contract(config)


def test_cached_contract_requires_positive_stability_steps():
    config = _config()
    _enable_cached_contract(config)
    config.scheduler_config.prefill_schedule_interval = 32
    config.parallel_config.dp_execution_contract_stability_steps = 0

    with pytest.raises(ValueError, match="stability_steps <= 0"):
        VllmConfig._verify_dp_execution_contract(config)


def test_cached_contract_accepts_scheduler_owned_epoch_config():
    config = _config()
    _enable_cached_contract(config)
    config.scheduler_config.prefill_schedule_interval = 32

    VllmConfig._verify_dp_execution_contract(config)
