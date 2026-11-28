# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.config import AuxOutputConfig, VllmConfig
from vllm.config.utils import hash_config
from vllm.distributed.aux_output_connector.connector import object_put_timeout_s
from vllm.engine.arg_utils import EngineArgs

pytestmark = pytest.mark.cpu_test


def _config(
    *,
    pp: int = 1,
    dcp: int = 1,
    pcp: int = 1,
    runner_type: str = "generate",
    is_moe: bool = True,
    sliding_window: int | None = None,
    attention_chunk_size: int | None = None,
    enable_prefix_caching: bool = True,
    adaptive_verification: bool = False,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type=runner_type,
            is_moe=is_moe,
            get_sliding_window=lambda: sliding_window,
            attention_chunk_size=attention_chunk_size,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
        aux_output_config=AuxOutputConfig(enable_return_routed_experts=True),
        cache_config=SimpleNamespace(
            enable_prefix_caching=enable_prefix_caching,
        ),
        speculative_config=(
            SimpleNamespace(enable_adaptive_verification=True)
            if adaptive_verification
            else None
        ),
    )


def test_aux_output_config_defaults():
    config = AuxOutputConfig()

    assert not config.enabled
    assert not config.enable_return_routed_experts
    assert config.max_bytes is None


def test_aux_output_capture_changes_compilation_hash():
    disabled = AuxOutputConfig()
    enabled = AuxOutputConfig(enable_return_routed_experts=True)

    assert hash_config(disabled) != hash_config(enabled)


def test_legacy_routed_experts_flag_updates_aux_output_config():
    args = EngineArgs(enable_return_routed_experts=True)

    assert args.aux_output_config.enabled
    assert args.aux_output_config.enable_return_routed_experts


@pytest.mark.parametrize("as_config", [False, True], ids=["dict", "config"])
def test_legacy_routed_experts_flag_enables_an_object_key_prefix(as_config):
    aux_output_config = {"object_key_prefix": "run-1"}
    args = EngineArgs(
        enable_return_routed_experts=True,
        aux_output_config=(
            AuxOutputConfig(**aux_output_config) if as_config else aux_output_config
        ),
    )

    assert args.aux_output_config.object_key_prefix == "run-1"
    assert args.aux_output_config.enable_return_routed_experts


def test_object_put_timeout_follows_execute_model_timeout(monkeypatch):
    # Not the default execute-model timeout: the derived default follows it.
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "500")
    assert object_put_timeout_s() == 400.0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"runner_type": "pooling"}, "only supports generate runners"),
        ({"is_moe": False}, "only supports MoE models"),
        ({"enable_prefix_caching": False}, "requires prefix caching"),
        (
            {"adaptive_verification": True},
            "adaptive speculative verification",
        ),
        ({"dcp": 2}, "context parallelism"),
        ({"pcp": 2}, "context parallelism"),
    ],
)
def test_aux_output_connector_rejects_unsupported_configuration(kwargs, error):
    with pytest.raises(ValueError, match=error):
        VllmConfig._verify_aux_output_compatibility(_config(**kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [{"sliding_window": 4096}, {"attention_chunk_size": 4096}, {"pp": 2}],
    ids=["sliding-window", "chunked-attention", "pipeline-parallel"],
)
def test_aux_output_config_accepts(kwargs):
    VllmConfig._verify_aux_output_compatibility(_config(**kwargs))


def test_aux_output_guards_are_inactive_when_capture_is_disabled():
    config: Any = VllmConfig.__new__(VllmConfig)
    config.model_config = SimpleNamespace()
    config.aux_output_config = AuxOutputConfig()

    config._verify_aux_output_compatibility()


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"object_key_prefix": ""}, "non-empty"),
        ({"object_key_prefix": "runs/1"}, "without '/'"),
    ],
    ids=["empty-prefix", "slash"],
)
def test_object_key_prefix_config_is_validated(kwargs, error):
    with pytest.raises(ValueError, match=error):
        AuxOutputConfig(**{"enable_return_routed_experts": True, **kwargs})
