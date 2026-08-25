# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.config import ArtifactConfig, VllmConfig
from vllm.engine.arg_utils import EngineArgs

pytestmark = pytest.mark.cpu_test


def _config(
    *,
    use_v2: bool = True,
    pp: int = 1,
    dcp: int = 1,
    pcp: int = 1,
    connector: str | None = None,
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
        use_v2_model_runner=use_v2,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
        artifact_config=ArtifactConfig(enable_return_routed_experts=True),
        cache_config=SimpleNamespace(
            enable_prefix_caching=enable_prefix_caching,
        ),
        speculative_config=(
            SimpleNamespace(enable_adaptive_verification=True)
            if adaptive_verification
            else None
        ),
        kv_transfer_config=(
            None
            if connector is None
            else SimpleNamespace(
                is_kv_transfer_instance=True,
                kv_connector=connector,
            )
        ),
    )


def test_artifact_config_defaults():
    config = ArtifactConfig()

    assert not config.enabled
    assert not config.enable_return_routed_experts
    assert config.max_bytes is None


def test_artifact_capture_changes_compilation_hash():
    disabled = ArtifactConfig()
    enabled = ArtifactConfig(enable_return_routed_experts=True)

    assert disabled.compute_hash() != enabled.compute_hash()


def test_legacy_routed_experts_flag_updates_artifact_config():
    args = EngineArgs(enable_return_routed_experts=True)

    assert args.artifact_config.enabled
    assert args.artifact_config.enable_return_routed_experts


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"use_v2": False}, "requires Model Runner V2"),
        ({"runner_type": "pooling"}, "only supports generate runners"),
        ({"is_moe": False}, "only supports MoE models"),
        ({"enable_prefix_caching": False}, "requires prefix caching"),
        (
            {"adaptive_verification": True},
            "adaptive speculative verification",
        ),
        ({"pp": 2}, "pipeline parallelism"),
        ({"dcp": 2}, "context parallelism"),
        ({"pcp": 2}, "context parallelism"),
        ({"connector": "MooncakeConnector"}, "incompatible with KV connectors"),
    ],
)
def test_artifact_connector_rejects_unsupported_configuration(kwargs, error):
    with pytest.raises(ValueError, match=error):
        VllmConfig._verify_artifact_compatibility(_config(**kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [{"sliding_window": 4096}, {"attention_chunk_size": 4096}],
)
def test_artifact_config_defers_attention_layout_to_kv_config(kwargs):
    VllmConfig._verify_artifact_compatibility(_config(**kwargs))


def test_artifact_guards_are_inactive_when_capture_is_disabled():
    config: Any = VllmConfig.__new__(VllmConfig)
    config.model_config = SimpleNamespace()
    config.artifact_config = ArtifactConfig()

    config._verify_artifact_compatibility()
