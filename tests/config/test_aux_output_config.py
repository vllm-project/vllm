# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.config import AuxOutputConfig, VllmConfig
from vllm.config.kv_transfer import KVRole, KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

pytestmark = pytest.mark.cpu_test


def _config(
    *,
    use_v2: bool = True,
    pp: int = 1,
    dcp: int = 1,
    pcp: int = 1,
    connector: str | None = None,
    kv_role: KVRole = "kv_both",
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
        aux_output_config=AuxOutputConfig(enable_return_routed_experts=True),
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
            else KVTransferConfig(
                kv_connector=connector,
                kv_role=kv_role,
            )
        ),
    )


def test_aux_output_config_defaults():
    config = AuxOutputConfig()

    assert not config.enabled
    assert not config.enable_return_routed_experts
    assert config.max_bytes is None
    assert config.backend == "shm"


def test_aux_output_capture_changes_compilation_hash():
    disabled = AuxOutputConfig()
    enabled = AuxOutputConfig(enable_return_routed_experts=True)

    assert disabled.compute_hash() != enabled.compute_hash()
    assert (
        enabled.compute_hash()
        == AuxOutputConfig(
            enable_return_routed_experts=True, backend="mooncake"
        ).compute_hash()
    )


def test_legacy_routed_experts_flag_updates_aux_output_config():
    args = EngineArgs(enable_return_routed_experts=True)

    assert args.aux_output_config.enabled
    assert args.aux_output_config.enable_return_routed_experts


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
    ],
)
def test_aux_output_connector_rejects_unsupported_configuration(kwargs, error):
    with pytest.raises(ValueError, match=error):
        VllmConfig._verify_aux_output_compatibility(_config(**kwargs))


@pytest.mark.parametrize(
    "connector",
    [
        "NixlConnector",
        "NixlPullConnector",
        "NixlPushConnector",
        "MoRIIOConnector",
        "MooncakeConnector",
        "MooncakeStoreConnector",
        "OffloadingConnector",
        "LMCacheConnectorV1",
        "LMCacheMPConnector",
        "SimpleCPUOffloadConnector",
    ],
)
@pytest.mark.parametrize("kv_role", ["kv_both", "kv_producer", "kv_consumer"])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("backend", ["shm", "mooncake"])
def test_aux_output_does_not_restrict_kv_connector_configuration(
    connector, kv_role, multi, backend
):
    config = _config(connector=connector, kv_role=kv_role)
    config.aux_output_config.backend = backend
    if multi:
        config.kv_transfer_config = KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role=kv_role,
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "OffloadingConnector", "kv_role": "kv_both"},
                    {"kv_connector": connector, "kv_role": kv_role},
                ]
            },
        )
    VllmConfig._verify_aux_output_compatibility(config)


@pytest.mark.parametrize(
    "kwargs",
    [{"sliding_window": 4096}, {"attention_chunk_size": 4096}],
)
def test_aux_output_config_defers_attention_layout_to_kv_config(kwargs):
    VllmConfig._verify_aux_output_compatibility(_config(**kwargs))


def test_aux_output_guards_are_inactive_when_capture_is_disabled():
    config: Any = VllmConfig.__new__(VllmConfig)
    config.model_config = SimpleNamespace()
    config.aux_output_config = AuxOutputConfig()

    config._verify_aux_output_compatibility()
