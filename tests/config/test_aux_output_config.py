# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.config import AuxOutputConfig, VllmConfig
from vllm.config.utils import hash_config
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


def test_aux_output_hash_separates_every_flag_combination():
    """Both flags always contribute, so a later `False` key cannot alias a hash."""
    hashes = {
        (routed, indexer): hash_config(
            AuxOutputConfig(
                enable_return_routed_experts=routed,
                enable_return_indexer_topk=indexer,
            )
        )
        for routed in (False, True)
        for indexer in (False, True)
    }

    assert len(set(hashes.values())) == 4


def test_legacy_routed_experts_flag_updates_aux_output_config():
    args = EngineArgs(enable_return_routed_experts=True)

    assert args.aux_output_config.enabled
    assert args.aux_output_config.enable_return_routed_experts


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
    [{"sliding_window": 4096}, {"attention_chunk_size": 4096}],
)
def test_aux_output_config_defers_attention_layout_to_kv_config(kwargs):
    VllmConfig._verify_aux_output_compatibility(_config(**kwargs))


def test_routed_experts_capture_accepts_pipeline_parallelism():
    VllmConfig._verify_aux_output_compatibility(_config(pp=2))


def test_indexer_topk_rejects_pipeline_parallelism():
    config = _config(pp=2)
    config.aux_output_config = AuxOutputConfig(enable_return_indexer_topk=True)
    with pytest.raises(ValueError, match="pipeline parallelism"):
        VllmConfig._verify_aux_output_compatibility(config)


def test_aux_output_guards_are_inactive_when_capture_is_disabled():
    config: Any = VllmConfig.__new__(VllmConfig)
    config.model_config = SimpleNamespace()
    config.aux_output_config = AuxOutputConfig()

    config._verify_aux_output_compatibility()


@pytest.mark.parametrize(
    "routed,indexer", [(False, False), (True, False), (False, True), (True, True)]
)
def test_aux_output_flags_are_independent(routed, indexer):
    config = AuxOutputConfig(
        enable_return_routed_experts=routed,
        enable_return_indexer_topk=indexer,
    )
    assert config.enabled == (routed or indexer)
    assert (
        EngineArgs(
            enable_return_indexer_topk=indexer
        ).aux_output_config.enable_return_indexer_topk
        == indexer
    )


def test_indexer_only_does_not_require_moe_routing():
    config = _config(is_moe=False)
    config.aux_output_config = AuxOutputConfig(enable_return_indexer_topk=True)
    VllmConfig._verify_aux_output_compatibility(config)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"object_key_prefix": "run-1"}, None),
        ({"object_key_prefix": ""}, "non-empty"),
        ({"object_key_prefix": "runs/1"}, "without '/'"),
        (
            {
                "object_key_prefix": "run-1",
                "enable_return_routed_experts": False,
                "enable_return_indexer_topk": True,
            },
            None,
        ),
    ],
    ids=["valid", "empty-prefix", "slash", "indexer-only"],
)
def test_object_key_prefix_is_validated(kwargs, error):
    kwargs = {"enable_return_routed_experts": True, **kwargs}
    if error is None:
        AuxOutputConfig(**kwargs)
        return
    with pytest.raises(ValueError, match=error):
        AuxOutputConfig(**kwargs)


@pytest.mark.parametrize(
    "name,value",
    [
        ("enable_return_hidden_states", True),
        ("return_topk_logprobs", 2),
        ("mooncake_config_path", "objects.json"),
    ],
    ids=["hidden-states", "topk-logprobs", "mooncake-config"],
)
def test_session_fields_require_object_key_prefix(name, value):
    with pytest.raises(ValueError, match=f"{name} requires object_key_prefix"):
        AuxOutputConfig(enable_return_routed_experts=True, **{name: value})
    AuxOutputConfig(
        enable_return_routed_experts=True, object_key_prefix="run-1", **{name: value}
    )


@pytest.mark.parametrize(
    "legacy_flag,error",
    [(True, None), (False, "object_key_prefix requires")],
    ids=["legacy-flag", "no-output"],
)
def test_object_key_prefix_takes_its_output_from_the_legacy_flag(legacy_flag, error):
    config = _config()
    config.aux_output_config = EngineArgs(
        aux_output_config={"object_key_prefix": "job"},
        enable_return_routed_experts=legacy_flag,
    ).aux_output_config
    if error is None:
        VllmConfig._verify_aux_output_compatibility(config)
        return
    with pytest.raises(ValueError, match=error):
        VllmConfig._verify_aux_output_compatibility(config)


@pytest.mark.parametrize("timeout_s", [0, -1.0], ids=["zero", "negative"])
def test_object_put_stall_timeout_must_be_positive(timeout_s):
    with pytest.raises(ValueError):
        AuxOutputConfig(object_put_stall_timeout_s=timeout_s)
