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


def test_aux_output_config_allows_pipeline_parallelism():
    VllmConfig._verify_aux_output_compatibility(_config(pp=2))


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
                "enable_return_hidden_states": True,
            },
            None,
        ),
        (
            {
                "object_key_prefix": "run-1",
                "enable_return_routed_experts": False,
                "return_topk_logprobs": 2,
            },
            None,
        ),
        (
            {
                "object_key_prefix": "run-1",
                "enable_return_routed_experts": False,
                "enable_return_indexer_topk": True,
            },
            None,
        ),
    ],
    ids=[
        "valid",
        "empty-prefix",
        "slash",
        "hidden-states-only",
        "topk-logprobs-only",
        "indexer-only",
    ],
)
def test_object_key_prefix_is_validated(kwargs, error):
    kwargs = {"enable_return_routed_experts": True, **kwargs}
    if error is None:
        AuxOutputConfig(**kwargs)
        return
    with pytest.raises(ValueError, match=error):
        AuxOutputConfig(**kwargs)


def test_hidden_states_require_object_key_prefix():
    with pytest.raises(ValueError, match="requires object_key_prefix"):
        AuxOutputConfig(
            enable_return_routed_experts=True, enable_return_hidden_states=True
        )
    AuxOutputConfig(
        enable_return_routed_experts=True,
        enable_return_hidden_states=True,
        object_key_prefix="run-1",
    )


@pytest.mark.parametrize(
    "k,lora,error",
    [(8, False, None), (9, False, "vocabulary size"), (2, True, "LoRA")],
    ids=["vocab-size", "above-vocab-size", "lora"],
)
def test_topk_logprobs_are_validated_against_the_model(k, lora, error):
    config = _config()
    config.model_config.get_vocab_size = lambda: 8
    config.lora_config = object() if lora else None
    config.aux_output_config = AuxOutputConfig(
        enable_return_routed_experts=True,
        object_key_prefix="run-1",
        return_topk_logprobs=k,
    )
    if error is None:
        VllmConfig._verify_aux_output_compatibility(config)
        return
    with pytest.raises(ValueError, match=error):
        VllmConfig._verify_aux_output_compatibility(config)


def test_dense_model_without_routed_experts_is_accepted():
    config = _config(is_moe=False)
    config.aux_output_config = AuxOutputConfig(
        object_key_prefix="run-1", enable_return_hidden_states=True
    )
    assert config.aux_output_config.enabled
    VllmConfig._verify_aux_output_compatibility(config)


@pytest.mark.parametrize(
    "legacy_flag,error",
    [(True, None), (False, "requires at least one output")],
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


@pytest.mark.parametrize(
    "routed_experts,legacy_flag,error",
    [(False, False, "max_bytes requires"), (True, False, None), (False, True, None)],
    ids=["without-routed-experts", "with-routed-experts", "legacy-flag"],
)
def test_max_bytes_requires_routed_experts(routed_experts, legacy_flag, error):
    config = _config()
    config.aux_output_config = EngineArgs(
        aux_output_config={
            "enable_return_routed_experts": routed_experts,
            "max_bytes": 1 << 20,
            "object_key_prefix": "run-1",
            "enable_return_hidden_states": True,
        },
        enable_return_routed_experts=legacy_flag,
    ).aux_output_config
    if error is None:
        VllmConfig._verify_aux_output_compatibility(config)
        return
    with pytest.raises(ValueError, match=error):
        VllmConfig._verify_aux_output_compatibility(config)
