# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engines share KV cache only when their KV identities are equal, so the
identity must change with what computes the KV and nothing else."""

import dataclasses
from typing import Any

import pytest

from vllm.config import CacheConfig, KVTransferConfig, ModelConfig, VllmConfig
from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.kv_identity import (
    CACHE_FACTORS_OUTSIDE_KV,
    MODEL_FACTORS_OUTSIDE_KV,
    KVIdentityTracker,
    compute_kv_identity,
    is_known_kv_identity,
)

MODEL = "facebook/opt-125m"
WEIGHTS = {"weights": "0" * 64, "weights_source": "hf_cache"}


def _identity(model: str = MODEL, **cache_kwargs) -> str:
    model_kwargs = cache_kwargs.pop("model_kwargs", {})
    config = VllmConfig(
        model_config=ModelConfig(model, **model_kwargs),
        cache_config=CacheConfig(**cache_kwargs),
    )
    return compute_kv_identity(config, [WEIGHTS, WEIGHTS]).digest


def test_factors_outside_kv_name_config_fields():
    """A renamed field would silently rejoin the identity."""
    assert {f.name for f in dataclasses.fields(ModelConfig)} >= MODEL_FACTORS_OUTSIDE_KV
    assert {f.name for f in dataclasses.fields(CacheConfig)} >= CACHE_FACTORS_OUTSIDE_KV


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_size": 32},
        {"gpu_memory_utilization": 0.5},
        {"enable_prefix_caching": False},
        {"model_kwargs": {"enforce_eager": True}},
        {"model_kwargs": {"max_logprobs": 5}},
    ],
)
def test_settings_outside_kv_keep_identity(kwargs):
    assert _identity(**kwargs) == _identity()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cache_dtype": "fp8"},
        {"cache_dtype": "fp8", "kv_cache_dtype_skip_layers": ["0"]},
        {"model_kwargs": {"dtype": "float32"}},
        {"model_kwargs": {"max_model_len": 1024}},
        {"model_kwargs": {"hf_overrides": {"num_hidden_layers": 6}}},
    ],
)
def test_settings_that_change_kv_change_identity(kwargs):
    assert _identity(**kwargs) != _identity()


def test_local_snapshot_matches_hub_id():
    """The checkpoint is identified by content: serving the same files from a
    local path keeps the identity."""
    snapshot = hf_api().snapshot_download(MODEL, allow_patterns=["*.json"])

    assert _identity(snapshot) == _identity()


def _tracker(weights: dict[str, str] = WEIGHTS) -> KVIdentityTracker:
    return KVIdentityTracker(VllmConfig(model_config=ModelConfig(MODEL)), [weights])


def test_unknown_weights_match_no_engine():
    """Weights that changed without being identified must not pair with any
    engine, including another engine whose weights are unknown."""
    tracker = _tracker()
    first, second = tracker.unknown(), tracker.unknown()

    assert not is_known_kv_identity(first.digest)
    assert first.digest != second.digest
    assert first.components["weights_source"] == "unknown"


def test_reload_identifies_files_by_content():
    """A reload of the startup files restores the startup identity. A reload
    the files cannot vouch for, identified by path or not digested, does not."""
    tracker = _tracker()
    startup = tracker.identity
    tracker.unknown()

    assert tracker.reloaded([WEIGHTS, None]) == startup
    other_files = {"weights": "1" * 64, "weights_source": "hashed"}
    assert tracker.reloaded([other_files]).digest != startup.digest
    path = {"weights": "2" * 64, "weights_source": "path"}
    assert not is_known_kv_identity(tracker.reloaded([path]).digest)
    assert not is_known_kv_identity(tracker.reloaded([None]).digest)


def test_weight_version_identifies_updated_weights():
    """Engines that load the same files and update to the same version match,
    whatever happened in between; other versions or files do not."""
    first, second = _tracker(), _tracker()
    second.unknown()
    step_1 = first.versioned("step-1")

    assert second.versioned("step-1") == step_1
    assert step_1.components["weight_version"] == "step-1"
    assert step_1.digest != _tracker().identity.digest
    assert first.versioned("step-2").digest != step_1.digest
    other_files = _tracker({"weights": "1" * 64, "weights_source": "hf_cache"})
    assert other_files.versioned("step-1").digest != step_1.digest


@pytest.mark.parametrize("nested", [False, True])
def test_enforcement_found_in_multi_connector(nested):
    extra_config: dict[str, Any] = {"enforce_kv_identity": True}
    if nested:
        child = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        extra_config = {
            "connectors": [{**child, "kv_connector_extra_config": extra_config}]
        }
    config = KVTransferConfig(
        kv_connector="MultiConnector" if nested else "NixlConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    assert config.enforces_kv_identity()
    config.kv_connector_extra_config = {}
    assert not config.enforces_kv_identity()


def test_engine_reports_identity(vllm_runner):
    """The engine identifies Hub weights by content and reports the identity
    to the frontend and on `vllm:cache_config_info`."""
    with vllm_runner(MODEL, max_model_len=256, enforce_eager=True) as runner:
        cache_config = runner.llm.llm_engine.vllm_config.cache_config

        assert cache_config.kv_identity is not None
        assert cache_config.kv_identity_components["weights_source"] == "hf_cache"
        assert cache_config.metrics_info()["kv_identity"] == cache_config.kv_identity
