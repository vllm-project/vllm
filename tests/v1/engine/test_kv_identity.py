# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engines share KV cache only when their KV identities are equal, so the
identity must change with what computes the KV and nothing else."""

import dataclasses

import pytest

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.kv_identity import (
    CACHE_FACTORS_OUTSIDE_KV,
    MODEL_FACTORS_OUTSIDE_KV,
    compute_kv_identity,
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


def test_weights_change_identity():
    config = VllmConfig(model_config=ModelConfig(MODEL))
    retrained = {"weights": "1" * 64, "weights_source": "hf_cache"}

    original = compute_kv_identity(config, [WEIGHTS])
    updated = compute_kv_identity(config, [retrained])

    assert original.digest != updated.digest
    assert original.components["config"] == updated.components["config"]


def test_engine_reports_identity(vllm_runner):
    """The engine identifies Hub weights by content and reports the identity
    to the frontend and on `vllm:cache_config_info`."""
    with vllm_runner(MODEL, max_model_len=256, enforce_eager=True) as runner:
        cache_config = runner.llm.llm_engine.vllm_config.cache_config

        assert cache_config.kv_identity is not None
        assert cache_config.kv_identity_components["weights_source"] == "hf_cache"
        assert cache_config.metrics_info()["kv_identity"] == cache_config.kv_identity
