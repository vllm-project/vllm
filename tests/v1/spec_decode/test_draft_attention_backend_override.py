# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft must honour ``attention_backend`` from --speculative-config.

``init_attn_backend`` reads the backend off each constructed layer, so these
assert on the config ``load_eagle_model`` hands to ``get_model``.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config import CacheConfig, LoadConfig, SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


@dataclass
class _AttentionConfig:
    backend: str | None = None
    use_non_causal: bool = False


@dataclass
class _KernelConfig:
    moe_backend: str | None = None


@dataclass
class _CacheConfig:
    cache_dtype: str = "auto"


@dataclass
class _SpeculativeConfig:
    attention_backend: str | None = None
    moe_backend: str | None = None
    kv_cache_dtype: str | None = None
    draft_model_config: object = None
    draft_load_config: object = None

    apply_draft_overrides = SpeculativeConfig.apply_draft_overrides


@dataclass
class _VllmConfig:
    attention_config: _AttentionConfig
    kernel_config: _KernelConfig
    cache_config: _CacheConfig
    speculative_config: _SpeculativeConfig
    load_config: LoadConfig = field(default_factory=LoadConfig)


def _config(target_backend: str, draft_backend: str | None) -> _VllmConfig:
    return _VllmConfig(
        attention_config=_AttentionConfig(backend=target_backend),
        kernel_config=_KernelConfig(),
        cache_config=_CacheConfig(),
        speculative_config=_SpeculativeConfig(attention_backend=draft_backend),
    )


class _Captured(Exception):
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config


def _capture_draft_config(cfg):
    def _fake_get_model(*, vllm_config, model_config):
        raise _Captured(vllm_config)

    with (
        patch("vllm.v1.worker.gpu.spec_decode.eagle.utils.get_model", _fake_get_model),
        patch(
            "vllm.v1.worker.gpu.spec_decode.utils.get_pp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        pytest.raises(_Captured) as exc,
    ):
        load_eagle_model(object(), cfg)
    return exc.value.vllm_config


def test_draft_attention_backend_overrides_the_target():
    used = _capture_draft_config(_config("FLASHINFER", "TRITON_ATTN"))
    assert used.attention_config.backend == "TRITON_ATTN"


def test_unset_leaves_the_target_backend_in_place():
    """Clearing it lets the draft autoselect a KV layout the target lacks."""
    cfg = _config("FLEX_ATTENTION", None)
    used = _capture_draft_config(cfg)
    assert used is cfg
    assert used.attention_config.backend == "FLEX_ATTENTION"


def test_override_does_not_mutate_the_target_config():
    cfg = _config("FLASHINFER", "TRITON_ATTN")
    _capture_draft_config(cfg)
    assert cfg.attention_config.backend == "FLASHINFER"


def test_draft_attn_builders_get_overrides_and_keep_resolved_cache_state():
    """The draft's metadata builders read the same kernel overrides as its
    layers, without losing state the engine core resolved on the target, e.g.
    an fp8 draft next to an NVFP4 target with a resolved KV cache layout."""
    cache_config = CacheConfig(cache_dtype="nvfp4")
    cache_config.kv_cache_layout = "HND"
    cfg = _VllmConfig(
        attention_config=_AttentionConfig(backend="FLASHINFER"),
        kernel_config=_KernelConfig(),
        cache_config=cache_config,
        speculative_config=_SpeculativeConfig(
            attention_backend="TRITON_ATTN", kv_cache_dtype="fp8"
        ),
    )
    speculator = object.__new__(DFlashSpeculator)
    speculator.vllm_config = cfg
    speculator.speculative_config = cfg.speculative_config
    speculator.requires_non_causal = True

    used = speculator.attn_vllm_config

    assert used.attention_config.backend == "TRITON_ATTN"
    assert used.attention_config.use_non_causal
    assert used.cache_config.cache_dtype == "fp8"
    assert used.cache_config.kv_cache_layout == "HND"
    assert cfg.attention_config.backend == "FLASHINFER"
    assert cfg.cache_config.cache_dtype == "nvfp4"
