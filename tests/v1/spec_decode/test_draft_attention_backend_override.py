# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft must honour ``attention_backend`` from --speculative-config.

``init_attn_backend`` reads the backend off each constructed layer, so these
assert on the config ``load_eagle_model`` hands to ``get_model``.
"""

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


@dataclass
class _AttentionConfig:
    backend: str | None = None


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


@dataclass
class _VllmConfig:
    attention_config: _AttentionConfig
    kernel_config: _KernelConfig
    cache_config: _CacheConfig
    speculative_config: _SpeculativeConfig


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
