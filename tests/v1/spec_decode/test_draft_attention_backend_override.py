# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft model must honour ``attention_backend`` from --speculative-config.

The V1 proposer sets the draft's attention backend from the speculative config
and never inherits the target's, because draft and target attention shapes
differ and not every backend supports both. On V2 the setting was dropped.

The override has to be applied before ``get_model()``: ``init_attn_backend``
reads the backend off each constructed layer via ``get_attn_backend()`` rather
than off the config, so anything applied after construction is inert. These
tests therefore assert on the config that ``load_eagle_model`` actually hands
to ``get_model``.
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
    """Stops load_eagle_model once we hold the config it would have built with."""

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
    """An explicit draft backend must reach the config the draft is built with."""
    used = _capture_draft_config(_config("FLASHINFER", "TRITON_ATTN"))
    assert used.attention_config.backend == "TRITON_ATTN"


def test_unset_clears_the_backend_so_the_draft_autoselects():
    """Unset must clear the target's backend, not inherit it.

    The V1 proposer assigns the draft's backend unconditionally so that a
    ``None`` erases the target's and the draft autoselects independently. It
    never inherits, because draft and target attention shapes differ and not
    every backend serves both.
    """
    used = _capture_draft_config(_config("FLASHINFER", None))
    assert used.attention_config.backend is None


def test_override_does_not_mutate_the_target_config():
    """The target must keep its own backend after the draft config is derived."""
    cfg = _config("FLASHINFER", "TRITON_ATTN")
    _capture_draft_config(cfg)
    assert cfg.attention_config.backend == "FLASHINFER"
