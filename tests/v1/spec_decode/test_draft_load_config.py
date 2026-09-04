# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft must load with ``draft_load_config`` from --speculative-config.

``get_model`` resolves ``load_config=None`` to the target's, so these assert on
the value ``load_eagle_model`` hands it rather than on the loaded model.
"""

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


@dataclass
class _CacheConfig:
    cache_dtype: str = "auto"


@dataclass
class _SpeculativeConfig:
    draft_load_config: object = None
    kv_cache_dtype: str | None = None
    draft_model_config: object = None


@dataclass
class _VllmConfig:
    cache_config: _CacheConfig
    speculative_config: _SpeculativeConfig


class _Captured(Exception):
    def __init__(self, load_config):
        self.load_config = load_config


def _capture_draft_load_config(draft_load_config: object) -> object:
    cfg = _VllmConfig(
        cache_config=_CacheConfig(),
        speculative_config=_SpeculativeConfig(draft_load_config=draft_load_config),
    )

    def _fake_get_model(*, vllm_config, model_config, load_config=None):
        raise _Captured(load_config)

    with (
        patch("vllm.v1.worker.gpu.spec_decode.eagle.utils.get_model", _fake_get_model),
        pytest.raises(_Captured) as exc,
    ):
        load_eagle_model(object(), cfg)
    return exc.value.load_config


def test_draft_load_config_reaches_the_loader() -> None:
    sentinel = object()
    assert _capture_draft_load_config(sentinel) is sentinel


def test_unset_draft_load_config_defers_to_the_target() -> None:
    assert _capture_draft_load_config(None) is None
