# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft model must honour ``moe_backend`` from --speculative-config.

The draft is loaded with the target's VllmConfig, so without an explicit
override it inherits the target's --moe-backend. An MTP head on a quantized
target is typically unquantized, and quantized-only backends reject it, so the
server fails to start rather than falling back.
"""

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


@dataclass
class _KernelConfig:
    moe_backend: str | None = None


@dataclass
class _CacheConfig:
    cache_dtype: str = "auto"


@dataclass
class _SpeculativeConfig:
    moe_backend: str | None = None
    kv_cache_dtype: str | None = None
    draft_model_config: object = None


@dataclass
class _VllmConfig:
    kernel_config: _KernelConfig
    cache_config: _CacheConfig
    speculative_config: _SpeculativeConfig


def _config(target_moe: str, draft_moe: str | None) -> _VllmConfig:
    return _VllmConfig(
        kernel_config=_KernelConfig(moe_backend=target_moe),
        cache_config=_CacheConfig(),
        speculative_config=_SpeculativeConfig(moe_backend=draft_moe),
    )


class _Captured(Exception):
    """Stops load_eagle_model once we hold the config it would have used."""

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


def test_draft_moe_backend_overrides_the_target():
    """A draft moe_backend must reach the draft's kernel config."""
    used = _capture_draft_config(_config("flashinfer_b12x", "flashinfer_cutlass"))
    assert used.kernel_config.moe_backend == "flashinfer_cutlass"


def test_draft_inherits_target_when_no_override():
    """No override means the draft still inherits the target, as before."""
    used = _capture_draft_config(_config("flashinfer_b12x", None))
    assert used.kernel_config.moe_backend == "flashinfer_b12x"


def test_override_does_not_mutate_the_target_config():
    """The target must keep its own backend after the draft is built."""
    cfg = _config("flashinfer_b12x", "flashinfer_cutlass")
    _capture_draft_config(cfg)
    assert cfg.kernel_config.moe_backend == "flashinfer_b12x"
