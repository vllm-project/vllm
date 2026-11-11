# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft model must honour `moe_backend` from --speculative-config.

The draft is loaded with the target's VllmConfig, so without an explicit
override it inherits the target's --moe-backend. An MTP head on a quantized
target is typically unquantized, and quantized-only backends reject it, so the
server fails to start rather than falling back.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config import LoadConfig, SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


@dataclass
class _KernelConfig:
    moe_backend: str | None = None


@dataclass
class _CacheConfig:
    cache_dtype: str = "auto"


@dataclass
class _ParallelConfig:
    all2all_backend: str = "allgather_reducescatter"


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
    kernel_config: _KernelConfig
    cache_config: _CacheConfig
    speculative_config: _SpeculativeConfig
    load_config: LoadConfig = field(default_factory=LoadConfig)
    parallel_config: _ParallelConfig = field(default_factory=_ParallelConfig)


def _config(
    target_moe: str,
    draft_moe: str | None,
    all2all_backend: str = "allgather_reducescatter",
) -> _VllmConfig:
    return _VllmConfig(
        kernel_config=_KernelConfig(moe_backend=target_moe),
        cache_config=_CacheConfig(),
        speculative_config=_SpeculativeConfig(moe_backend=draft_moe),
        parallel_config=_ParallelConfig(all2all_backend=all2all_backend),
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
        patch(
            "vllm.v1.worker.gpu.spec_decode.utils.get_pp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
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


def test_draft_cannot_drop_the_targets_passthrough_binding():
    """A self-dispatching target binds all2all=passthrough for the process.

    A draft that selects a backend using the framework all-to-all cannot run
    on that binding, so it must be rejected with both backends named instead
    of failing on a half-overridden config.
    """
    cfg = _config(
        "flashinfer_moe_ep_cutedsl",
        "flashinfer_cutlass",
        all2all_backend="passthrough",
    )
    with pytest.raises(ValueError, match="relies on the framework all-to-all"):
        _capture_draft_config(cfg)
    assert cfg.kernel_config.moe_backend == "flashinfer_moe_ep_cutedsl"
    assert cfg.parallel_config.all2all_backend == "passthrough"


def test_draft_keeps_passthrough_when_it_needs_it_too():
    """Inheriting or sharing the self-dispatching backend stays valid."""
    inherited = _capture_draft_config(
        _config("flashinfer_moe_ep_cutedsl", None, all2all_backend="passthrough")
    )
    assert inherited.parallel_config.all2all_backend == "passthrough"

    shared = _capture_draft_config(
        _config(
            "flashinfer_moe_ep_cutedsl",
            "flashinfer_moe_ep_mega_deep_gemm",
            all2all_backend="passthrough",
        )
    )
    assert shared.parallel_config.all2all_backend == "passthrough"


def test_draft_override_keeps_the_kv_cache_dtype():
    """The shared helper still owns the cache-dtype override on the draft."""
    cfg = _config("flashinfer_b12x", None)
    cfg.speculative_config.kv_cache_dtype = "fp8"
    used = _capture_draft_config(cfg)
    assert used.cache_config.cache_dtype == "fp8"
    assert cfg.cache_config.cache_dtype == "auto"
