# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration gates of Mamba2 exact-replay mode.

The mode must fail closed on every unsupported combination, and the mamba
backend selector must accept the Mamba2 backend under VLLM_BATCH_INVARIANT=1
only when the mode is on (a config-level decision, not process state).
"""

import pytest

from vllm.config import CacheConfig, SchedulerConfig, VllmConfig
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.selector import (
    _cached_get_mamba_attn_backend,
    _mamba_backend_supports_batch_invariance,
    get_mamba_attn_backend,
)


def _cache(**overrides) -> CacheConfig:
    kwargs = dict(
        mamba_exact_replay=True,
        mamba_ssm_cache_dtype="float32",
        enable_prefix_caching=False,
    )
    kwargs.update(overrides)
    return CacheConfig(**kwargs)


def _supported_config(**cache_overrides) -> VllmConfig:
    # Async scheduling is resolved on by default in some setups; the mode
    # requires it off, so pin it explicitly.
    return VllmConfig(
        cache_config=_cache(**cache_overrides),
        scheduler_config=SchedulerConfig(
            max_model_len=2048, is_encoder_decoder=False, async_scheduling=False
        ),
    )


def test_exact_replay_accepts_the_supported_configuration():
    _supported_config()


def test_exact_replay_requires_fp32_ssm_cache():
    with pytest.raises(ValueError, match="float32"):
        _supported_config(mamba_ssm_cache_dtype="auto")


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda c: setattr(c.cache_config, "mamba_cache_mode", "align"), "prefix"),
        (
            lambda c: setattr(c.cache_config, "use_replayssm", True),
            "mutually exclusive",
        ),
        (
            lambda c: setattr(c.scheduler_config, "async_scheduling", True),
            "async scheduling",
        ),
        (lambda c: setattr(c.parallel_config, "enable_dbo", True), "micro-batching"),
        (
            lambda c: setattr(c.parallel_config, "tensor_parallel_size", 2),
            "TP=1 and PP=1",
        ),
    ],
)
def test_exact_replay_fails_closed_on_unsupported_settings(mutate, message):
    # Other validators reject some of these combinations earlier at
    # construction time, so exercise this mode's validator directly.
    cfg = _supported_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        cfg.validate_mamba_exact_replay()


def test_mamba2_backend_batch_invariance_depends_on_exact_replay():
    assert Mamba2AttentionBackend.supports_batch_invariance() is False
    assert not _mamba_backend_supports_batch_invariance(Mamba2AttentionBackend, False)
    assert _mamba_backend_supports_batch_invariance(Mamba2AttentionBackend, True)


def test_mamba_backend_selector_gate(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    _cached_get_mamba_attn_backend.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="batch_invariant"):
            get_mamba_attn_backend(MambaAttentionBackendEnum.MAMBA2)
        backend = get_mamba_attn_backend(
            MambaAttentionBackendEnum.MAMBA2, mamba_exact_replay=True
        )
        assert backend is Mamba2AttentionBackend
    finally:
        _cached_get_mamba_attn_backend.cache_clear()
