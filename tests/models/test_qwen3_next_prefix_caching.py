# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3Next declares SupportsMambaPrefixCaching: mamba_cache_mode='all'
(per-block SSM checkpointing for GDN layers) is enabled instead of being
silently downgraded to 'align' or rejected at model init."""

from types import SimpleNamespace

from vllm.model_executor.models.config import MambaModelConfig
from vllm.model_executor.models.interfaces import supports_mamba_prefix_caching
from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM


def test_qwen3_next_declares_mamba_prefix_caching_support():
    assert supports_mamba_prefix_caching(Qwen3NextForCausalLM)


def _fake_vllm_config(mamba_cache_mode, supports):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            supports_mamba_prefix_caching=supports,
            architecture="Qwen3NextForCausalLM",
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            mamba_cache_mode=mamba_cache_mode,
            mamba_block_size=None,
            block_size=16,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )


def test_supported_model_defaults_to_all_mode():
    """With prefix caching on and no explicit mode, a supporting model gets
    'all' (previously Qwen3Next fell to 'align')."""
    cfg = _fake_vllm_config("none", supports=True)
    MambaModelConfig.verify_and_update_config(cfg)
    assert cfg.cache_config.mamba_cache_mode == "all"


def test_supported_model_keeps_explicit_all_mode():
    """Explicit --mamba-cache-mode=all is no longer downgraded."""
    cfg = _fake_vllm_config("all", supports=True)
    MambaModelConfig.verify_and_update_config(cfg)
    assert cfg.cache_config.mamba_cache_mode == "all"


def test_unsupported_model_still_downgrades_to_align():
    """REGRESSION: models without the interface keep the align fallback."""
    cfg = _fake_vllm_config("all", supports=False)
    MambaModelConfig.verify_and_update_config(cfg)
    assert cfg.cache_config.mamba_cache_mode == "align"
