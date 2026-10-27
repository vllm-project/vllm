# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_FA_GLUON: which layers it decodes on gluon, and when the Qwen3.5
config hook selects it."""

import importlib
from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


def _reload_envs():
    import vllm.envs as envs

    return importlib.reload(envs)


@pytest.fixture
def aiter_env(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops

    def apply(shuffle: bool) -> None:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MHA", "1")
        monkeypatch.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "1" if shuffle else "0")
        _reload_envs()
        rocm_aiter_ops.refresh_env_variables()

    yield apply
    monkeypatch.undo()
    _reload_envs()
    rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "head_size", "kv_cache_dtype", "shuffle", "expected"),
    [
        (24, 4, 256, "auto", True, True),
        (24, 4, 256, "auto", False, False),
        (24, 4, 256, "fp8", True, False),
        (16, 2, 256, "auto", True, False),
        (32, 8, 128, "auto", True, False),
    ],
    ids=["qwen38", "plain-kv", "fp8-kv", "gqa-8", "head-128"],
)
def test_pa_gluon_supports(
    num_heads, num_kv_heads, head_size, kv_cache_dtype, shuffle, expected, aiter_env
):
    from vllm.v1.attention.backends.rocm_aiter_fa_gluon import pa_gluon_supports

    aiter_env(shuffle)
    assert (
        pa_gluon_supports(num_heads, num_kv_heads, head_size, kv_cache_dtype)
        is expected
    )


@pytest.mark.parametrize(
    ("backend", "heads", "kv_cache_dtype", "shuffle", "expected"),
    [
        (None, (24, 4), "auto", True, "ROCM_AITER_FA_GLUON"),
        (None, (24, 4), "auto", False, None),
        ("ROCM_ATTN", (24, 4), "auto", True, "ROCM_ATTN"),
        (None, (16, 2), "auto", True, None),
        (None, (24, 4), "fp8", True, None),
    ],
    ids=["qwen38", "plain-kv", "user-backend", "gqa-8", "fp8-kv"],
)
def test_qwen3_5_config_selects_gluon_backend(
    backend, heads, kv_cache_dtype, shuffle, expected, aiter_env
):
    from vllm.config import AttentionConfig
    from vllm.model_executor.models.config import _default_rocm_aiter_fa_gluon
    from vllm.platforms.rocm import on_gfx950
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if not on_gfx950():
        pytest.skip("gfx950 only")
    aiter_env(shuffle)
    config = SimpleNamespace(
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum[backend] if backend else None
        ),
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: heads[0],
            get_num_kv_heads=lambda _: heads[1],
            get_head_size=lambda: 256,
        ),
        parallel_config=None,
        cache_config=SimpleNamespace(cache_dtype=kv_cache_dtype),
    )
    _default_rocm_aiter_fa_gluon(config)
    chosen = config.attention_config.backend
    assert (chosen.name if chosen else None) == expected
