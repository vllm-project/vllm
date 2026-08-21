# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the extensible --kv-cache-dtype registry.

Run `pytest tests/config/test_kv_cache_dtype.py`.
"""

import logging
import re
from pathlib import Path

import pytest
import torch

from vllm.config.cache import (
    KV_CACHE_DTYPES,
    CacheConfig,
    get_kv_cache_dtype_handler,
    is_known_kv_cache_dtype,
    register_kv_cache_dtype,
)
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    is_quantized_kv_cache,
    kv_cache_uses_per_token_head_scales,
)
from vllm.v1.kv_cache_interface import (
    KVQuantMode,
    get_kv_quant_mode,
)

pytestmark = pytest.mark.skip_global_cleanup


@register_kv_cache_dtype("test-c8")
class TestC8Handler:
    """Backend-managed custom dtype used by the tests below."""

    name = "test-c8"

    def torch_dtype(self):
        return torch.int8

    def is_quantized(self):
        return True

    def quant_mode(self):
        return KVQuantMode.BACKEND


@register_kv_cache_dtype("test-ith")
class TestIthHandler:
    """Custom dtype reusing the INT8 per-token-head kernel path."""

    name = "test-ith"

    def torch_dtype(self):
        return torch.int8

    def is_quantized(self):
        return True

    def quant_mode(self):
        return KVQuantMode.INT8_PER_TOKEN_HEAD


def test_register_kv_cache_dtype():
    """Registration appends to the mutable lists and injects the dtype."""
    assert "test-c8" in KV_CACHE_DTYPES
    assert STR_DTYPE_TO_TORCH_DTYPE["test-c8"] == torch.int8

    handler = get_kv_cache_dtype_handler("test-c8")
    assert handler is not None
    assert handler.quant_mode() == KVQuantMode.BACKEND
    assert handler.is_quantized() is True

    assert is_known_kv_cache_dtype("test-c8") is True
    assert is_known_kv_cache_dtype("auto") is True
    assert is_known_kv_cache_dtype("definitely-not-a-dtype") is False

    # Upstream dtypes have no handler and keep their existing mapping.
    assert get_kv_cache_dtype_handler("fp8") is None
    assert get_kv_quant_mode("fp8") == KVQuantMode.FP8_PER_TENSOR
    assert get_kv_quant_mode("test-c8") == KVQuantMode.BACKEND


def test_merged_quantized_helpers():
    """The shared helpers answer for custom dtypes via their handlers."""
    assert is_quantized_kv_cache("test-c8") is True
    assert kv_cache_uses_per_token_head_scales("test-c8") is False

    assert is_quantized_kv_cache("test-ith") is True
    assert kv_cache_uses_per_token_head_scales("test-ith") is True

    # String fallback for upstream dtypes is unchanged.
    assert is_quantized_kv_cache("fp8_e4m3") is True
    assert is_quantized_kv_cache("float16") is False
    assert kv_cache_uses_per_token_head_scales("int4_per_token_head") is True


def test_re_register_kv_cache_dtype_warns(caplog_vllm):
    """Re-registering an existing dtype warns and overwrites."""
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        register_kv_cache_dtype("test-c8")(TestC8Handler)

    assert any(
        "The kv-cache-dtype 'test-c8' already exists" in message
        for message in caplog_vllm.messages
    )


def test_cache_config_accepts_custom_dtype():
    """Custom dtypes pass Pydantic validation; unknown names fail fast."""
    cache_config = CacheConfig(cache_dtype="test-c8")
    assert cache_config.cache_dtype == "test-c8"

    with pytest.raises(ValueError, match="Invalid kv_cache_dtype: 'not-a-dtype'"):
        CacheConfig(cache_dtype="not-a-dtype")


def test_backend_sentinel_properties():
    """BACKEND reads as quantized but matches no generic kernel mode."""
    mode = KVQuantMode.BACKEND
    assert mode != KVQuantMode.NONE
    assert mode.is_per_token_head is False
    assert mode.is_nvfp4 is False


def test_no_snapshot_of_shared_dtype_registry():
    """The registration-time-mutable registries must never be snapshotted.

    Snapshots (dict(...), .copy(), list(.keys()), etc.) would silently miss
    dtypes registered later by platform backends.
    """
    vllm_root = Path(__file__).resolve().parents[2] / "vllm"
    patterns = [
        re.compile(r"dict\(\s*STR_DTYPE_TO_TORCH_DTYPE\s*\)"),
        re.compile(r"STR_DTYPE_TO_TORCH_DTYPE\.copy\(\)"),
        re.compile(r"list\(\s*STR_DTYPE_TO_TORCH_DTYPE"),
        re.compile(r"tuple\(\s*STR_DTYPE_TO_TORCH_DTYPE"),
        re.compile(r"STR_DTYPE_TO_TORCH_DTYPE\.keys\(\)"),
        re.compile(r"list\(\s*KV_CACHE_DTYPES"),
        re.compile(r"tuple\(\s*KV_CACHE_DTYPES"),
    ]
    offenders: dict[str, list[str]] = {}
    for path in vllm_root.rglob("*.py"):
        text = path.read_text()
        for pattern in patterns:
            if pattern.search(text):
                offenders.setdefault(pattern.pattern, []).append(str(path))
    assert not offenders, f"Shared dtype registry snapshots found: {offenders}"
