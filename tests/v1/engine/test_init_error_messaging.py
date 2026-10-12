# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.core.kv_cache_utils import check_enough_kv_cache_memory
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _fake_layer(backend_name: str) -> MagicMock:
    layer = MagicMock()
    layer.get_attn_backend.return_value.get_name.return_value = backend_name
    return layer


def _oom_config(static_forward_context: dict) -> MagicMock:
    config = MagicMock()
    config.model_config.max_model_len = 2048
    config.cache_config.block_size = 16
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.compilation_config.static_forward_context = static_forward_context
    return config


def _spec_for(layer_names: list[str]) -> dict[str, FullAttentionSpec]:
    return {
        name: FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
        )
        for name in layer_names
    }


def test_kv_cache_oom_no_memory():
    from unittest.mock import MagicMock

    config = MagicMock()
    config.model_config.max_model_len = 2048

    spec = {
        "layer_0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
        )
    }

    with pytest.raises(ValueError):
        check_enough_kv_cache_memory(config, spec, 0)


def test_kv_cache_oom_insufficient_memory(monkeypatch):
    from unittest.mock import MagicMock

    config = MagicMock()
    config.model_config.max_model_len = 2048
    config.cache_config.block_size = 16
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1

    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_utils.max_memory_usage_bytes",
        lambda c, s: 100 * 1024**3,  # 100 GiB
    )

    spec = {
        "layer_0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
        )
    }

    with pytest.raises(ValueError):
        check_enough_kv_cache_memory(config, spec, 1024**3)  # 1 GiB


def test_oom_message_names_resolved_backend(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_utils.max_memory_usage_bytes",
        lambda c, s: 100 * 1024**3,
    )
    config = _oom_config({"layer_0": _fake_layer("FLASHINFER")})
    spec = _spec_for(["layer_0"])

    with pytest.raises(ValueError) as excinfo:
        check_enough_kv_cache_memory(config, spec, 1024**3)

    assert "FLASHINFER" in str(excinfo.value)


def test_oom_message_lists_distinct_backends(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_utils.max_memory_usage_bytes",
        lambda c, s: 100 * 1024**3,
    )
    config = _oom_config(
        {
            "layer_0": _fake_layer("FLASHINFER"),
            "layer_1": _fake_layer("FLASH_ATTN"),
            "layer_2": _fake_layer("FLASHINFER"),
        }
    )
    spec = _spec_for(["layer_0", "layer_1", "layer_2"])

    with pytest.raises(ValueError) as excinfo:
        check_enough_kv_cache_memory(config, spec, 1024**3)

    message = str(excinfo.value)
    assert "FLASHINFER" in message
    assert "FLASH_ATTN" in message


def test_oom_message_omits_backend_when_unresolved(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_utils.max_memory_usage_bytes",
        lambda c, s: 100 * 1024**3,
    )
    config = _oom_config({})
    spec = _spec_for(["layer_0"])

    with pytest.raises(ValueError) as excinfo:
        check_enough_kv_cache_memory(config, spec, 1024**3)

    assert "attention backend" not in str(excinfo.value)
