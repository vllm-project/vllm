# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional dependency and vLLM-to-KVServe cache adapter contracts."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression import adapter
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.tokens import (
    chunk_token_counts,
    chunk_window,
    suffix_token_counts,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"compression": None},
        {"compression": {"enable-compression-pipeline": False, "unknown": 1}},
    ],
)
def test_disabled_compression_does_not_import_backend(monkeypatch, extra):
    def unexpected_import():
        pytest.fail("disabled compression imported the optional backend")

    monkeypatch.setattr(adapter, "load_backend", unexpected_import)
    assert adapter.get_compression_config(extra) is None
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra)
    )
    assert adapter.build_memory_plan(config, {}) is None


def test_missing_backend_fails_explicitly(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(adapter, "import_module", missing)
    with pytest.raises(ImportError, match="requires KVServe"):
        adapter.get_compression_config(
            {"compression": {"enable-compression-pipeline": True}}
        )


def test_incompatible_backend_api_is_rejected(monkeypatch):
    monkeypatch.setattr(
        adapter, "import_module", lambda name: SimpleNamespace(API_VERSION=2)
    )
    with pytest.raises(RuntimeError, match="API version"):
        adapter.load_backend()


@pytest.fixture
def planned_adapter(monkeypatch):
    if os.environ.get("VLLM_TEST_REQUIRE_MOONCAKE_COMPRESSION") != "1":
        pytest.importorskip("kvserve_v1.compression.runtime")
    backend = adapter.load_backend()
    calls = []

    def plan(config, specs, **kwargs):
        calls.append((config, specs, kwargs))
        return SimpleNamespace(reserved_bytes=123)

    monkeypatch.setattr(backend, "build_memory_plan", plan)
    monkeypatch.setattr(adapter, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(adapter, "get_current_attn_backends", lambda config: [])
    monkeypatch.setattr(
        adapter,
        "get_attn_backends_and_specs_for_layers",
        lambda config, specs, backends: ([], []),
    )
    monkeypatch.setattr(adapter, "select_common_block_size", lambda *args: 8)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "compression": {"enable-compression-pipeline": True}
            }
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=4)),
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.bfloat16
    )
    return config, spec, calls


def test_plan_translates_cache_geometry_and_excludes_raw_state(planned_adapter):
    config, spec, calls = planned_adapter
    result = adapter.build_memory_plan(
        config,
        {
            "model.layers.3.self_attn": spec,
            "model.layers.2": MambaSpec(
                block_size=16, shapes=((4, 8),), dtypes=(torch.bfloat16,), num_heads=4
            ),
        },
    )
    assert result.reserved_bytes == 123
    _, specs, kwargs = calls[0]
    assert list(specs) == ["model.layers.3.self_attn"]
    assert specs["model.layers.3.self_attn"].layer_index == 3
    assert specs["model.layers.3.self_attn"].head_size_v == 8
    assert (kwargs["blocks_per_logical"], kwargs["tp_size"], kwargs["tp_rank"]) == (
        2,
        2,
        1,
    )


@pytest.mark.parametrize(
    "changes",
    [{"tokens_per_state": 2}, {"num_head_slots": 1}, {"state_content_bytes": 24}],
)
def test_packed_or_pooled_cache_rejected_before_backend_plan(planned_adapter, changes):
    config, spec, calls = planned_adapter
    with pytest.raises(ValueError, match="dense K/V"):
        adapter.build_memory_plan(
            config, {"model.layers.0.self_attn": replace(spec, **changes)}
        )
    assert not calls


def test_mixed_logical_block_sizes_rejected_before_backend_plan(planned_adapter):
    config, spec, calls = planned_adapter
    with pytest.raises(ValueError, match="common logical block size"):
        adapter.build_memory_plan(
            config,
            {
                "model.layers.0.self_attn": spec,
                "model.layers.1.self_attn": replace(spec, block_size=32),
            },
        )
    assert not calls


def test_chunk_metadata_preserves_partial_suffix():
    assert suffix_token_counts(33, [[0, 1]], {0: 16}) == {0: 17}
    assert chunk_window(3, 3, 2) == (2, 1, 1)
    assert chunk_token_counts({0: 17}, {0: 16}, 1, 1) == {0: 1}
