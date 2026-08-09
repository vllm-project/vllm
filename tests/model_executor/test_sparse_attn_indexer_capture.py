# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.sparse_attn_indexer_capturer import (
    IndexerTopkCapturer,
    _get_num_indexer_layers,
    get_indexer_shape,
    get_sparse_attn_indexers,
)


def test_num_indexer_layers_with_short_pattern():
    config = SimpleNamespace(
        index_topk=128,
        num_hidden_layers=4,
        index_topk_pattern="SS",
    )

    assert _get_num_indexer_layers(config) == 2


def test_capture_discovers_target_indexers_only(monkeypatch):
    class DummyIndexer:
        pass

    target_indexer = DummyIndexer()
    draft_indexer = DummyIndexer()
    target_model = SimpleNamespace(modules=lambda: [target_indexer])
    static_forward_context = {
        "model.layers.0.indexer": target_indexer,
        "mtp.layers.0.indexer": draft_indexer,
    }

    indexer_module = ModuleType("vllm.model_executor.layers.sparse_attn_indexer")
    indexer_module.SparseAttnIndexer = DummyIndexer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, indexer_module.__name__, indexer_module)

    assert get_sparse_attn_indexers(target_model) == [target_indexer]
    assert draft_indexer in static_forward_context.values()


def test_indexer_capture_requires_complete_and_matching_layers():
    capturer = IndexerTopkCapturer(
        max_num_batched_tokens=4,
        num_indexer_layers=2,
        index_topk=3,
        device="cpu",
    )
    capturer.begin_step()
    capturer.capture(0, torch.ones(2, 3, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="missed indexer layers"):
        capturer.validate_step()

    capturer.capture(1, torch.full((2, 3), 2, dtype=torch.int32))
    capturer.validate_step()
    torch.testing.assert_close(
        capturer.get_device_buffer()[:2, 1],
        torch.full((2, 3), 2, dtype=torch.int32),
    )


def test_indexer_capture_rejects_invalid_tensor_shape_and_layer():
    capturer = IndexerTopkCapturer(4, 1, 3, "cpu")
    with pytest.raises(IndexError):
        capturer.capture(-1, torch.ones(1, 3, dtype=torch.int32))
    with pytest.raises(ValueError, match="must have shape"):
        capturer.capture(0, torch.ones(1, 2, dtype=torch.int32))


def test_indexer_shape_requires_positive_config():
    config = SimpleNamespace(
        index_topk=0,
        num_hidden_layers=4,
        index_topk_pattern=None,
        index_topk_freq=1,
        index_skip_topk_offset=2,
    )
    with pytest.raises(ValueError, match="positive"):
        get_indexer_shape(config)
