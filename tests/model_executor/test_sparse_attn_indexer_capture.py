# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

from vllm.model_executor.layers.sparse_attn_indexer_capturer import (
    _get_num_indexer_layers,
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
