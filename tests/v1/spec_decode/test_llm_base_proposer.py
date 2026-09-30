# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SpecDecodeBaseProposer.initialize_attn_backend.

Block tables are stored at kernel-block granularity, so the proposer's
``block_size`` (used for slot-mapping math) must be the kernel block size,
not the KV cache manager's block size — the two differ when manager blocks
are split for the attention kernel. The value must also be deterministic:
``_draft_attn_layer_names`` is a set, whose iteration order varies across
processes, so anything derived from iteration order must not leak into
``block_size``.
"""

from types import SimpleNamespace

import pytest

import vllm.v1.spec_decode.llm_base_proposer as llm_base_proposer
from vllm.v1.spec_decode.eagle import EagleProposer

SCHEDULER_BLOCK_SIZE = 256
KERNEL_BLOCK_SIZE = 64


class _FakeAttentionGroup:
    def __init__(self, backend, layer_names, kv_cache_spec, kv_cache_group_id):
        self.backend = backend
        self.layer_names = list(layer_names)
        self.kv_cache_spec = kv_cache_spec
        self.kv_cache_group_id = kv_cache_group_id
        self.kernel_block_size = None

    def create_metadata_builders(self, vllm_config, device, kernel_block_size=None):
        self.kernel_block_size = kernel_block_size

    def get_metadata_builder(self):
        return SimpleNamespace(kv_cache_spec=self.kv_cache_spec)


def _make_proposer(
    monkeypatch: pytest.MonkeyPatch, layer_names: set[str]
) -> EagleProposer:
    fake_layers = {}
    for name in layer_names:
        backend = SimpleNamespace(full_cls_name=lambda: "FakeBackend")
        fake_layers[name] = SimpleNamespace(
            get_attn_backend=lambda backend=backend: backend
        )
    monkeypatch.setattr(
        llm_base_proposer, "get_layers_from_vllm_config", lambda *a, **k: fake_layers
    )
    monkeypatch.setattr(llm_base_proposer, "AttentionGroup", _FakeAttentionGroup)

    proposer = EagleProposer.__new__(EagleProposer)
    proposer.vllm_config = None
    proposer.device = None
    proposer._draft_attn_layer_names = set(layer_names)
    proposer.kv_cache_gid = -1
    proposer.draft_attn_groups = []
    proposer.block_size = -1
    return proposer


def _make_kv_cache_config(layer_names: set[str]) -> SimpleNamespace:
    spec = SimpleNamespace(block_size=SCHEDULER_BLOCK_SIZE)
    group = SimpleNamespace(layer_names=list(layer_names), kv_cache_spec=spec)
    return SimpleNamespace(kv_cache_groups=[group])


def test_block_size_uses_kernel_block_size(monkeypatch: pytest.MonkeyPatch):
    """The proposer's slot-mapping math runs against the kernel-granularity
    block table, so block_size must come from kernel_block_sizes."""
    layer_names = {"draft.0.self_attn.attn"}
    proposer = _make_proposer(monkeypatch, layer_names)

    proposer.initialize_attn_backend(
        _make_kv_cache_config(layer_names),
        kernel_block_sizes=[KERNEL_BLOCK_SIZE],
    )

    assert proposer.block_size == KERNEL_BLOCK_SIZE
    assert proposer.block_size != SCHEDULER_BLOCK_SIZE
    # The metadata builder keeps receiving the kernel block size as well.
    assert proposer.draft_attn_groups[0].kernel_block_size == KERNEL_BLOCK_SIZE


def test_block_size_falls_back_to_kv_cache_spec(monkeypatch: pytest.MonkeyPatch):
    layer_names = {"draft.0.self_attn.attn"}
    proposer = _make_proposer(monkeypatch, layer_names)

    proposer.initialize_attn_backend(
        _make_kv_cache_config(layer_names), kernel_block_sizes=None
    )

    assert proposer.block_size == SCHEDULER_BLOCK_SIZE


def test_draft_layer_iteration_is_deterministic(monkeypatch: pytest.MonkeyPatch):
    """_draft_attn_layer_names is a set; the attention groups built from it
    must not depend on its (process-random) iteration order."""
    layer_names = {"draft.c.attn", "draft.a.attn", "draft.b.attn"}
    expected_order = sorted(layer_names)

    for insertion_order in (expected_order, expected_order[::-1]):
        proposer = _make_proposer(monkeypatch, set(insertion_order))
        proposer.initialize_attn_backend(
            _make_kv_cache_config(set(insertion_order)),
            kernel_block_sizes=[KERNEL_BLOCK_SIZE],
        )
        assert len(proposer.draft_attn_groups) == 1
        assert proposer.draft_attn_groups[0].layer_names == expected_order
        assert proposer.block_size == KERNEL_BLOCK_SIZE
