# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Caller-level tests for target-layer-scoped adaptive finalization.

These drive init_attn_backend itself (with patched layer discovery and
builder creation) to pin the two fail-closed invariants and the
target/draft/mixed group role classification that the builder-local tests
cannot see. CPU-safe: builders are recorders.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.attn_utils as attn_utils_mod
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.worker.utils import AttentionGroup


class _FakeLayer:
    def __init__(self, num_heads: int = 32):
        self.num_heads = num_heads

    def get_attn_backend(self):
        return SimpleNamespace(full_cls_name=lambda: "FakeBackend")


class _RecorderBuilder:
    def __init__(self):
        self.calls = []

    def _finalize_adaptive_decode(self, decode_query_len, is_target=None):
        self.calls.append((decode_query_len, is_target))

    def _get_workspace_buffer(self):
        return None


def _kv_cfg(layer_names):
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )
    return SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec(layer_names=list(layer_names),
                                          kv_cache_spec=spec)]
    )


def _vc(adaptive: bool = True):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            enable_adaptive_verification=adaptive,
        ),
    )


@pytest.fixture
def patched_init(monkeypatch):
    """Patch init_attn_backend's collaborators; yield per-group recorders."""
    recorders = {}

    def fake_create_metadata_builders(self, **kwargs):
        rec = _RecorderBuilder()
        recorders[tuple(self.layer_names)] = rec
        self.metadata_builders = [rec]

    monkeypatch.setattr(
        attn_utils_mod, "get_shared_kv_cache_layers", lambda vc: []
    )
    monkeypatch.setattr(
        attn_utils_mod, "add_kv_sharing_layers_to_kv_cache_groups",
        lambda shared, groups: None,
    )
    monkeypatch.setattr(
        attn_utils_mod, "get_layers_from_vllm_config",
        lambda vc, lt, names: {
            n: _FakeLayer(8 if n.startswith("d") else 32) for n in names
        },
    )
    monkeypatch.setattr(
        attn_utils_mod, "prepare_kernel_block_sizes", lambda cfg, groups: []
    )
    monkeypatch.setattr(
        attn_utils_mod, "get_attn_cg_support", lambda groups, vc,
        checked=None: SimpleNamespace(
            min_cg_support=None, min_cg_attn_backend=None
        ),
    )
    monkeypatch.setattr(
        AttentionGroup, "create_metadata_builders",
        fake_create_metadata_builders,
    )
    return recorders


def test_missing_target_set_raises(patched_init):
    """Invariant: adaptive finalization without a target set fails closed."""
    with pytest.raises(ValueError, match="target-layer scoping"):
        attn_utils_mod.init_attn_backend(
            _kv_cfg(["t0"]), _vc(adaptive=True), torch.device("cpu"),
            decode_query_len=8, target_attn_layer_names=None,
        )


def test_empty_target_classification_raises(patched_init):
    """Invariant: a target-scoped init that classifies zero target builders
    fails closed — an adaptive target with no attention groups is
    unsupported, not silently vacuous."""
    with pytest.raises(ValueError, match="zero.*target builders"):
        attn_utils_mod.init_attn_backend(
            _kv_cfg(["d0"]), _vc(adaptive=True), torch.device("cpu"),
            decode_query_len=8, target_attn_layer_names={"nonexistent"},
        )


def test_role_classification_target_draft_mixed(patched_init):
    """Target groups arm (is_target=True), draft-only groups stay off
    (False), and a group mixing target+draft layers counts as target —
    the inverse of #52783's scan skip. Draft layers group separately via
    their distinct Q-head count."""
    rec = patched_init
    attn_utils_mod.init_attn_backend(
        _kv_cfg(["t0", "t1", "d0", "d1"]), _vc(adaptive=True),
        torch.device("cpu"),
        decode_query_len=8, target_attn_layer_names={"t0", "t1"},
    )
    by_names = {names: r.calls for names, r in rec.items()}
    target_calls = next(v for k, v in by_names.items() if "t0" in k)
    draft_calls = next(v for k, v in by_names.items() if "d0" in k)
    assert target_calls == [(8, True)]
    assert draft_calls == [(8, False)]


def test_mixed_group_finalizes_as_target(patched_init):
    """A single group containing both target and draft layers (same head
    count, so one AttentionGroup) intersects the target set and finalizes
    as target — consistent with #52783's isdisjoint scan skip treating a
    mixed group as checked."""
    rec = patched_init
    attn_utils_mod.init_attn_backend(
        _kv_cfg(["t0", "x0"]), _vc(adaptive=True), torch.device("cpu"),
        decode_query_len=8, target_attn_layer_names={"t0"},
    )
    (calls,) = [r.calls for r in rec.values()]
    assert calls == [(8, True)]


def test_no_decode_query_len_skips_finalization(patched_init):
    """The speculator's draft-scoped call (no decode_query_len) never
    finalizes builders."""
    rec = patched_init
    attn_utils_mod.init_attn_backend(
        _kv_cfg(["d0"]), _vc(adaptive=True), torch.device("cpu"),
        active_layer_names={"d0"},
    )
    assert all(r.calls == [] for r in rec.values())


def test_nonadaptive_boot_passes_without_target_set(patched_init):
    """Invariant #1 must not fire on legal non-adaptive boots."""
    rec = patched_init
    attn_utils_mod.init_attn_backend(
        _kv_cfg(["t0"]), _vc(adaptive=False), torch.device("cpu"),
        decode_query_len=8,
    )
    # finalize still runs (is_target=None) but backends gate on the
    # predicate themselves; the call itself must not raise.
    assert all(len(r.calls) == 1 for r in rec.values())
