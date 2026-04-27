# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GPUWorker.update_weights routing + FusedMoE sharded load checks.

These tests avoid spinning up the full vLLM runtime (no distributed init, no real
model) by exercising the pure-Python routing and validation surfaces added in
support of EP-sharded weight loading.
"""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE


class _FakeMoE(FusedMoE):
    """Minimal FusedMoE stand-in for callback routing tests.

    Subclasses FusedMoE so ``isinstance(m, FusedMoE)`` in the callback matches,
    but bypasses FusedMoE.__init__ (which requires a full vLLM runtime).
    """

    def __init__(self, layer_name: str) -> None:  # type: ignore[no-redef]
        torch.nn.Module.__init__(self)
        self.layer_name = layer_name
        self.sharded_calls: list[tuple[list[tuple[str, torch.Tensor]], dict]] = []

    def load_routed_expert_weights(self, weights, expert_ids_map):
        self.sharded_calls.append((list(weights), dict(expert_ids_map)))
        # Mimic the real generator: yield a synthetic param name per call.
        for expert_name, _ in self.sharded_calls[-1][0]:
            yield f"{self.layer_name}.{expert_name}.loaded"


class _FakeModel:
    """Minimal nn.Module-ish object exposing `modules()` and `load_weights`."""

    def __init__(self, moes: list[_FakeMoE]) -> None:
        self._moes = moes
        self.passthrough_calls: list[list[tuple[str, torch.Tensor]]] = []

    def modules(self):
        yield from self._moes

    def load_weights(self, weights):
        self.passthrough_calls.append(list(weights))
        # Return a set, exercising the set-return branch of _collect.
        return {name for name, _ in self.passthrough_calls[-1]}


def _build_callback(model, ids_map, loaded_names):
    """Re-implement the builder on a stub self to avoid instantiating GPUWorker.

    We only need `_make_routed_load_callback`, which does not touch worker
    internals other than importing FusedMoE at call time. Import the unbound
    method and pass a dummy self.
    """
    from vllm.v1.worker.gpu_worker import Worker

    # The method accesses `self` purely to namespace — no self attrs used.
    return Worker._make_routed_load_callback(
        None,  # type: ignore[arg-type]
        model,  # type: ignore[arg-type]
        ids_map,
        loaded_names_out=loaded_names,
    )


def test_callback_passthrough_when_ids_map_none():
    model = _FakeModel([])
    loaded: set[str] = set()
    cb = _build_callback(model, None, loaded)

    weights = [("a", torch.zeros(1)), ("b", torch.zeros(1))]
    cb(weights)

    # model.load_weights is called exactly once with the full list.
    assert len(model.passthrough_calls) == 1
    assert [n for n, _ in model.passthrough_calls[0]] == ["a", "b"]
    assert loaded == {"a", "b"}


def test_callback_routes_sharded_weights_longest_prefix():
    # Outer layer name is a prefix of inner layer name; longest prefix wins.
    outer = _FakeMoE("model.layers.0.mlp.experts")
    inner = _FakeMoE("model.layers.0.mlp.experts.mtp.experts")
    model = _FakeModel([outer, inner])
    loaded: set[str] = set()

    ids_map = {
        "model.layers.0.mlp.experts.gate_up_proj": [4, 5],
        "model.layers.0.mlp.experts.mtp.experts.gate_up_proj": [0, 1],
    }
    cb = _build_callback(model, ids_map, loaded)

    shared_tensor = torch.zeros(1)
    cb(
        [
            ("model.layers.0.mlp.experts.gate_up_proj", shared_tensor),
            ("model.layers.0.mlp.experts.mtp.experts.gate_up_proj", shared_tensor),
            ("model.layers.0.mlp.gate.weight", shared_tensor),  # passthrough
        ]
    )

    # Outer MoE got only its direct weight (not the mtp one).
    assert len(outer.sharded_calls) == 1
    assert outer.sharded_calls[0][0][0][0] == "gate_up_proj"
    assert outer.sharded_calls[0][1] == {"gate_up_proj": [4, 5]}

    # Inner MoE got its own weight via longest-prefix match.
    assert len(inner.sharded_calls) == 1
    assert inner.sharded_calls[0][0][0][0] == "gate_up_proj"
    assert inner.sharded_calls[0][1] == {"gate_up_proj": [0, 1]}

    # Passthrough weight went through model.load_weights.
    assert len(model.passthrough_calls) == 1
    assert model.passthrough_calls[0][0][0] == "model.layers.0.mlp.gate.weight"

    # loaded_names captures both sharded yields and passthrough return set.
    assert "model.layers.0.mlp.experts.gate_up_proj.loaded" in loaded
    assert "model.layers.0.mlp.experts.mtp.experts.gate_up_proj.loaded" in loaded
    assert "model.layers.0.mlp.gate.weight" in loaded


def test_callback_preserves_arrival_order():
    """Order-sensitive dispatch: GGUF weight_type must precede its data."""
    moe = _FakeMoE("model.layers.0.mlp.experts")
    model = _FakeModel([moe])

    # Make model.load_weights record the *order* it saw names.
    seen_order: list[str] = []

    def recording_load_weights(weights):
        for n, _ in weights:
            seen_order.append(n)
        return {n for n, _ in weights}

    model.load_weights = recording_load_weights  # type: ignore[assignment]

    loaded: set[str] = set()
    ids_map = {"model.layers.0.mlp.experts.gate_up_proj": [0]}
    cb = _build_callback(model, ids_map, loaded)

    # Interleave passthrough and sharded; the callback must not reorder them.
    t = torch.zeros(1)
    cb(
        [
            ("fake_weight_type", t),
            ("fake_weight_data", t),
            ("model.layers.0.mlp.experts.gate_up_proj", t),
            ("fake_trailing", t),
        ]
    )

    # Passthrough observed strictly in original order (without the sharded entry).
    assert seen_order == [
        "fake_weight_type",
        "fake_weight_data",
        "fake_trailing",
    ]
    # The sharded dispatch happened between the two passthrough blocks, once.
    assert len(moe.sharded_calls) == 1


def test_collect_handles_none_return():
    class _QuietModel(_FakeModel):
        def load_weights(self, weights):
            return None  # legacy loaders that write eagerly and return nothing

    model = _QuietModel([])
    loaded: set[str] = set()
    cb = _build_callback(model, None, loaded)
    cb([("x", torch.zeros(1))])
    # Nothing was added, but no exception either.
    assert loaded == set()


def test_collect_does_not_swallow_typeerror_from_generator():
    """A generator that raises TypeError mid-iteration must surface, not be
    silently absorbed by `_collect`."""

    class _BoomMoE(_FakeMoE):
        def load_routed_expert_weights(self, weights, expert_ids_map):
            yield "ok"
            raise TypeError("real bug inside loader")

    moe = _BoomMoE("layer0")
    model = _FakeModel([moe])
    loaded: set[str] = set()
    ids_map = {"layer0.gate_up_proj": [0]}
    cb = _build_callback(model, ids_map, loaded)

    with pytest.raises(TypeError, match="real bug inside loader"):
        cb([("layer0.gate_up_proj", torch.zeros(1))])


# --- load_routed_expert_weights input-validation tests (no real FusedMoE init) ---


class _StubFusedMoE(FusedMoE):
    """Skip FusedMoE.__init__; only test pure-Python validation branches of
    load_routed_expert_weights without needing a real model layer.

    ``FusedMoE.load_routed_expert_weights`` looks up
    ``getattr(self, param_name_r)`` where ``param_name_r`` is the full
    dotted path from the mapping (e.g. ``"w13_weight"``). We register a dummy
    tensor attribute so the lookup succeeds and we can reach the shape/len
    validation branches.
    """

    def __init__(self, layer_name: str, expert_mapping):
        torch.nn.Module.__init__(self)
        self.layer_name = layer_name
        self.expert_mapping = expert_mapping
        # Provide a dummy attribute matching the mapping param_name so
        # `getattr(self, param_name_r)` succeeds before validation runs.
        for param_name, _weight_name, _expert_id, _shard_id in expert_mapping:
            # param_name_r equals param_name here because qual_name strips
            # only the "{layer_name}." prefix.
            setattr(self, param_name, torch.zeros(1))

    def weight_loader(self, **kwargs):  # pragma: no cover
        return True


# Use a mapping that mirrors FusedMoE's flat attribute layout.
_FLAT_EXPERT_MAPPING = [
    ("w13_weight", "gate_up_proj", 0, "w1"),
    ("w13_weight", "gate_up_proj", 1, "w3"),
    ("w2_weight", "down_proj", 0, "w2"),
]


def test_load_routed_expert_weights_missing_key_raises():
    moe = _StubFusedMoE("layer0", expert_mapping=_FLAT_EXPERT_MAPPING)
    it = moe.load_routed_expert_weights(
        [("gate_up_proj", torch.zeros(2, 4, 8))], expert_ids_map={}
    )
    with pytest.raises(ValueError, match="missing from expert_ids_map"):
        list(it)


def test_load_routed_expert_weights_shape_mismatch_raises():
    moe = _StubFusedMoE("layer0", expert_mapping=_FLAT_EXPERT_MAPPING)
    # tensor has 3 experts but expert_ids has 2
    it = moe.load_routed_expert_weights(
        [("gate_up_proj", torch.zeros(3, 4, 8))],
        expert_ids_map={"gate_up_proj": [10, 11]},
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        list(it)


def test_load_routed_expert_weights_non_3d_requires_single_id():
    moe = _StubFusedMoE("layer0", expert_mapping=_FLAT_EXPERT_MAPPING)
    # 2D input with 2 ids should error (protects against global-scale tensors
    # accidentally routed through the sharded path).
    it = moe.load_routed_expert_weights(
        [("gate_up_proj", torch.zeros(4, 8))],
        expert_ids_map={"gate_up_proj": [10, 11]},
    )
    with pytest.raises(ValueError, match="requires exactly 1 expert_id"):
        list(it)


def test_load_routed_expert_weights_suffix_match_no_substring_false_positive():
    """``gate_up_proj`` must not match a weight named
    ``gate_up_proj.weight_scale``."""
    moe = _StubFusedMoE("layer0", expert_mapping=_FLAT_EXPERT_MAPPING)
    # Caller sends a weight whose name contains the base as a substring but is
    # *not* the base weight. With suffix-component matching, the loop
    # should find no mapping entry and yield nothing.
    it = moe.load_routed_expert_weights(
        [("gate_up_proj.weight_scale", torch.zeros(2, 4, 8))],
        expert_ids_map={"gate_up_proj.weight_scale": [10, 11]},
    )
    assert list(it) == []
