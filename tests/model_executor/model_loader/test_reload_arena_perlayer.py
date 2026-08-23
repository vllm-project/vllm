# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer arena verification inside the layerwise reload pipeline.

The arena's storage-identity check used to run only at the model level, in
`reload_weights`. It now also runs per layer via
`LayerReloadingInfo.arena_snapshot`: snapshotted in
`initialize_layerwise_reload`, verified as each layer's PWAL re-runs, and
reset with the rest of the per-layer info. This exercises that path over the
real reload machinery on a synthetic layer -- no GPU, no checkpoint.

The field is VERIFY-ONLY: unlike `kernel_tensors`, arena slots are never
restored-to-meta or copied back. The tests pin both that a stable arena
passes and that a rebinding arena is caught.
"""

import torch
from torch import nn

from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase)
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload, get_layer_arena_findings,
    initialize_layerwise_reload, record_metadata_for_reloading)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.reload_arena import (
    InitPolicy, get_reload_arena, snapshot_model_arenas,
    verify_model_arenas)


class _ArenaLayer(nn.Module):
    """A layer that owns a checkpoint param plus arena storage rebuilt in
    process_weights_after_loading -- the migrated-backend shape."""

    def __init__(self, drift: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4))
        self._drift = drift
        self._materialize_scratch()

    def _materialize_scratch(self):
        # stands in for a first-forward lazy allocation
        arena = get_reload_arena(self)
        self.scratch = arena.get_or_alloc(
            "scratch", (8,), torch.int32, "cpu", init=InitPolicy.PRESERVE)
        self.derived = arena.put("derived", self.weight.detach().sum().view(1))

    class _QM(QuantizeMethodBase):
        def __init__(self, outer):
            self.outer = outer

        def create_weights(self, *a, **k):
            pass

        def apply(self, *a, **k):
            pass

        def process_weights_after_loading(self, layer):
            arena = get_reload_arena(layer)
            if layer._drift:
                # The arena's own slot storage moves between snapshot and
                # verify -- the shape verify is a tripwire for (e.g. an
                # experts object rebuilt with freshly-allocated scratch that
                # replaced the arena slot). This is what the check catches;
                # a backend that bypasses the arena entirely (rebinds only
                # the attribute) is NOT caught here -- see
                # test_bypassing_the_arena_is_a_known_blind_spot.
                arena._slots["scratch"] = torch.empty(8, dtype=torch.int32)
                arena._slots["derived"] = torch.zeros(1)
            else:
                layer.scratch = arena.get_or_alloc(
                    "scratch", (8,), torch.int32, "cpu",
                    init=InitPolicy.PRESERVE)
                layer.derived = arena.put(
                    "derived", layer.weight.detach().sum().view(1))

    @property
    def quant_method(self):
        return _ArenaLayer._QM(self)


class _Model(nn.Module):
    def __init__(self, drift: bool):
        super().__init__()
        self.layer = _ArenaLayer(drift)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, w in weights:
            p = params[name]
            getattr(p, "weight_loader", default_weight_loader)(p, w)
            loaded.add(name)
        return loaded


class _Cfg:
    model = "synthetic"
    dtype = torch.float32


def _reload(model):
    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    model.load_weights([("layer.weight", torch.ones(4, 4))])
    finalize_layerwise_reload(model, _Cfg())


def test_snapshot_taken_and_reset():
    """arena_snapshot is populated at initialize and cleared by reset."""
    from vllm.model_executor.model_loader.reload.layerwise import (
        get_layerwise_info)
    model = _Model(drift=False)
    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    info = get_layerwise_info(model.layer)
    assert info.arena_snapshot is not None
    assert set(info.arena_snapshot) == {"scratch", "derived"}
    model.load_weights([("layer.weight", torch.ones(4, 4))])
    finalize_layerwise_reload(model, _Cfg())
    # reset after finalize
    assert get_layerwise_info(model.layer).arena_snapshot is None


def test_stable_arena_reports_no_findings():
    model = _Model(drift=False)
    _reload(model)
    assert get_layer_arena_findings() == []


def test_rebinding_arena_is_caught_per_layer():
    model = _Model(drift=True)
    _reload(model)
    findings = get_layer_arena_findings()
    assert findings, "per-layer verify missed a rebinding arena"
    assert any("scratch" in f or "derived" in f for f in findings)
    assert all(f.startswith("layer:") for f in findings)


def test_per_layer_findings_match_model_level_finding_set():
    model = _Model(drift=True)
    snaps = snapshot_model_arenas(model)
    _reload(model)

    assert set(get_layer_arena_findings()) == set(
        verify_model_arenas(model, snaps))


def test_findings_cleared_between_reloads():
    model = _Model(drift=True)
    _reload(model)
    assert get_layer_arena_findings()          # dirty
    clean = _Model(drift=False)
    _reload(clean)
    assert get_layer_arena_findings() == []     # fresh reload cleared them


def test_bypassing_the_arena_is_a_known_blind_spot():
    """Honest scope: the runtime arena verify checks the ARENA's slots, so a
    backend that rebinds a layer attribute to non-arena storage without
    touching the arena is NOT caught here. That failure mode is the CI
    sweep's job (test_post_load_storage_stability censuses layer attributes,
    not arena slots). This test pins the boundary so a future reader does
    not assume the runtime check covers it."""

    class _BypassLayer(_ArenaLayer):
        class _QM(QuantizeMethodBase):
            def __init__(self, outer):
                self.outer = outer

            def create_weights(self, *a, **k):
                pass

            def apply(self, *a, **k):
                pass

            def process_weights_after_loading(self, layer):
                # bypass: fresh storage on the attribute, arena untouched
                layer.scratch = torch.empty(8, dtype=torch.int32)

        @property
        def quant_method(self):
            return _BypassLayer._QM(self)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = _BypassLayer(drift=False)

        def load_weights(self, weights):
            p = self.layer.weight
            for name, w in weights:
                getattr(p, "weight_loader", default_weight_loader)(p, w)
            return {n for n, _ in weights}

    m = M()
    _reload(m)
    # arena slots never moved, so the runtime verify is (correctly) silent
    assert get_layer_arena_findings() == []


def test_layer_without_arena_is_a_noop():
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, 4))

        def load_weights(self, weights):
            for name, w in weights:
                dict(self.named_parameters())[name].data.copy_(w)
            return {n for n, _ in weights}

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Plain()

        def load_weights(self, weights):
            return self.layer.load_weights(
                [(n.split(".", 1)[1], w) for n, w in weights])

    m = M()
    record_metadata_for_reloading(m)
    initialize_layerwise_reload(m)
    model_load = m.load_weights([("layer.weight", torch.ones(4, 4))])  # noqa
    finalize_layerwise_reload(m, _Cfg())
    assert get_layer_arena_findings() == []
