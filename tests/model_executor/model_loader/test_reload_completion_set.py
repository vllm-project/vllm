# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe-not-predict completion accounting (COMPLETION.md).

The reload completion criterion (`load_numel >= load_numel_total`) produced
#44814/#37334/#38746. The root cause is that the expected side is predicted
from layer structure (`get_layer_size`, minus a hand-maintained
`SKIP_TENSORS` allowlist). This exercises the alternative: observe the
required key set from the first load, reconcile against it on reload.

Runs on CPU over the real reload machinery on synthetic layers -- no GPU, no
checkpoint. Dual-run: numel stays authoritative; the set criterion records
disagreements.
"""

import torch
from torch import nn

from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase)
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload, finalize_load_recording,
    get_layer_completion_findings, get_layerwise_info,
    initialize_layerwise_reload, record_load_consumption,
    record_metadata_for_reloading)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class _Cfg:
    model = "synthetic"
    dtype = torch.float32


def _named_param_loader(model, name_value_pairs):
    """Load via each param's own weight_loader, honoring the observation
    wrappers installed by record_load_consumption."""
    params = dict(model.named_parameters())
    loaded = set()
    for name, w in name_value_pairs:
        p = params[name]
        getattr(p, "weight_loader", default_weight_loader)(p, w)
        loaded.add(name)
    return loaded


class TestObservation:

    def test_required_excludes_never_loaded_buffer(self):
        """A buffer that is set directly (never through a loader) -- the
        _expert_map / e_score_correction_bias shape -- is absent from the
        observed required set, with no SKIP_TENSORS consulted."""

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(4))
                # bookkeeping: exists, but no weight_loader, never loaded
                self.register_buffer("expert_map", torch.zeros(4),
                                     persistent=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = Layer()

            def load_weights(self, weights):
                return _named_param_loader(self, weights)

        m = M()
        record_load_consumption(m)
        m.load_weights([("layer.weight", torch.ones(4))])
        finalize_load_recording(m)

        req = get_layerwise_info(m.layer).required_keys
        assert req == {"weight"}, req          # expert_map NOT in it
        # and the recording wrappers were removed
        assert not getattr(m.layer.weight.weight_loader, "_vllm_recording",
                           False)

    def test_required_survives_reset(self):
        """The observed baseline must persist across reloads, like
        restore_metadata."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 4)

            def load_weights(self, weights):
                return _named_param_loader(self, weights)

        m = M()
        record_load_consumption(m)
        m.load_weights([("layer.weight", torch.ones(4, 4)),
                        ("layer.bias", torch.ones(4))])
        finalize_load_recording(m)
        info = get_layerwise_info(m.layer)
        baseline = set(info.required_keys)
        assert baseline == {"weight", "bias"}

        info.reset()
        assert info.required_keys == baseline      # survived
        assert info.received_keys == set()         # cleared


class _CountedLayer(nn.Module):
    """A layer whose composed loader copies twice into one param -- the
    #44814 shape, where numel double-counts and can reach the total before a
    later param arrives."""

    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(4))
        self.D = nn.Parameter(torch.zeros(4))
        self.dt_bias = nn.Parameter(torch.zeros(4))

        def composed(param, w):        # copies twice: numel counts 2x
            param.data.copy_(w)
            param.data.copy_(-torch.exp(w.float()))

        self.A.weight_loader = composed

    class _QM(QuantizeMethodBase):
        def create_weights(self, *a, **k):
            pass

        def apply(self, *a, **k):
            pass

        def process_weights_after_loading(self, layer):
            pass

    @property
    def quant_method(self):
        return _CountedLayer._QM()


class _CountedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _CountedLayer()

    def load_weights(self, weights):
        return _named_param_loader(self, weights)


class TestDualRunCatchesDroppedKey:

    def test_set_criterion_flags_the_44814_shape(self):
        """First load consumes A, D, dt_bias. On reload, if only A and
        dt_bias arrive, numel (2n from A's double copy + n from dt_bias = 3n
        == total) says complete, but the observed set requires D -- the set
        criterion records the disagreement."""
        m = _CountedModel()
        # observe the correct baseline: all three loaded
        record_load_consumption(m)
        m.load_weights([("layer.A", torch.ones(4)),
                        ("layer.D", torch.ones(4)),
                        ("layer.dt_bias", torch.ones(4))])
        finalize_load_recording(m)
        assert get_layerwise_info(m.layer).required_keys == {
            "A", "D", "dt_bias"}

        # reload dropping D (arrives after premature numel completion)
        record_metadata_for_reloading(m)
        initialize_layerwise_reload(m)
        m.load_weights([("layer.A", torch.ones(4)),
                        ("layer.dt_bias", torch.ones(4))])
        finalize_layerwise_reload(m, _Cfg())

        findings = get_layer_completion_findings()
        assert findings, "set criterion missed the dropped key that numel let through"
        assert any("D" in f for f in findings)

    def test_clean_reload_no_disagreement(self):
        m = _CountedModel()
        record_load_consumption(m)
        m.load_weights([("layer.A", torch.ones(4)),
                        ("layer.D", torch.ones(4)),
                        ("layer.dt_bias", torch.ones(4))])
        finalize_load_recording(m)

        record_metadata_for_reloading(m)
        initialize_layerwise_reload(m)
        m.load_weights([("layer.A", torch.ones(4)),
                        ("layer.D", torch.ones(4)),
                        ("layer.dt_bias", torch.ones(4))])
        finalize_layerwise_reload(m, _Cfg())
        assert get_layer_completion_findings() == []

    def test_no_baseline_means_no_dual_run(self):
        """Without a first-load observation (required_keys is None), the
        dual-run is silent rather than guessing."""
        m = _CountedModel()   # no record_load_consumption
        record_metadata_for_reloading(m)
        initialize_layerwise_reload(m)
        m.load_weights([("layer.A", torch.ones(4)),
                        ("layer.dt_bias", torch.ones(4))])
        finalize_layerwise_reload(m, _Cfg())
        assert get_layer_completion_findings() == []
