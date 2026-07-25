# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe-not-predict completion accounting (COMPLETION.md).

The reload completion criterion (`load_numel >= load_numel_total`) produced
#44814/#37334/#38746. The root cause is that the expected side is predicted
from layer structure (`get_layer_size`, minus a hand-maintained
`SKIP_TENSORS` allowlist). This exercises the alternative: observe the
required key set from the first load, reconcile against it on reload.

Runs on CPU over the real reload machinery on synthetic layers -- no GPU, no
checkpoint. These tests cover both the historical disagreement and the
manifest-authoritative completion path that replaces copied-numel accounting.
"""

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase)
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload, finalize_load_recording,
    get_layer_completion_findings, get_layerwise_info,
    get_load_manifest_report,
    initialize_layerwise_reload, record_load_consumption,
    record_direct_load_consumption, record_dummy_load_manifest,
    record_metadata_for_reloading)
from vllm.model_executor.model_loader.reload.source import observe_weight_sources
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

    def test_direct_write_loader_records_source_and_target(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        record_direct_load_consumption(model, "0.weight", "rank0.weight")
        assert get_layerwise_info(model[0]).required_keys == {
            "rank0.weight=>weight"
        }

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

    def test_packed_expert_calls_are_distinct_events(self):
        """One packed destination still has one event per expert/shard call."""

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.packed = nn.Parameter(torch.zeros(2, 3, 4))

                def expert_loader(param, loaded_weight, weight_name, shard_id,
                                  expert_id, return_success=False):
                    shard_idx = {"w1": 0, "w2": 1, "w3": 2}[shard_id]
                    param.data[expert_id, shard_idx].copy_(loaded_weight)
                    return True if return_success else None

                self.packed.weight_loader = expert_loader

        layer = Layer()
        model = nn.Sequential(layer)
        record_load_consumption(model)
        for expert_id in range(2):
            for shard_id in ("w1", "w2", "w3"):
                layer.packed.weight_loader(
                    layer.packed,
                    torch.ones(4),
                    weight_name=f"experts.{expert_id}.{shard_id}.weight",
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
        finalize_load_recording(model)

        required = get_layerwise_info(layer).required_keys
        assert required is not None
        assert len(required) == 6
        assert any("expert_id=1" in event and "shard_id='w3'" in event
                   for event in required)

    def test_non_local_expert_is_not_required(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.packed = nn.Parameter(torch.zeros(4))

                def expert_loader(param, loaded_weight, weight_name, shard_id,
                                  expert_id, return_success=False):
                    success = expert_id == 0
                    if success:
                        param.data.copy_(loaded_weight)
                    return success if return_success else None

                self.packed.weight_loader = expert_loader

        layer = Layer()
        model = nn.Sequential(layer)
        record_load_consumption(model)
        for expert_id in (0, 1):
            layer.packed.weight_loader(
                layer.packed,
                torch.ones(4),
                weight_name=f"experts.{expert_id}.w1.weight",
                shard_id="w1",
                expert_id=expert_id,
                return_success=True,
            )
        finalize_load_recording(model)

        required = get_layerwise_info(layer).required_keys
        assert required is not None
        assert len(required) == 1
        assert "expert_id=0" in next(iter(required))

    def test_source_keys_survive_qkv_routing(self):
        """The original checkpoint keys remain attached after q/k/v merge."""

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(12, 4))

                def qkv_loader(param, loaded_weight, loaded_shard_id):
                    offset = {"q": 0, "k": 4, "v": 8}[loaded_shard_id]
                    param.data[offset:offset + 4].copy_(loaded_weight)

                self.weight.weight_loader = qkv_loader

        layer = Layer()
        model = nn.Sequential(layer)
        source_weights = [
            ("model.layers.0.self_attn.q_proj.weight", torch.ones(4, 4)),
            ("model.layers.0.self_attn.k_proj.weight", torch.ones(4, 4)),
            ("model.layers.0.self_attn.v_proj.weight", torch.ones(4, 4)),
        ]

        record_load_consumption(model)
        for (source_key, loaded_weight), shard_id in zip(
                observe_weight_sources(source_weights), ("q", "k", "v")):
            # Stand in for a model-specific mapper/routing layer. The source
            # context must remain the pre-mapping checkpoint key.
            layer.weight.weight_loader(
                layer.weight, loaded_weight, loaded_shard_id=shard_id)
        finalize_load_recording(model)

        required = get_layerwise_info(layer).required_keys
        assert required == {
            "model.layers.0.self_attn.q_proj.weight"
            "=>weight[loaded_shard_id='q']",
            "model.layers.0.self_attn.k_proj.weight"
            "=>weight[loaded_shard_id='k']",
            "model.layers.0.self_attn.v_proj.weight"
            "=>weight[loaded_shard_id='v']",
        }

    def test_source_keys_cover_one_to_one_and_merged_mlp(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.Parameter(torch.zeros(4))
                self.gate_up = nn.Parameter(torch.zeros(8, 4))

                def merged_loader(param, loaded_weight, loaded_shard_id):
                    offset = 0 if loaded_shard_id == 0 else 4
                    param.data[offset:offset + 4].copy_(loaded_weight)

                self.gate_up.weight_loader = merged_loader

        layer = Layer()
        model = nn.Sequential(layer)
        record_load_consumption(model)
        weights = observe_weight_sources([
            ("input_layernorm.weight", torch.ones(4)),
            ("mlp.gate_proj.weight", torch.ones(4, 4)),
            ("mlp.up_proj.weight", torch.ones(4, 4)),
        ])
        _, norm = next(weights)
        layer.norm.weight_loader(layer.norm, norm)
        _, gate = next(weights)
        layer.gate_up.weight_loader(
            layer.gate_up, gate, loaded_shard_id=0)
        _, up = next(weights)
        layer.gate_up.weight_loader(
            layer.gate_up, up, loaded_shard_id=1)
        finalize_load_recording(model)

        assert get_layerwise_info(layer).required_keys == {
            "input_layernorm.weight=>norm",
            "mlp.gate_proj.weight=>gate_up[loaded_shard_id=0]",
            "mlp.up_proj.weight=>gate_up[loaded_shard_id=1]",
        }

    def test_one_fused_source_can_emit_multiple_target_fragments(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.packed = nn.Parameter(torch.zeros(2, 4))

                def shard_loader(param, loaded_weight, loaded_shard_id):
                    param.data[loaded_shard_id].copy_(loaded_weight)

                self.packed.weight_loader = shard_loader

        layer = Layer()
        model = nn.Sequential(layer)
        record_load_consumption(model)
        for _, fused in observe_weight_sources([
                ("mlp.gate_up_proj.weight", torch.ones(2, 4))]):
            layer.packed.weight_loader(
                layer.packed, fused[0], loaded_shard_id=0)
            layer.packed.weight_loader(
                layer.packed, fused[1], loaded_shard_id=1)
        finalize_load_recording(model)

        assert get_layerwise_info(layer).required_keys == {
            "mlp.gate_up_proj.weight=>packed[loaded_shard_id=0]",
            "mlp.gate_up_proj.weight=>packed[loaded_shard_id=1]",
        }


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

    def test_dummy_manifest_validates_first_real_transfer(self):
        """Dummy initialization must not leave the first RL transfer unchecked."""
        m = _CountedModel()
        record_metadata_for_reloading(m)
        record_dummy_load_manifest(m)
        info = get_layerwise_info(m.layer)
        assert info.required_target_keys == {"A", "D", "dt_bias"}
        assert info.required_keys == set()

        initialize_layerwise_reload(m)
        # The old numel counter reaches its total because A's composed loader
        # counts twice, but the dummy target baseline still requires D.
        m.load_weights(observe_weight_sources([
            ("layer.A", torch.ones(4)),
            ("layer.dt_bias", torch.ones(4)),
        ]))
        with pytest.raises(RuntimeError, match="after dummy initialization"):
            finalize_layerwise_reload(m, _Cfg())

        findings = get_layer_completion_findings()
        assert any("dummy-baseline" in finding and "D" in finding
                   for finding in findings)
        assert info.required_target_keys == {"A", "D", "dt_bias"}

    def test_dummy_manifest_promotes_first_complete_real_transfer(self):
        m = _CountedModel()
        record_metadata_for_reloading(m)
        record_dummy_load_manifest(m)

        initialize_layerwise_reload(m)
        m.load_weights(observe_weight_sources([
            ("layer.A", torch.ones(4)),
            ("layer.D", torch.ones(4)),
            ("layer.dt_bias", torch.ones(4)),
        ]))
        finalize_layerwise_reload(m, _Cfg())

        info = get_layerwise_info(m.layer)
        assert get_layer_completion_findings() == []
        assert info.required_target_keys is None
        assert info.required_keys == {
            "layer.A=>A",
            "layer.D=>D",
            "layer.dt_bias=>dt_bias",
        }

    def test_dummy_target_manifest_replaces_numel_completion(self, monkeypatch):
        m = _CountedModel()
        record_metadata_for_reloading(m)
        record_dummy_load_manifest(m)
        initialize_layerwise_reload(m)

        # Even a permanently zero diagnostic counter must not prevent target
        # manifest completion during the first real transfer.
        monkeypatch.setattr(
            "vllm.model_executor.model_loader.reload.layerwise.get_numel_loaded",
            lambda loader, args: (
                0,
                loader(*args.args, **args.kwargs),
            ),
        )
        m.load_weights(observe_weight_sources([
            ("layer.A", torch.ones(4)),
            ("layer.D", torch.full((4,), 2.0)),
            ("layer.dt_bias", torch.full((4,), 3.0)),
        ]))

        # A provisional target set cannot say how many QKV/MoE fragments map
        # to one target, so the first source-bearing load remains buffered
        # until transaction finalization.
        assert get_layerwise_info(m.layer).can_load()
        finalize_layerwise_reload(m, _Cfg())
        assert torch.equal(m.layer.D, torch.full((4,), 2.0))
        assert get_layerwise_info(m.layer).required_target_keys is None

    def test_checkpoint_backed_skip_tensor_receipt_can_arrive_last(self):
        """A skipped alias remains device-resident but is still load-required.

        DeepSeek's gate.e_score_correction_bias has this shape: the ordinary
        gate weight can trigger numel completion before the correction bias
        appears later in the checkpoint stream.
        """

        class Gate(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(4, 4))
                self.e_score_correction_bias = nn.Parameter(torch.zeros(4))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = Gate()

            def load_weights(self, weights):
                return _named_param_loader(self, weights)

        m = Model()
        first = [
            ("gate.weight", torch.ones(4, 4)),
            ("gate.e_score_correction_bias", torch.arange(4.0)),
        ]
        record_load_consumption(m)
        m.load_weights(first)
        finalize_load_recording(m)
        assert get_layerwise_info(m.gate).required_keys == {
            "weight",
            "e_score_correction_bias",
        }

        record_metadata_for_reloading(m)
        initialize_layerwise_reload(m)
        # Deliberately place the skipped tensor after numel completion.
        m.load_weights(first)
        finalize_layerwise_reload(m, _Cfg())

        assert get_layer_completion_findings() == []
        torch.testing.assert_close(
            m.gate.e_score_correction_bias, torch.arange(4.0))
        assert m.gate.e_score_correction_bias.weight_loader.__name__ == (
            "default_weight_loader")

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
        report = get_load_manifest_report(m)
        assert report.ok
        assert report.required_event_count == 3
        assert report.received_event_count == 3
        # CPU tests do not initialize model-parallel groups. Reporting remains
        # usable and clearly marks the coordinates as unavailable.
        assert report.scope.global_rank is None

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
