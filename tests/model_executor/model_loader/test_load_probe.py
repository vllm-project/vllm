# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    freeze_load_plan,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.plan import get_load_plan
from vllm.model_executor.model_loader.reload.probe import (
    LoadProbeError,
    probe_model_load,
    safetensors_meta_weights,
    validate_probe_plan_coverage,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class ProbeModel(torch.nn.Module):
    def __init__(self, loader=default_weight_loader):
        super().__init__()
        self.layer = torch.nn.Module()
        weight = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
        weight.weight_loader = loader
        self.layer.register_parameter("weight", weight)

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            param = self.get_parameter(name)
            param.weight_loader(param, value)
            loaded.add(name)
        return loaded


def meta_weight(size=8):
    return torch.empty(size, dtype=torch.float32, device="meta")


def test_probe_records_plan_without_changing_parameter():
    model = ProbeModel()
    record_metadata_for_reloading(model)
    before = model.layer.weight.detach().clone()

    report = probe_model_load(model, [("layer.weight", meta_weight())])
    validate_probe_plan_coverage(model, report)
    freeze_load_plan(model)

    assert report.loaded_weights == {"layer.weight"}
    assert report.intercepted_writes == ["aten.copy_.default"]
    assert torch.equal(model.layer.weight, before)
    assert get_load_plan(model.layer) == {
        ("layer.weight", "weight", ()): 1,
    }


def test_probe_records_selector_from_real_routing():
    def shard_loader(param, loaded_weight, loaded_shard_id):
        param.copy_(loaded_weight)

    class RoutedModel(ProbeModel):
        def load_weights(self, weights):
            for name, value in weights:
                source, shard = name.rsplit(".", 1)
                assert source == "layer.weight"
                self.layer.weight.weight_loader(self.layer.weight, value, shard)
            return {"layer.weight"}

    model = RoutedModel(shard_loader)
    record_metadata_for_reloading(model)
    report = probe_model_load(
        model,
        [
            ("layer.weight.q", meta_weight()),
            ("layer.weight.k", meta_weight()),
        ],
    )
    validate_probe_plan_coverage(model, report)
    freeze_load_plan(model)

    assert get_load_plan(model.layer) == {
        ("layer.weight.q", "weight", (("loaded_shard_id", "q"),)): 1,
        ("layer.weight.k", "weight", (("loaded_shard_id", "k"),)): 1,
    }


def test_probe_records_expert_selector_and_omits_declined_load():
    def expert_loader(param, loaded_weight, expert_id, return_success=False):
        if expert_id != 0:
            return False if return_success else None
        param.copy_(loaded_weight)
        return True if return_success else None

    class ExpertModel(ProbeModel):
        def load_weights(self, weights):
            for name, value in weights:
                expert_id = int(name.rsplit(".", 1)[1])
                self.layer.weight.weight_loader(
                    self.layer.weight,
                    value,
                    expert_id,
                    return_success=True,
                )

    model = ExpertModel(expert_loader)
    record_metadata_for_reloading(model)
    report = probe_model_load(
        model,
        [("layer.weight.0", meta_weight()), ("layer.weight.1", meta_weight())],
    )
    validate_probe_plan_coverage(model, report)
    freeze_load_plan(model)

    assert get_load_plan(model.layer) == {
        ("layer.weight.0", "weight", (("expert_id", 0),)): 1,
    }


def test_probe_rejects_direct_write_without_plan_key():
    class DirectWriteModel(ProbeModel):
        def load_weights(self, weights):
            for name, value in weights:
                self.get_parameter(name).copy_(value)
            return {"layer.weight"}

    model = DirectWriteModel()
    record_metadata_for_reloading(model)
    report = probe_model_load(model, [("layer.weight", meta_weight())])

    with pytest.raises(LoadProbeError, match="without LoadPlan coverage"):
        validate_probe_plan_coverage(model, report)


def test_probe_fails_closed_on_data_dependent_loader():
    def data_dependent_loader(param, loaded_weight):
        if loaded_weight.sum().item() > 0:
            param.copy_(loaded_weight)

    model = ProbeModel(data_dependent_loader)
    record_metadata_for_reloading(model)
    with pytest.raises(LoadProbeError, match="PROBE_UNSUPPORTED_OPERATOR"):
        probe_model_load(model, [("layer.weight", meta_weight())])


def test_safetensors_schema_reader_returns_meta_tensors(tmp_path):
    save_file(
        {
            "layer.weight": torch.ones(8, dtype=torch.float32),
            "layer.scale": torch.ones(2, dtype=torch.bfloat16),
        },
        tmp_path / "model.safetensors",
    )

    weights = dict(safetensors_meta_weights(str(tmp_path)))

    assert set(weights) == {"layer.weight", "layer.scale"}
    assert weights["layer.weight"].is_meta
    assert weights["layer.weight"].shape == (8,)
    assert weights["layer.scale"].dtype == torch.bfloat16


def test_dummy_loader_probe_is_explicitly_configured():
    default_loader = DummyModelLoader(LoadConfig(load_format="dummy"))
    enabled_loader = DummyModelLoader(
        LoadConfig(
            load_format="dummy",
            model_loader_extra_config={"enable_load_probe": True},
        )
    )

    assert not default_loader.enable_load_probe
    assert enabled_loader.enable_load_probe


@pytest.mark.parametrize(
    "extra_config, match",
    [
        ({"enable_load_probe": "yes"}, "enable_load_probe must be a bool"),
        ({"unknown": True}, "Unexpected extra config keys"),
    ],
)
def test_dummy_loader_rejects_invalid_probe_config(extra_config, match):
    with pytest.raises(ValueError, match=match):
        DummyModelLoader(
            LoadConfig(
                load_format="dummy",
                model_loader_extra_config=extra_config,
            )
        )


def test_dummy_loader_probe_resolves_remote_safetensors(monkeypatch, tmp_path):
    save_file({"layer.weight": torch.ones(8)}, tmp_path / "model.safetensors")
    calls = []

    def fake_download_weights_from_hf(
        model_name_or_path,
        cache_dir,
        allow_patterns,
        revision=None,
        subfolder=None,
        ignore_patterns=None,
    ):
        calls.append((model_name_or_path, allow_patterns, revision))
        return str(tmp_path)

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.dummy_loader.download_weights_from_hf",
        fake_download_weights_from_hf,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.dummy_loader."
        "download_safetensors_index_file_from_hf",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.dummy_loader.maybe_download_from_modelscope",
        lambda *args, **kwargs: None,
    )

    loader = DummyModelLoader(
        LoadConfig(
            load_format="dummy",
            model_loader_extra_config={"enable_load_probe": True},
        )
    )
    model_config = type(
        "ModelConfigStub",
        (),
        {"model": "org/model", "revision": "main"},
    )()

    weights = dict(loader._probe_meta_weights(model_config))

    assert calls == [("org/model", ["*.safetensors"], "main")]
    assert weights["layer.weight"].is_meta
