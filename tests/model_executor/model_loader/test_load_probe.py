# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.load_receipt import returns_load_receipt
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_load_recording,
    get_layerwise_info,
    record_load_consumption,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.probe import (
    LoadProbeError,
    probe_dummy_load_manifest,
    probe_model_load,
    safetensors_meta_weights,
    validate_probe_receipt_coverage,
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
            assert name.startswith("layer.")
            param = self.get_parameter(name)
            param.weight_loader(param, value)
            loaded.add(name)
        return loaded


def meta_weight(size=8):
    return torch.empty(size, dtype=torch.float32, device="meta")


def test_probe_runs_real_loader_without_changing_parameter():
    model = ProbeModel()
    before = model.layer.weight.detach().clone()

    report = probe_model_load(model, [("layer.weight", meta_weight())])

    assert report.ok
    assert report.loaded_weights == {"layer.weight"}
    assert report.intercepted_writes == ["aten.copy_.default"]
    assert report.write_sources == {"layer.weight"}
    assert torch.equal(model.layer.weight, before)


def test_probe_receipt_matches_real_source_and_fragment():
    @returns_load_receipt("loaded_shard_id")
    def shard_loader(param, loaded_weight, loaded_shard_id):
        param.copy_(loaded_weight)

    model = ProbeModel(shard_loader)
    record_load_consumption(model)
    try:
        with pytest.raises(LoadProbeError, match="PROBE_LOADER_EXCEPTION"):
            # The model does not supply the required routing argument. A
            # probe executes the real call signature rather than inventing a
            # default fragment.
            probe_model_load(model, [("layer.weight", meta_weight())])
    finally:
        finalize_load_recording(model)


def test_probe_records_routed_fragment_from_real_model_mapping():
    @returns_load_receipt("loaded_shard_id")
    def shard_loader(param, loaded_weight, loaded_shard_id):
        param.copy_(loaded_weight)

    class RoutedProbeModel(ProbeModel):
        def load_weights(self, weights):
            for name, value in weights:
                source, shard = name.rsplit(".", 1)
                assert source == "layer.weight"
                self.layer.weight.weight_loader(
                    self.layer.weight,
                    value,
                    shard,
                )
            return {"layer.weight"}

    model = RoutedProbeModel(shard_loader)
    record_metadata_for_reloading(model)
    info = get_layerwise_info(model.layer)
    info.required_keys = set()
    info.required_target_keys = {"weight"}

    probe_dummy_load_manifest(
        model,
        [
            ("layer.weight.q", meta_weight()),
            ("layer.weight.k", meta_weight()),
            ("layer.weight.v", meta_weight()),
        ],
    )

    assert info.required_keys == {
        "layer.weight.k=>weight[loaded_shard_id='k']",
        "layer.weight.q=>weight[loaded_shard_id='q']",
        "layer.weight.v=>weight[loaded_shard_id='v']",
    }


def test_probe_preserves_composed_loader_write_count():
    def composed_loader(param, loaded_weight):
        param.copy_(loaded_weight)
        param.copy_(loaded_weight)

    model = ProbeModel(composed_loader)
    report = probe_model_load(model, [("layer.weight", meta_weight())])

    assert report.intercepted_writes == [
        "aten.copy_.default",
        "aten.copy_.default",
    ]


def test_probe_redirects_loader_factories_to_meta():
    def allocating_loader(param, loaded_weight):
        temporary = torch.zeros(
            loaded_weight.shape,
            dtype=loaded_weight.dtype,
            device=param.device,
        )
        param.copy_(loaded_weight + temporary)

    model = ProbeModel(allocating_loader)
    report = probe_model_load(model, [("layer.weight", meta_weight())])

    assert report.ok
    assert report.intercepted_writes == ["aten.copy_.default"]


def test_probe_rejects_real_source_storage():
    model = ProbeModel()
    with pytest.raises(ValueError, match="requires meta source tensors"):
        probe_model_load(model, [("layer.weight", torch.empty(8))])


def test_probe_fails_closed_on_data_dependent_loader():
    def data_dependent_loader(param, loaded_weight):
        if loaded_weight.sum().item() > 0:
            param.copy_(loaded_weight)

    model = ProbeModel(data_dependent_loader)
    with pytest.raises(LoadProbeError, match="PROBE_UNSUPPORTED_OPERATOR"):
        probe_model_load(model, [("layer.weight", meta_weight())])


def test_probe_can_establish_exact_recorded_event():
    model = ProbeModel()
    record_load_consumption(model)
    try:
        report = probe_model_load(model, [("layer.weight", meta_weight())])
    finally:
        finalize_load_recording(model)

    assert report.ok
    info = get_layerwise_info(model.layer)
    assert info.required_keys == {"layer.weight=>weight"}


def test_probe_rejects_direct_write_without_receipt():
    class DirectWriteModel(ProbeModel):
        def load_weights(self, weights):
            for name, value in weights:
                self.get_parameter(name).copy_(value)
            return {"layer.weight"}

    model = DirectWriteModel()
    record_load_consumption(model)
    try:
        report = probe_model_load(model, [("layer.weight", meta_weight())])
        with pytest.raises(
            LoadProbeError,
            match="without LoadReceipt coverage",
        ):
            validate_probe_receipt_coverage(model, report)
    finally:
        finalize_load_recording(model)


def test_dummy_probe_replaces_provisional_target_baseline():
    model = ProbeModel()
    record_metadata_for_reloading(model)
    info = get_layerwise_info(model.layer)
    info.required_keys = set()
    info.required_target_keys = {"weight"}

    report = probe_dummy_load_manifest(
        model,
        [("layer.weight", meta_weight())],
    )

    assert report.ok
    assert info.required_target_keys is None
    assert info.required_keys == {"layer.weight=>weight"}


def test_failed_dummy_probe_restores_provisional_baseline():
    def data_dependent_loader(param, loaded_weight):
        if loaded_weight.sum().item() > 0:
            param.copy_(loaded_weight)

    model = ProbeModel(data_dependent_loader)
    record_metadata_for_reloading(model)
    info = get_layerwise_info(model.layer)
    info.required_keys = set()
    info.required_target_keys = {"weight"}

    with pytest.raises(LoadProbeError):
        probe_dummy_load_manifest(
            model,
            [("layer.weight", meta_weight())],
        )

    assert info.required_target_keys == {"weight"}
    assert info.required_keys == set()


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
    assert weights["layer.weight"].dtype == torch.float32
    assert weights["layer.scale"].is_meta
    assert weights["layer.scale"].dtype == torch.bfloat16


def test_dummy_finalize_keeps_exact_probe_manifest():
    model = ProbeModel()
    info = get_layerwise_info(model.layer)
    info.required_keys = {"layer.weight=>weight"}
    info.required_target_keys = None

    DummyModelLoader.finalize_load_manifest(object(), model)

    assert info.required_keys == {"layer.weight=>weight"}
    assert info.required_target_keys is None


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
        calls.append(
            (
                model_name_or_path,
                cache_dir,
                allow_patterns,
                revision,
                subfolder,
                ignore_patterns,
            )
        )
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
        "vllm.model_executor.model_loader.dummy_loader."
        "maybe_download_from_modelscope",
        lambda *args, **kwargs: None,
    )

    loader = DummyModelLoader(
        LoadConfig(
            load_format="dummy",
            download_dir="/tmp/vllm-test-cache",
            ignore_patterns=["original/**/*"],
            model_loader_extra_config={"enable_load_probe": True},
        )
    )
    model_config = type(
        "ModelConfigStub",
        (),
        {"model": "org/model", "revision": "main"},
    )()

    weights = dict(loader._probe_meta_weights(model_config))

    assert calls == [
        (
            "org/model",
            "/tmp/vllm-test-cache",
            ["*.safetensors"],
            "main",
            None,
            ["original/**/*"],
        )
    ]
    assert weights["layer.weight"].is_meta
    assert weights["layer.weight"].shape == (8,)
