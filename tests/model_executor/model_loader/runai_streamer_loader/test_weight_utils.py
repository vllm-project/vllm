# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import os
import tempfile

import huggingface_hub.constants
import pytest
import torch
from safetensors.torch import save_file
from transformers import GPT2Config

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader, weight_utils
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    runai_safetensors_weights_iterator,
    safetensors_weights_iterator,
)


def test_runai_safetensors_weights_iterator_yields_owned_tensors(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNAI_STREAMER_MEMORY_LIMIT", "0")
    weights_file = tmp_path / "model.safetensors"
    expected_tensors = {
        "first": torch.tensor([1.0, 2.0]),
        "second": torch.tensor([3.0, 4.0]),
    }
    save_file(expected_tensors, weights_file)

    clone_count = 0
    original_clone = torch.Tensor.clone

    def counting_clone(self, *args, **kwargs):
        nonlocal clone_count
        clone_count += 1
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counting_clone)
    actual_tensors = dict(
        runai_safetensors_weights_iterator([str(weights_file)], False)
    )

    assert clone_count == 0
    assert actual_tensors.keys() == expected_tensors.keys()
    assert actual_tensors["first"].data_ptr() != actual_tensors["second"].data_ptr()
    for name, expected_tensor in expected_tensors.items():
        assert torch.equal(actual_tensors[name], expected_tensor)


class _FakeStreamer:
    """Streams two tensors through one shared buffer, like runai<0.17.0."""

    def __init__(self):
        self.files_to_tensors_metadata = {0: ["first", "second"]}
        self.stream_kwargs = None
        self.buffer = torch.empty(2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_tensors(self):
        for name, value in (("first", 1.0), ("second", 3.0)):
            self.buffer.fill_(value)
            yield name, self.buffer


class _FakeOwnedStreamer(_FakeStreamer):
    """Yields a new tensor each step, like runai>=0.17.0 with owned=True."""

    def stream_files(self, paths, device, is_distributed, owned=True):
        self.stream_kwargs = {"owned": owned}

    def get_tensors(self):
        for name, tensor in super().get_tensors():
            yield name, tensor.clone()


class _FakeLegacyStreamer(_FakeStreamer):
    def stream_files(self, paths, device, is_distributed):
        self.stream_kwargs = {}


@pytest.mark.parametrize(
    ("streamer_cls", "expected_kwargs"),
    [(_FakeOwnedStreamer, {"owned": True}), (_FakeLegacyStreamer, {})],
)
def test_runai_safetensors_weights_iterator_streamer_versions(
    monkeypatch, streamer_cls, expected_kwargs
):
    """Weights stay distinct for streamers with and without `owned` support."""
    streamer = streamer_cls()
    monkeypatch.setattr(weight_utils, "SafetensorsStreamer", lambda: streamer)

    tensors = dict(runai_safetensors_weights_iterator(["unused"], False))

    assert streamer.stream_kwargs == expected_kwargs
    assert torch.equal(tensors["first"], torch.tensor([1.0, 1.0]))
    assert torch.equal(tensors["second"], torch.tensor([3.0, 3.0]))


def test_runai_model_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "openai-community/gpt2", allow_patterns=["*.safetensors"], cache_dir=tmpdir
        )
        safetensors = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
        assert len(safetensors) > 0

        runai_model_streamer_tensors = {}
        hf_safetensors_tensors = {}

        for name, tensor in runai_safetensors_weights_iterator(safetensors, True):
            runai_model_streamer_tensors[name] = tensor

        for name, tensor in safetensors_weights_iterator(safetensors, True):
            hf_safetensors_tensors[name] = tensor

        assert len(runai_model_streamer_tensors) == len(hf_safetensors_tensors)

        for name, runai_tensor in runai_model_streamer_tensors.items():
            assert runai_tensor.dtype == hf_safetensors_tensors[name].dtype
            assert runai_tensor.shape == hf_safetensors_tensors[name].shape
            assert torch.all(runai_tensor.eq(hf_safetensors_tensors[name]))


def _cpu_checkpoint(tmp_path):
    GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        n_positions=32,
        vocab_size=16,
        bos_token_id=0,
        eos_token_id=1,
        architectures=["GPT2LMHeadModel"],
    ).save_pretrained(tmp_path)
    return ModelConfig(model=str(tmp_path), skip_tokenizer_init=True, max_model_len=32)


def _cpu_loader(shard, **extra):
    return get_model_loader(
        LoadConfig(
            load_format="runai_streamer",
            use_tqdm_on_load=False,
            model_loader_extra_config={
                "distributed": True,
                "cpu_weight_files": [shard],
                "cpu_memory_limit": 4096,
                **extra,
            },
        )
    )


def test_runai_cpu_shards_reuse_ring_without_corrupting_retained_weights(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RUNAI_STREAMER_MEMORY_LIMIT", "64")
    path = tmp_path / "cpu.safetensors"
    expected = {
        f"model.layers.{layer}.engram.{name}": torch.arange(1024, dtype=torch.float32)
        + (2 * layer + index) * 1024
        for layer in range(4)
        for index, name in enumerate(("embed.weight", "wkv.weight"))
    }
    save_file(expected, path)
    model_config = _cpu_checkpoint(tmp_path)
    loader = _cpu_loader(path.name)
    retained = {}
    for name, tensor in loader.get_all_weights(model_config, torch.nn.Module()):
        assert torch.equal(tensor, expected[name])
        if ".engram.embed." not in name:
            retained[name] = tensor
    assert retained
    for name, tensor in retained.items():
        assert torch.equal(tensor, expected[name])
    assert os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] == "64"


class _RetainingCpuWeightsModel(torch.nn.Module):
    def __init__(self, expected):
        super().__init__()
        self.keys = {name: f"weight_{i}" for i, name in enumerate(expected)}
        self.weights = torch.nn.ParameterDict(
            {
                self.keys[name]: torch.nn.Parameter(
                    torch.empty_like(tensor), requires_grad=False
                )
                for name, tensor in expected.items()
            }
        )

    def load_weights(self, weights):
        for name, tensor in weights:
            self.weights[self.keys[name]] = torch.nn.Parameter(
                tensor, requires_grad=False
            )


class _CopyingCpuWeightsModel(_RetainingCpuWeightsModel):
    def __init__(self, expected):
        super().__init__(expected)
        self.borrowed_inputs = []

    def can_borrow_checkpoint_weight(self, name):
        return name.startswith("copy_")

    def load_weights(self, weights):
        for name, tensor in weights:
            if self.can_borrow_checkpoint_weight(name):
                self.borrowed_inputs.append(tensor)
                self.weights[self.keys[name]].data.copy_(tensor)
            else:
                self.weights[self.keys[name]] = torch.nn.Parameter(
                    tensor, requires_grad=False
                )


@pytest.mark.parametrize(
    "model_type", [_RetainingCpuWeightsModel, _CopyingCpuWeightsModel]
)
def test_runai_cpu_shards_ownership_comes_from_model(tmp_path, monkeypatch, model_type):
    monkeypatch.setenv("RUNAI_STREAMER_MEMORY_LIMIT", "64")
    expected = {
        f"{mode}_{i}": torch.arange(65536, dtype=torch.float32) + i * 65536
        for mode in ("copy", "retain")
        for i in range(8)
    }
    path = tmp_path / "cpu.safetensors"
    save_file(expected, path)
    model = model_type(expected)
    loader = _cpu_loader(path.name, cpu_memory_limit=65536 * 4)
    loader.load_weights(model, _cpu_checkpoint(tmp_path))
    for name, tensor in expected.items():
        assert torch.equal(model.weights[model.keys[name]], tensor)
    if isinstance(model, _CopyingCpuWeightsModel):
        # Holding these views keeps owned tensors distinct. Borrowed views
        # point into the reused ring; inspect addresses, not expired data.
        assert len({t.data_ptr() for t in model.borrowed_inputs}) < len(
            model.borrowed_inputs
        )
    assert os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] == "64"


def test_runai_cpu_shards_restore_limit_when_consumer_closes(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNAI_STREAMER_MEMORY_LIMIT", "64")
    path = tmp_path / "cpu.safetensors"
    save_file({"first": torch.ones(1024), "second": torch.zeros(1024)}, path)
    loader = _cpu_loader(path.name)
    stream = loader.get_all_weights(_cpu_checkpoint(tmp_path), torch.nn.Module())
    next(stream)
    assert os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] == "4096"
    stream.close()
    assert os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] == "64"


@pytest.mark.parametrize(
    "extra",
    [
        {"cpu_weight_files": ["cpu.safetensors"]},
        {"cpu_memory_limit": 4096},
        {"cpu_weight_files": "cpu.safetensors", "cpu_memory_limit": 4096},
        {"cpu_weight_files": ["../cpu.safetensors"], "cpu_memory_limit": 4096},
        {"cpu_weight_files": ["cpu.safetensors"], "cpu_memory_limit": True},
        {"cpu_weight_files": ["cpu.safetensors"], "cpu_memory_limit": 0},
        {
            "distributed": False,
            "cpu_weight_files": ["cpu.safetensors"],
            "cpu_memory_limit": 4096,
        },
    ],
)
def test_runai_cpu_shards_validate_before_changing_environment(monkeypatch, extra):
    monkeypatch.setenv("RUNAI_STREAMER_MEMORY_LIMIT", "64")
    with pytest.raises(ValueError, match="cpu_"):
        get_model_loader(
            LoadConfig(
                model_loader_extra_config={
                    "distributed": True,
                    "memory_limit": 128,
                    **extra,
                },
                load_format="runai_streamer",
            )
        )
    assert os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] == "64"


def test_runai_cpu_shards_missing_file_fails_before_streaming(tmp_path):
    save_file({"weight": torch.ones(8)}, tmp_path / "present.safetensors")
    loader = _cpu_loader("missing.safetensors")
    with pytest.raises(ValueError, match=r"missing\.safetensors"):
        list(loader.get_all_weights(_cpu_checkpoint(tmp_path), torch.nn.Module()))


if __name__ == "__main__":
    test_runai_model_loader()
