# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import tempfile

import huggingface_hub.constants
import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader import weight_utils
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


if __name__ == "__main__":
    test_runai_model_loader()
