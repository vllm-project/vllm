# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import os
import tempfile
from types import SimpleNamespace

import huggingface_hub.constants
import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    download_weights_from_hf,
    runai_safetensors_weights_iterator,
    safetensors_weights_iterator,
)


@pytest.mark.parametrize(
    "distributed,shared_cpu", [(False, False), (True, False), (True, True)]
)
def test_explicit_cpu_shards_load_every_weight_before_buffer_reuse(
    monkeypatch, distributed, shared_cpu
):
    files = ["s3://weights/regular.safetensors", "s3://weights/engram.safetensors"]
    source = {
        files[0]: ("regular.weight", torch.tensor([3.0])),
        files[1]: ("layers.1.engram.embed.weight", torch.tensor([7.0])),
    }
    requests = []
    reader_groups = []
    world_group = object()

    class Streamer:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def stream_files(self, paths, *, device, is_distributed, process_group):
            self.paths = paths
            reader_groups.append(process_group)
            requests.append(
                (
                    paths,
                    device,
                    is_distributed,
                    os.environ["RUNAI_STREAMER_MEMORY_LIMIT"],
                )
            )
            self.files_to_tensors_metadata = {i: [None] for i in range(len(paths))}

        def get_tensors(self):
            for path in self.paths:
                name, tensor = source[path]
                yield name, tensor
                tensor.fill_(99.0)

    monkeypatch.setenv("VLLM_RUNAI_CPU_WEIGHT_FILES", "engram.safetensors")
    monkeypatch.setenv("RUNAI_STREAMER_MEMORY_LIMIT", "4294967296")
    monkeypatch.setenv("VLLM_RUNAI_CPU_MEMORY_LIMIT", "107374182400")
    monkeypatch.setattr(weight_utils, "SafetensorsStreamer", Streamer)
    monkeypatch.setattr(weight_utils.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(weight_utils.current_platform, "current_device", lambda: 0)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "group", SimpleNamespace(WORLD=world_group))
    cpu_group = SimpleNamespace(
        rank_in_group=0, world_size=8, broadcast_object=lambda value: value
    )
    loaded = {}
    for name, tensor in runai_safetensors_weights_iterator(
        files,
        False,
        distributed,
        cpu_weights_group=cpu_group if shared_cpu else None,
        is_shared_cpu_weight=lambda _: True,
    ):
        param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        default_weight_loader(param, tensor)
        loaded[name] = param

    assert set(loaded) == {"regular.weight", "layers.1.engram.embed.weight"}
    assert loaded["regular.weight"].item() == 3.0
    assert loaded["layers.1.engram.embed.weight"].item() == 7.0
    assert requests == (
        [
            ([files[0]], "cuda:0", True, "4294967296"),
            ([files[1]], "cpu", False, "107374182400"),
        ]
        if distributed
        else [(files, "cpu", False, "4294967296")]
    )
    assert os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] == "4294967296"
    assert reader_groups == (
        [world_group, None if shared_cpu else world_group]
        if distributed
        else [world_group]
    )


def test_runai_safetensors_weights_iterator_clones_reused_buffers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RUNAI_STREAMER_MEMORY_LIMIT", "0")
    weights_file = tmp_path / "model.safetensors"
    expected_tensors = {
        "first": torch.tensor([1.0, 2.0]),
        "second": torch.tensor([3.0, 4.0]),
    }
    save_file(expected_tensors, weights_file)

    actual_tensors = dict(
        runai_safetensors_weights_iterator([str(weights_file)], False)
    )

    assert actual_tensors.keys() == expected_tensors.keys()
    assert actual_tensors["first"].data_ptr() != actual_tensors["second"].data_ptr()
    for name, expected_tensor in expected_tensors.items():
        assert torch.equal(actual_tensors[name], expected_tensor)


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
