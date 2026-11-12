# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import tempfile
from datetime import timedelta

import huggingface_hub.constants
import pytest
import torch

from vllm.config import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
@pytest.mark.parametrize("queue_size", [0, 1])
@pytest.mark.parametrize("nogds", [None, True, False])
def test_fastsafetensors_model_loader(monkeypatch, queue_size, nogds):
    """The real loader must preserve every tensor at both pipeline depths."""
    monkeypatch.setenv("VLLM_FASTSAFETENSORS_QUEUE_SIZE", str(queue_size))
    with tempfile.TemporaryDirectory() as tmpdir:
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "openai-community/gpt2", allow_patterns=["*.safetensors"], cache_dir=tmpdir
        )
        safetensors = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
        assert len(safetensors) > 0

        fastsafetensors_tensors = {}
        hf_safetensors_tensors = {}

        for name, tensor in fastsafetensors_weights_iterator(
            safetensors, True, nogds_override=nogds
        ):
            fastsafetensors_tensors[name] = tensor

        for name, tensor in safetensors_weights_iterator(safetensors, True):
            hf_safetensors_tensors[name] = tensor

        assert len(fastsafetensors_tensors) == len(hf_safetensors_tensors)

        for name, fastsafetensors_tensor in fastsafetensors_tensors.items():
            fastsafetensors_tensor = fastsafetensors_tensor.to("cpu")
            assert fastsafetensors_tensor.dtype == hf_safetensors_tensors[name].dtype
            assert fastsafetensors_tensor.shape == hf_safetensors_tensors[name].shape
            assert torch.all(fastsafetensors_tensor.eq(hf_safetensors_tensors[name]))


@pytest.mark.parametrize("nogds", [None, 0, 1, "true", [], {}])
def test_nogds_config_rejects_non_boolean(nogds):
    with pytest.raises(ValueError, match="must be a bool"):
        DefaultModelLoader(
            LoadConfig(
                load_format="fastsafetensors",
                model_loader_extra_config={"fastsafetensors_nogds": nogds},
            )
        )


@pytest.mark.parametrize("load_format", ["auto", "pt", "safetensors"])
def test_nogds_config_rejects_other_load_formats(load_format):
    with pytest.raises(ValueError, match="requires fastsafetensors"):
        DefaultModelLoader(
            LoadConfig(
                load_format=load_format,
                model_loader_extra_config={"fastsafetensors_nogds": True},
            )
        )


def _reject_multirank_gds(rank, rendezvous):
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        config = LoadConfig(
            load_format="fastsafetensors",
            model_loader_extra_config={"fastsafetensors_nogds": False},
        )
        loader = DefaultModelLoader(config)
        assert loader.load_config.model_loader_extra_config == {
            "fastsafetensors_nogds": False
        }
        assert torch.distributed.get_world_size() == 2
        with pytest.raises(
            ValueError,
            match="fastsafetensors_nogds=False requires a single-rank process group",
        ):
            next(fastsafetensors_weights_iterator([], False, nogds_override=False))
    finally:
        torch.distributed.destroy_process_group()


def test_fastsafetensors_rejects_forced_gds_with_real_multirank_group(tmp_path):
    """Reject the unsafe selection using actual Gloo ranks, not a group stub."""
    torch.multiprocessing.spawn(
        _reject_multirank_gds,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
