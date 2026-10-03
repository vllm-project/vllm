# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob

import pytest
import torch
from safetensors import safe_open

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    safetensors_weights_iterator,
    st_prefetch_safetensors_weights_iterator,
    st_prefetch_sharded_weights_iterator,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not hasattr(safe_open, "prefetch"),
    reason="safetensors' prefetch loader needs CUDA and safetensors >= 0.9.0rc1",
)


def _gpt2_shards() -> list[str]:
    model_dir = download_weights_from_hf(
        "openai-community/gpt2", cache_dir=None, allow_patterns=["*.safetensors"]
    )
    shards = glob.glob(f"{model_dir}/*.safetensors")
    assert len(shards) > 0
    return shards


def _assert_same_tensors(iterator, shards: list[str]) -> None:
    got = {name: tensor for name, tensor in iterator(shards, True)}
    for name, tensor in got.items():
        assert tensor.device.type == "cuda", name
    expected = dict(safetensors_weights_iterator(shards, True))
    assert got.keys() == expected.keys()
    for name, tensor in expected.items():
        assert got[name].dtype == tensor.dtype, name
        assert got[name].shape == tensor.shape, name
        assert torch.equal(got[name].cpu(), tensor), name


def test_st_prefetch_weights_iterator():
    _assert_same_tensors(st_prefetch_safetensors_weights_iterator, _gpt2_shards())


def test_st_prefetch_sharded_falls_back_without_tp_group():
    # no parallel state initialised: the sharded reader degrades to the plain one
    _assert_same_tensors(st_prefetch_sharded_weights_iterator, _gpt2_shards())
