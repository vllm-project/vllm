# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.v1.worker.replayssm_utils import (
    ReplaySSMBlockCopier,
    get_replayssm_block_copy_tensors,
)
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
def test_packed_replayssm_binding_and_block_copy(state_dtype, monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.worker.utils.async_tensor_h2d",
        lambda data, device: torch.as_tensor(data, device=device),
    )
    layer = MambaMixer2.__new__(MambaMixer2)
    torch.nn.Module.__init__(layer)
    layer.use_flashinfer_replayssm = True
    layer.use_replayssm = True
    shapes = ((2,), (2,), (1, 2, 2), (1, 2), (1, 2, 2))
    dtypes = (
        torch.bfloat16,
        state_dtype,
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
    )
    layer.get_state_shape = lambda: shapes
    layer.get_state_dtype = lambda: dtypes
    layer._replayssm_ring_start = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    layer._replayssm_prev_num_accepted = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    raw = torch.zeros((4, 1, 1, 64), dtype=torch.uint8)
    layer.bind_kv_cache(raw)
    assert len(layer.kv_cache) == 5
    assert len(layer.replayssm_cache) == 3
    for i, tensor in enumerate(layer.kv_cache):
        assert tensor.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        assert tensor.stride(0) * tensor.element_size() == 64
        tensor[0].fill_(i + 1)
    assert layer.replayssm_cache == layer.kv_cache[2:]

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_block_size=16,
            mamba_page_size_padded=64,
            mamba_cache_mode="align",
            use_replayssm=True,
        ),
        num_speculative_tokens=3,
    )
    spec = layer.get_kv_cache_spec(config)
    assert spec.shapes == shapes
    assert spec.page_size_bytes == 64
    assert spec.requires_live_state_copy
    assert spec.num_speculative_blocks == 0
    before = raw.clone()
    trackers = (layer._replayssm_ring_start, layer._replayssm_prev_num_accepted)
    expected_trackers = [t.clone() for t in trackers]
    for t in expected_trackers:
        t[1] = t[0]
    copy_kv_cache_blocks_inplace(
        [raw, *get_replayssm_block_copy_tensors({"layer": layer})], 4, [(0, 1)]
    )
    assert torch.equal(raw[1], before[0])
    assert torch.equal(raw[0], before[0])
    assert torch.equal(raw[2:], before[2:])
    for actual, expected in zip(trackers, expected_trackers):
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_replayssm_block_copier_excludes_ring_subsegments():
    num_blocks = 4
    raw = torch.zeros((num_blocks * 5, 1), device="cuda")
    canonical = tuple(
        raw[index * num_blocks : (index + 1) * num_blocks] for index in range(5)
    )
    layer = SimpleNamespace(
        use_flashinfer_replayssm=True,
        kv_cache=canonical,
        replayssm_cache=canonical[2:5],
        _replayssm_ring_start=torch.zeros(num_blocks, dtype=torch.int32, device="cuda"),
        _replayssm_prev_num_accepted=torch.zeros(
            num_blocks, dtype=torch.int32, device="cuda"
        ),
    )

    extras = get_replayssm_block_copy_tensors({"layer": layer})
    assert extras == [layer._replayssm_ring_start, layer._replayssm_prev_num_accepted]

    copier = ReplaySSMBlockCopier([*canonical, *extras], num_blocks)
    assert copier._meta is not None
    assert copier._meta[-1] == len(canonical) + len(extras)
