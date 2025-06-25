# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
import torch_xla

import vllm.v1.attention.backends.pallas  # noqa: F401
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This is a test for TPU only")
@pytest.mark.parametrize("page_size", [32, 33])
@pytest.mark.parametrize("combined_kv_head_num", [2, 16])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("num_slices_per_block", [4, 8])
def test_kv_cache_update_kernel(page_size: int, combined_kv_head_num: int,
                                head_dim: int, num_slices_per_block: int):
    page_num = 1000
    padded_num_tokens = 128
    kv_cache_cpu = torch.zeros(
        (page_num * page_size, combined_kv_head_num, head_dim),
        dtype=torch.bfloat16,
        device="cpu")
    kv_cache_xla = kv_cache_cpu.to(torch_xla.device())
    new_kv_cpu = torch.randn(
        (padded_num_tokens, combined_kv_head_num, head_dim),
        dtype=torch.bfloat16,
        device="cpu")
    new_kv_xla = new_kv_cpu.to(torch_xla.device())
    slice_lens = np.array([7, page_size, page_size, 1, 1, 1, 9],
                          dtype=np.int32)
    kv_cache_start_indices = np.array([
        page_size * 2 - 7, page_size * 2, page_size * 3, page_size * 4 + 6,
        page_size * 5 + 7, page_size * 6 + 8, page_size * 15 + 3
    ],
                                      dtype=np.int32)
    new_kv_cache_indices = np.concatenate(
        [np.array([0], dtype=np.int32),
         np.cumsum(slice_lens[:-1])])
    slot_mapping = np.stack(
        [kv_cache_start_indices, new_kv_cache_indices, slice_lens], axis=1)
    padded_size = (slot_mapping.shape[0] + num_slices_per_block -
                   1) // num_slices_per_block * num_slices_per_block
    slot_mapping = np.pad(slot_mapping,
                          [[0, padded_size - slot_mapping.shape[0]], [0, 0]],
                          constant_values=0)
    slot_mapping = np.transpose(slot_mapping)
    slot_mapping_cpu = torch.tensor(slot_mapping,
                                    device="cpu",
                                    dtype=torch.int32)
    slot_mapping_xla = slot_mapping_cpu.to(torch_xla.device())
    torch_xla.sync()

    torch.ops.xla.dynamo_set_buffer_donor_(kv_cache_xla, True)
    new_kv_cache_xla = torch.ops.xla.kv_cache_update_op(
        new_kv_xla, slot_mapping_xla, kv_cache_xla, page_size,
        num_slices_per_block)
    kv_cache_xla.copy_(new_kv_cache_xla)
    torch_xla.sync()

    for ni, ci, sl in zip(new_kv_cache_indices, kv_cache_start_indices,
                          slice_lens):
        kv_cache_cpu[ci:ci + sl, :, :] = new_kv_cpu[ni:ni + sl, :, :]

    assert torch.allclose(kv_cache_xla.cpu(),
                          kv_cache_cpu,
                          atol=1e-4,
                          rtol=1e-4)
