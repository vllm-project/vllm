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
def test_kv_cache_update_kernel():
    page_num = 1000
    page_size = 32
    combined_kv_head_num = 16
    head_dim = 128
    kernel_block_size = 16
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
    slice_lens = np.array([7, 32, 32, 1, 1, 1, 9], dtype=np.int32)
    kv_cache_start_indices = np.array([57, 64, 96, 104, 213, 345, 488],
                                      dtype=np.int32)
    new_kv_cache_indices = np.array([0, 7, 39, 71, 72, 73, 74], dtype=np.int32)
    slot_mapping = np.stack(
        [kv_cache_start_indices, new_kv_cache_indices, slice_lens], axis=1)
    slot_mapping = np.pad(
        slot_mapping, [[0, kernel_block_size - slot_mapping.shape[0]], [0, 0]],
        constant_values=0)
    slot_mapping_cpu = torch.tensor(slot_mapping, device="cpu")
    slot_mapping_xla = slot_mapping_cpu.to(torch_xla.device())
    torch_xla.sync()

    torch.ops.xla.dynamo_set_buffer_donor_(kv_cache_xla, True)
    new_kv_cache_xla = torch.ops.xla.kv_cache_update_op(
        new_kv_xla, slot_mapping_xla, kv_cache_xla, page_size,
        kernel_block_size)
    kv_cache_xla.copy_(new_kv_cache_xla)
    torch_xla.sync()

    for ni, ci, sl in zip(new_kv_cache_indices, kv_cache_start_indices,
                          slice_lens):
        kv_cache_cpu[ci:ci + sl, :, :] = new_kv_cpu[ni:ni + sl, :, :]

    assert torch.allclose(kv_cache_xla.cpu(),
                          kv_cache_cpu,
                          atol=1e-4,
                          rtol=1e-4)
