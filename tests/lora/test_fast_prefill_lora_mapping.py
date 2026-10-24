# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU


@pytest.mark.skip_global_cleanup
def test_fast_prefill_lora_mapping_rebuilds_compact_metadata():
    wrapper = PunicaWrapperGPU(
        max_num_batched_tokens=16,
        max_batches=2,
        device="cpu",
        lora_config=SimpleNamespace(max_loras=2, specialize_active_lora=False),
    )
    wrapper._token_lora_indices[:6] = torch.tensor([-1, -1, -1, -1, 0, 0])
    wrapper.indices_len[0] = 6

    compact_mapping = wrapper.prepare_fast_prefill_token_mapping(
        torch.tensor([3, 5], dtype=torch.int32)
    )
    assert compact_mapping is not None
    assert compact_mapping.tolist() == [-1, 0]

    (
        token_lora_mapping,
        sorted_token_indices,
        num_tokens_per_lora,
        lora_token_start_loc,
        active_lora_ids,
        no_lora_flag,
        _,
    ) = wrapper.fast_prefill_mapping_meta.meta_args(2, False)
    assert token_lora_mapping.tolist() == [-1, 0]
    assert sorted_token_indices.tolist() == [0, 1]
    assert num_tokens_per_lora[:3].tolist() == [1, 1, 0]
    assert lora_token_start_loc[:4].tolist() == [0, 1, 2, 0]
    assert active_lora_ids[:2].tolist() == [-1, 0]
    assert not no_lora_flag.item()
