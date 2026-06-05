# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.ubatch_utils import UBatchSlice, split_attn_metadata


def test_split_attn_metadata_preserves_position_and_prefill_flags():
    query_start_loc = torch.tensor([0, 3, 7, 9], dtype=torch.int32)
    metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=torch.tensor([16, 24, 32], dtype=torch.int32),
        num_reqs=3,
        num_actual_tokens=9,
        max_query_len=4,
        max_seq_len=32,
        block_table_tensor=torch.arange(12, dtype=torch.int32).reshape(3, 4),
        slot_mapping=torch.arange(9, dtype=torch.int64),
        positions=torch.arange(100, 109, dtype=torch.int64),
        is_prefilling=torch.tensor([False, True, True]),
        seq_lens_cpu_upper_bound=torch.tensor([16, 24, 32], dtype=torch.int32),
    )

    split = split_attn_metadata(
        [UBatchSlice(request_slice=slice(1, 3), token_slice=slice(3, 9))],
        metadata,
    )[0]

    torch.testing.assert_close(split.positions, metadata.positions[3:9])
    torch.testing.assert_close(split.is_prefilling, metadata.is_prefilling[1:3])
