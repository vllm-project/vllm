# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative verify requests decode as causal single-query rows."""

from __future__ import annotations

import pytest
import torch

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.turboquant_attn import TurboQuantMetadata
from vllm.v1.attention.backends.ultraquant_attn import _split_multi_query_decode
from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.reference import reference_ultraquant_attention
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store
from vllm.v1.attention.ops.ultraquant.triton_unified_attention import (
    ultraquant_unified_attention,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)


def test_ragged_verify_batch_matches_causal_reference():
    device = torch.device("cuda")
    set_random_seed(0)

    query_lens = [3, 1, 4]
    B, Hk, Hq, N, D = len(query_lens), 1, 6, 128, 256
    block_size = 32
    blocks_per_seq = N // block_size
    scale = D**-0.5

    key = torch.randn(B, N, Hk, D, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    kv_cache = torch.zeros(
        B * blocks_per_seq,
        block_size,
        Hk,
        slot_size(D),
        dtype=torch.uint8,
        device=device,
    )
    ultraquant_store(
        key.reshape(-1, Hk, D),
        value.reshape(-1, Hk, D),
        kv_cache,
        torch.arange(B * N, dtype=torch.int64, device=device),
    )
    block_table = torch.arange(
        B * blocks_per_seq, dtype=torch.int32, device=device
    ).view(B, blocks_per_seq)

    # One zero-length padded request and two padded query rows, as the model
    # runner produces for a CUDA-graph batch.
    num_rows = sum(query_lens) + 2
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist(), sum(query_lens)],
        dtype=torch.int32,
        device=device,
    )
    metadata = _split_multi_query_decode(
        num_rows,
        TurboQuantMetadata(
            seq_lens=torch.tensor([N] * B + [0], dtype=torch.int32, device=device),
            slot_mapping=torch.empty(0, dtype=torch.int64, device=device),
            block_table=torch.cat([block_table, torch.zeros_like(block_table[:1])]),
            query_start_loc=query_start_loc,
            max_seq_len=N,
        ),
    )
    query = torch.randn(num_rows, Hq, D, dtype=torch.bfloat16, device=device)
    actual = ultraquant_unified_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=metadata.block_table,
        seq_lens=metadata.seq_lens,
        query_start_loc=metadata.query_start_loc,
        scale=scale,
        max_query_len=1,
        max_seq_len=N,
    )

    row = 0
    for req, q_len in enumerate(query_lens):
        for j in range(q_len):
            ctx = N - q_len + 1 + j
            expected = reference_ultraquant_attention(
                query=query[row : row + 1],
                key=key[req : req + 1, :ctx],
                value=value[req : req + 1, :ctx],
                scale=scale,
            )
            cos = torch.nn.functional.cosine_similarity(
                actual[row].float().reshape(-1),
                expected.float().reshape(-1),
                dim=0,
            ).item()
            assert cos > 0.98, (req, j, cos)
            row += 1
    assert torch.isfinite(actual[row:]).all()
