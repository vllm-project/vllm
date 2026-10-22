# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.gpu.spec_decode.longspark.state import (
    merge_normalized_attention,
    normalized_gqa_attention,
    normalized_gqa_attention_batched,
)


def test_longspark_incremental_global_state_matches_full_prefix():
    """Accepted-delta updates must equal rescanning the committed prefix."""
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(8, 4, 16, generator=generator, dtype=torch.bfloat16)
    keys = torch.randn(2, 29, 16, generator=generator, dtype=torch.bfloat16)
    values = torch.randn(2, 29, 16, generator=generator, dtype=torch.bfloat16)

    full_output, full_lse = normalized_gqa_attention(query, keys, values)
    base_output, base_lse = normalized_gqa_attention(
        query, keys[:, :21], values[:, :21]
    )
    delta_output, delta_lse = normalized_gqa_attention(
        query, keys[:, 21:], values[:, 21:]
    )
    merged_output, merged_lse = merge_normalized_attention(
        base_output, base_lse, delta_output, delta_lse
    )

    torch.testing.assert_close(merged_output, full_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(merged_lse, full_lse, rtol=2e-5, atol=2e-5)


def test_longspark_global_state_preserves_grouped_query_head_order():
    """The GQA reshape must agree with explicit KV-head replication."""
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(8, 3, 16, generator=generator, dtype=torch.float32)
    keys = torch.randn(2, 11, 16, generator=generator, dtype=torch.float32)
    values = torch.randn(2, 11, 16, generator=generator, dtype=torch.float32)

    output, lse = normalized_gqa_attention(query, keys, values)
    repeated_keys = keys.repeat_interleave(4, dim=0)
    repeated_values = values.repeat_interleave(4, dim=0)
    scores = torch.einsum("hqd,hkd->hqk", query, repeated_keys) * (16**-0.5)
    reference_lse = scores.logsumexp(dim=-1)
    reference_output = torch.einsum(
        "hqk,hkd->hqd", scores.softmax(dim=-1), repeated_values
    )

    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(lse, reference_lse)


def test_longspark_batched_global_state_matches_individual_requests():
    generator = torch.Generator().manual_seed(29)
    query = torch.randn(8, 4, 16, generator=generator, dtype=torch.bfloat16)
    keys = torch.randn(3, 2, 11, 16, generator=generator, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    valid_mask = torch.arange(11)[None] < torch.tensor([11, 7, 3])[:, None]

    output, lse = normalized_gqa_attention_batched(query, keys, values, valid_mask)
    for request, length in enumerate((11, 7, 3)):
        expected_output, expected_lse = normalized_gqa_attention(
            query, keys[request, :, :length], values[request, :, :length]
        )
        torch.testing.assert_close(output[request], expected_output)
        torch.testing.assert_close(lse[request], expected_lse)
