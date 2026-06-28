# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local copies of small platform-agnostic metadata helpers.

The hw_agnostic isolation lint forbids imports from
``vllm.v1.attention.backends.*``. The helpers below are duplicated from
upstream so the hw_agnostic builders don't need to reach across that
boundary. Keep this file in sync with upstream when its public
signature changes; it is small and stable on purpose.
"""

import torch

from vllm.model_executor.hw_agnostic.v1.attention.backend import (
    CommonAttentionMetadata,
)


def split_decodes_and_prefills(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
    require_uniform: bool = False,
    treat_short_extends_as_decodes: bool = True,
) -> tuple[int, int, int, int]:
    """Find the boundary between prefill and decode requests in a reordered
    batch.

    Mirrors ``vllm.v1.attention.backends.utils.split_decodes_and_prefills``
    so the hw_agnostic builders can stay independent of
    ``vllm/v1/attention/backends/utils.py``. See that module for the
    canonical docstring.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if (
        max_query_len <= decode_threshold
        and (not require_uniform or decode_threshold <= 1)
        and treat_short_extends_as_decodes
    ):
        return num_reqs, 0, num_tokens, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    if query_lens[0].item() > decode_threshold:
        # First request is not decode, so no decode requests.
        return 0, num_reqs, 0, num_tokens

    if require_uniform:
        # Padded-uniform batches (some query_lens == 0) are still all-decode;
        # this lets full-CGs treat padding rows as decodes so num_decodes
        # matches the captured size.
        if torch.all((query_lens == query_lens[0]) | (query_lens == 0)):
            return num_reqs, 0, num_tokens, 0
        is_prefill = query_lens != query_lens[0]
    else:
        is_prefill = query_lens > decode_threshold

    if not treat_short_extends_as_decodes:
        assert common_attn_metadata.is_prefilling is not None
        is_prefill |= common_attn_metadata.is_prefilling

    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens
