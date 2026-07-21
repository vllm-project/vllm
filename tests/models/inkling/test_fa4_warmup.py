# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
    bucket_max_seqlen_q,
)
from vllm.models.inkling.nvidia.ops.fa4_warmup import (
    InklingFA4RelAttentionKernel,
)


def _make_kernel() -> InklingFA4RelAttentionKernel:
    return InklingFA4RelAttentionKernel(
        num_heads=16,
        num_kv_heads=2,
        head_dim=128,
        rel_extent=1024,
        window_size=(-1, -1),
        is_local=False,
        max_kv_len=65536,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        block_size=16,
        max_num_reqs=64,
        max_num_batched_tokens=192,
    )


def test_bucket_max_seqlen_q():
    assert [bucket_max_seqlen_q(n) for n in range(1, 10)] == [
        1,
        2,
        4,
        4,
        8,
        8,
        8,
        8,
        16,
    ]


def test_warmup_enumerates_every_runtime_compile_class():
    kernel = _make_kernel()
    warmed_keys = set(kernel.get_warmup_keys())

    for query_len in range(1, kernel.max_num_batched_tokens + 1):
        max_num_reqs = min(
            kernel.max_num_reqs,
            kernel.max_num_batched_tokens - query_len + 1,
        )
        for num_reqs in range(1, max_num_reqs + 1):
            runtime_key = kernel.dispatch(query_len=query_len, num_reqs=num_reqs)
            assert runtime_key in warmed_keys
