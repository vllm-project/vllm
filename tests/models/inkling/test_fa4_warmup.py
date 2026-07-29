# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
    bucket_max_seqlen_q,
    inkling_fa4_num_splits,
)
from vllm.models.inkling.nvidia.ops.fa4_warmup import (
    InklingFA4WarmupConfig,
    _iter_compile_units,
    _num_warps_bucket,
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
    config = InklingFA4WarmupConfig(
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
    warmed_keys = {unit.key[-1] for unit in _iter_compile_units(config)}

    for query_len in range(1, config.max_num_batched_tokens + 1):
        max_seqlen_q = bucket_max_seqlen_q(query_len)
        max_num_reqs = min(
            config.max_num_reqs,
            config.max_num_batched_tokens - query_len + 1,
        )
        for num_reqs in range(1, max_num_reqs + 1):
            num_splits = inkling_fa4_num_splits(
                is_local=config.is_local,
                batch_size=num_reqs,
                max_query_len=max_seqlen_q,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                max_kv_len=config.max_kv_len,
            )
            runtime_key = (
                max_seqlen_q,
                num_splits,
                _num_warps_bucket(num_reqs) if num_splits > 1 else None,
                num_reqs > 1024,
            )
            assert runtime_key in warmed_keys
