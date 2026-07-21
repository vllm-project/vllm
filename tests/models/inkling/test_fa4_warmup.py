# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.inkling.nvidia.ops import fa4_warmup
from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
    bucket_max_seqlen_q,
    inkling_fa4_num_splits,
)
from vllm.models.inkling.nvidia.ops.fa4_warmup import (
    InklingFA4RelAttentionKernel,
    _num_warps_bucket,
)


def _vllm_config_from_reference_config(config: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_attention_heads=config.num_heads,
                num_key_value_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                rel_extent=config.rel_extent,
                swa_num_attention_heads=config.num_heads,
                swa_num_key_value_heads=config.num_kv_heads,
                swa_head_dim=config.head_dim,
                sliding_window_size=config.rel_extent,
            ),
            max_model_len=config.max_kv_len,
            dtype=config.dtype,
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            block_size=config.block_size,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=config.max_num_reqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
        ),
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


def test_warmup_enumerates_every_runtime_compile_class(monkeypatch):
    monkeypatch.setattr(
        fa4_warmup,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    config = SimpleNamespace(
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
    vllm_config = _vllm_config_from_reference_config(config)
    kernel = InklingFA4RelAttentionKernel()
    warmed_keys = {
        (
            key.max_seqlen_q,
            key.num_splits,
            key.num_warps_bucket,
            key.large_num_reqs,
        )
        for key in kernel.get_warmup_keys(vllm_config)
        if key.is_local == config.is_local
    }

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
