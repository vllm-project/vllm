# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ token split and exchanges (vllm/v1/attention/ops/shardq.py)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.ops import shardq
from vllm.v1.attention.ops.shardq import (
    ShardQMetadata,
    allocate_shardq_buffers,
    build_shardq_split,
    can_use_sequence_parallel_shardq,
    sequence_parallel_split,
    shard_q,
    unshard_out,
    zigzag_split,
)


@pytest.mark.parametrize("num_tokens", [1, 7, 8, 9, 1025, 8192])
@pytest.mark.parametrize("world", [2, 4, 8])
def test_sp_split_matches_padded_contiguous_input(num_tokens, world):
    """No query may change owners between the SP projection and attention."""
    split = sequence_parallel_split(num_tokens, world)
    chunk = -(-num_tokens // world)
    assert split.num_decode_tokens == 0
    assert split.sequence_parallel
    shards = np.split(split.token_ids, np.cumsum(split.num_tokens_per_rank)[:-1])
    for rank, ids in enumerate(shards):
        np.testing.assert_array_equal(
            ids, np.arange(num_tokens)[rank * chunk : (rank + 1) * chunk]
        )


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize(
    "num_tokens,input_tokens,mode,prefill_limit,capture_limit,expected",
    [
        (2000, 3000, CUDAGraphMode.NONE, None, None, False),
        (3000, 3000, CUDAGraphMode.NONE, None, None, True),
        (2001, 2002, CUDAGraphMode.NONE, None, None, True),
        (2001, 2003, CUDAGraphMode.NONE, None, None, False),
        (2048, 2048, CUDAGraphMode.PIECEWISE, 2048, 1024, False),
        (2049, 2049, CUDAGraphMode.PIECEWISE, 2048, 4096, True),
        (2048, 2048, CUDAGraphMode.PIECEWISE, None, 2048, False),
        (2049, 2049, CUDAGraphMode.PIECEWISE, None, 2048, True),
        (2048, 2048, CUDAGraphMode.NONE, 2048, 2048, True),
    ],
)
def test_sp_shardq_requires_matching_projection_layout(
    sequence_parallel,
    num_tokens,
    input_tokens,
    mode,
    prefill_limit,
    capture_limit,
    expected,
):
    """Graph capture and DP padding must preserve the local query layout."""
    metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[num_tokens], query_lens=[num_tokens]),
        block_size=16,
        device=torch.device("cpu"),
    )
    metadata.slot_mapping = torch.nn.functional.pad(
        metadata.slot_mapping, (0, input_tokens - num_tokens), value=-1
    )
    compilation = CompilationConfig(
        cudagraph_mode=mode,
        max_prefill_cudagraph_capture_tokens=prefill_limit,
        max_cudagraph_capture_size=capture_limit,
    )
    assert can_use_sequence_parallel_shardq(
        sequence_parallel, metadata, compilation, 2
    ) == (sequence_parallel and expected)


def _sp_projection_round_trip(rank, world, port, model_path):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_shard

    torch.accelerator.set_device_index(rank)
    config = VllmConfig(model_config=ModelConfig(model=model_path))
    config.attention_config.sparse_mla_shard_q_min_tokens = 1
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        ensure_model_parallel_initialized(world, 1)
        try:
            with torch.device("cuda"):
                full_proj = ColumnParallelLinear(
                    8, 32, bias=False, disable_tp=True, params_dtype=torch.float32
                )
                head_proj = ColumnParallelLinear(
                    8, 32, bias=False, params_dtype=torch.float32
                )
            # Dyadic inputs make the GEMM exact despite its different M and N.
            weight = torch.arange(256, device="cuda").reshape(32, 8).float() / 256
            full_proj.weight.weight_loader(full_proj.weight, weight)
            head_proj.weight.weight_loader(head_proj.weight, weight)
            for tokens in (8, 11, 1025):
                hidden = torch.arange(tokens * 8, device="cuda").reshape(tokens, 8)
                hidden = hidden.remainder(16).float() / 16
                local = sp_shard(hidden)
                torch.testing.assert_close(
                    sp_all_gather(local)[:tokens], hidden, rtol=0, atol=0
                )
                split = sequence_parallel_split(tokens, world)
                metadata = ShardQMetadata(
                    num_decode_tokens=0,
                    num_tokens_per_rank=split.num_tokens_per_rank.tolist(),
                    token_ids=torch.from_numpy(split.token_ids).cuda(),
                    rank_offsets=torch.from_numpy(
                        np.r_[0, np.cumsum(split.num_tokens_per_rank)]
                    ).cuda(),
                    sequence_parallel=True,
                )
                local_q = full_proj(local)[0].view(-1, 8, 4)
                local_q = local_q[: split.num_tokens_per_rank[rank]]
                restored = unshard_out(local_q, metadata, None)
                expected = head_proj(hidden)[0].view(tokens, 8 // world, 4)
                torch.testing.assert_close(restored, expected, rtol=0, atol=0)

                # Cache writes consume all rows on every rank, including the
                # uneven last shard. Keep packed slice strides and nontrivial
                # slots to catch accidental local-only writes or row swaps.
                cache_inputs = torch.arange(tokens * 704, device="cuda")
                cache_inputs = (
                    cache_inputs.reshape(tokens, 704).remainder(127).bfloat16() / 128
                )
                gathered = sp_all_gather(sp_shard(cache_inputs))[:tokens]
                slots = torch.arange(tokens, device="cuda", dtype=torch.int64).flip(0)
                blocks = -(-tokens // 64)
                scale = torch.ones(1, device="cuda", dtype=torch.float32)
                for dtype, width in (("auto", 576), ("fp8_ds_mla", 656)):
                    storage_dtype = torch.bfloat16 if dtype == "auto" else torch.uint8
                    actual_cache = torch.zeros(
                        blocks, 64, width, device="cuda", dtype=storage_dtype
                    )
                    expected_cache = torch.zeros_like(actual_cache)
                    for inputs, cache in (
                        (gathered, actual_cache),
                        (cache_inputs, expected_cache),
                    ):
                        ops.concat_and_cache_mla(
                            inputs[:, :512],
                            inputs[:, 512:576],
                            cache,
                            slots,
                            dtype,
                            scale,
                        )
                    torch.testing.assert_close(
                        actual_cache, expected_cache, rtol=0, atol=0
                    )
                actual_index_cache = torch.zeros(
                    blocks, 64, 132, device="cuda", dtype=torch.uint8
                )
                expected_index_cache = torch.zeros_like(actual_index_cache)
                for inputs, cache in (
                    (gathered, actual_index_cache),
                    (cache_inputs, expected_index_cache),
                ):
                    ops.indexer_k_quant_and_cache(
                        inputs[:, 576:].contiguous(), cache, slots, 128, "ue8m0"
                    )
                torch.testing.assert_close(
                    actual_index_cache, expected_index_cache, rtol=0, atol=0
                )

            # DP ranks with 5000/6000 tokens split into 3000-row microbatches.
            # The last microbatch has 2000/3000 real tokens, respectively.
            # Also retain SP when extra padding leaves TP boundaries unchanged.
            for tokens, padded_tokens in ((2000, 3000), (3000, 3000), (2001, 2002)):
                common = create_common_attn_metadata(
                    BatchSpec(seq_lens=[tokens + 128], query_lens=[tokens]),
                    block_size=16,
                    device=torch.device("cpu"),
                )
                common.slot_mapping = torch.nn.functional.pad(
                    common.slot_mapping, (0, padded_tokens - tokens), value=-1
                )
                split = build_shardq_split(common, config, sequence_parallel=True)
                assert split is not None
                metadata = ShardQMetadata(
                    num_decode_tokens=split.num_decode_tokens,
                    num_tokens_per_rank=split.num_tokens_per_rank.tolist(),
                    token_ids=torch.from_numpy(split.token_ids).cuda(),
                    rank_offsets=torch.from_numpy(
                        np.r_[0, np.cumsum(split.num_tokens_per_rank)]
                    ).cuda(),
                    sequence_parallel=split.sequence_parallel,
                )
                # Unique, exactly representable rows expose lost or mislabeled Q.
                hidden = torch.arange(padded_tokens, device="cuda").float()
                hidden[tokens:] = -1
                hidden = hidden[:, None].expand(-1, 8).contiguous()
                local = sp_shard(hidden)
                if split.sequence_parallel:
                    q = full_proj(local)[0].view(-1, 8, 4)
                    q = q[: split.num_tokens_per_rank[rank]]
                else:
                    gathered = sp_all_gather(local)[:tokens]
                    q = head_proj(gathered)[0].view(tokens, 8 // world, 4)
                    _, q = shard_q(q, metadata)
                restored = unshard_out(q, metadata, None)
                expected = full_proj(hidden[:tokens])[0].view(tokens, 8, 4)
                torch.testing.assert_close(
                    q, expected[metadata.local_token_ids], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    restored,
                    head_proj(hidden[:tokens])[0].view(tokens, 8 // world, 4),
                    rtol=0,
                    atol=0,
                )
                assert split.sequence_parallel == (tokens != 2000)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def test_sp_replicated_query_projection_returns_original_head_shards(tmp_path):
    """Restore Q head shards through SP and the DP-padding fallback."""
    from transformers import DeepseekV2Config

    DeepseekV2Config(
        architectures=["DeepseekV2ForCausalLM"], index_topk=128
    ).save_pretrained(tmp_path)
    mp.spawn(
        _sp_projection_round_trip, args=(2, get_open_port(), str(tmp_path)), nprocs=2
    )


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("lens", [[4096], [65, 300, 7000], [1000, 64, 64, 129]])
def test_zigzag_split_partitions_tokens(world, lens):
    qsl = np.r_[0, np.cumsum(lens)]
    token_ids, num_tokens_per_rank = zigzag_split(np.array(lens), world)
    per_rank = np.split(token_ids, np.cumsum(num_tokens_per_rank)[:-1])
    merged = np.sort(np.concatenate(per_rank))
    np.testing.assert_array_equal(merged, np.arange(qsl[-1]))
    for idx in per_rank:
        assert np.all(np.diff(idx) > 0)
    # Each rank takes one early and one late chunk of every request, so its
    # summed causal positions stay within a couple of tokens per request of
    # the mean.
    positions = [
        sum(int((idx[(idx >= a) & (idx < b)] - a).sum()) for a, b in zip(qsl, qsl[1:]))
        for idx in per_rank
    ]
    assert max(positions) - min(positions) <= sum(4 * n for n in lens)


@pytest.mark.parametrize("num_spec", [0, 2])
def test_build_shardq_split_after_indexer_decodes(monkeypatch, num_spec):
    """The MLA and indexer builders derive the same split from the batch:
    requests of at most 1 + num_spec tokens stay head-sharded, and the tokens
    after them are split once they reach sparse_mla_shard_q_min_tokens and a
    request is longer than the top-k."""
    monkeypatch.setattr(
        shardq, "get_tp_group", lambda: SimpleNamespace(world_size=4, rank_in_group=1)
    )
    query_lens = [1 + num_spec] * 3 + [100, 5000, 4000]
    metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[q + 300 for q in query_lens], query_lens=query_lens),
        block_size=16,
        device=torch.device("cpu"),
    )
    attention_config = SimpleNamespace(sparse_mla_shard_q_min_tokens=9100)
    hf_config = SimpleNamespace(index_topk=5299)
    vllm_config = SimpleNamespace(
        compilation_config=CompilationConfig(),
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=num_spec) if num_spec else None
        ),
        attention_config=attention_config,
        model_config=SimpleNamespace(hf_text_config=hf_config),
    )
    split = build_shardq_split(metadata, vllm_config)
    assert split is not None
    assert split.num_decode_tokens == 3 * (1 + num_spec)
    assert split.num_tokens_per_rank.sum() == 9100
    hf_config.index_topk = 5300
    assert build_shardq_split(metadata, vllm_config) is None
    hf_config.index_topk = 5299
    attention_config.sparse_mla_shard_q_min_tokens = 9101
    assert build_shardq_split(metadata, vllm_config) is None
    # Five tokens split into eight chunks, three of them empty, leaving rank 2
    # without tokens.
    attention_config.sparse_mla_shard_q_min_tokens = 1
    short = create_common_attn_metadata(
        BatchSpec(seq_lens=[5300], query_lens=[5]),
        block_size=16,
        device=torch.device("cpu"),
    )
    assert build_shardq_split(short, vllm_config) is None


def _round_trip(rank, world, port, use_sm, q_dtype, num_decode_tokens):
    torch.accelerator.set_device_index(rank)
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
        )
        ensure_model_parallel_initialized(world, 1)
        torch.manual_seed(0)
        lens = [700, 64, 3001]
        heads, dim = 4, 576
        num_tokens = num_decode_tokens + sum(lens)
        vllm_config.scheduler_config.max_num_batched_tokens = num_tokens
        if use_sm:
            allocate_shardq_buffers(
                vllm_config, heads, dim, 512, q_dtype, torch.bfloat16
            )
        all_heads = world * heads
        full = torch.randn(num_tokens, all_heads, dim, device="cuda").to(q_dtype)
        token_ids, num_tokens_per_rank = zigzag_split(np.array(lens), world)
        shardq = ShardQMetadata(
            num_decode_tokens=num_decode_tokens,
            num_tokens_per_rank=num_tokens_per_rank.tolist(),
            token_ids=torch.from_numpy(token_ids).cuda(),
            rank_offsets=torch.from_numpy(
                np.r_[0, np.cumsum(num_tokens_per_rank)]
            ).cuda(),
        )
        local_heads = full[:, rank * heads : (rank + 1) * heads].contiguous()
        for _ in range(2):  # The second pass reuses the symmetric buffers.
            decode_q, sharded = shard_q(local_heads, shardq)
            # Bitwise, as the exchanges only move bytes (and fp8 lacks
            # torch.equal).
            assert torch.equal(
                decode_q.view(torch.uint8),
                local_heads[:num_decode_tokens].view(torch.uint8),
            )
            expected = full[num_decode_tokens:][shardq.local_token_ids]
            assert torch.equal(sharded.view(torch.uint8), expected.view(torch.uint8))
            # Attention outputs are a head-padded slice; unshard reads its
            # strides.
            padded = torch.zeros(sharded.shape[0], all_heads + 8, 512, device="cuda")
            padded[:, :all_heads] = sharded[..., :512].float()
            out = padded.bfloat16()[:, :all_heads]
            restored = unshard_out(out, shardq, decode_q[..., :512].bfloat16())
            assert torch.equal(restored, local_heads[..., :512].bfloat16())
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("use_sm", [False, True])
@pytest.mark.parametrize("num_decode_tokens", [0, 33])
def test_shard_q_round_trip(use_sm, q_dtype, num_decode_tokens):
    """The exchanges move each rank's tokens to it with all heads and back,
    through symmetric memory or NCCL, leaving the decode tokens head-sharded."""
    world = 2
    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    mp.spawn(
        _round_trip,
        args=(world, get_open_port(), use_sm, q_dtype, num_decode_tokens),
        nprocs=world,
    )
