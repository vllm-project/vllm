# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ token split and exchanges (vllm/v1/attention/ops/shardq.py)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import (
    CUDAGraphMode,
    HiSparseConfig,
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
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.ops import shardq
from vllm.v1.attention.ops.shardq import (
    ShardQMetadata,
    allocate_shardq_buffers,
    build_shardq_split,
    shard_q,
    unshard_out,
    zigzag_split,
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
    attention_config = SimpleNamespace(
        sparse_mla_shard_q_min_tokens=9100, hisparse_config=None
    )
    hf_config = SimpleNamespace(index_topk=5299)
    vllm_config = SimpleNamespace(
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


@pytest.mark.parametrize("query_lens", [[4, 2, 4096], [4, 4, 4096]])
def test_build_shardq_split_skips_mixed_spec_verify_under_hisparse(
    monkeypatch, query_lens
):
    """HiSparse forces the sparse-MLA builder to treat only 1-token requests as
    decodes with uniform lengths, so speculative-verification tokens in a mixed
    batch are prefills there. ShardQ's 1 + num_spec boundary would count them as
    decodes and disagree with both the indexer's token-sharded top-k layout and
    the separate prefill path, so it must be skipped for that step."""
    monkeypatch.setattr(
        shardq, "get_tp_group", lambda: SimpleNamespace(world_size=4, rank_in_group=1)
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        attention_config=SimpleNamespace(
            hisparse_config=object(), sparse_mla_shard_q_min_tokens=1024
        ),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048)),
    )
    metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[q + 300 for q in query_lens], query_lens=query_lens),
        block_size=16,
        device=torch.device("cpu"),
    )
    assert build_shardq_split(metadata, vllm_config) is None

    # A regular decode step batched with a long prefill has the same boundary in
    # both derivations and still shards.
    regular = create_common_attn_metadata(
        BatchSpec(seq_lens=[301, 301, 4396], query_lens=[1, 1, 4096]),
        block_size=16,
        device=torch.device("cpu"),
    )
    split = build_shardq_split(regular, vllm_config)
    assert split is not None
    assert split.num_decode_tokens == 2
    assert split.num_tokens_per_rank.sum() == 4096


def _round_trip(rank, world, port, use_sm, q_dtype, num_decode_tokens):
    torch.accelerator.set_device_index(rank)
    device = torch.device(current_platform.device_type, rank)
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend=current_platform.dist_backend,
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
        full = torch.randn(num_tokens, all_heads, dim, device=device).to(q_dtype)
        token_ids, num_tokens_per_rank = zigzag_split(np.array(lens), world)
        shardq = ShardQMetadata(
            num_decode_tokens=num_decode_tokens,
            num_tokens_per_rank=num_tokens_per_rank.tolist(),
            token_ids=torch.from_numpy(token_ids).to(device),
            rank_offsets=torch.from_numpy(np.r_[0, np.cumsum(num_tokens_per_rank)]).to(
                device
            ),
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
            padded = torch.zeros(sharded.shape[0], all_heads + 8, 512, device=device)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_indexer_build_survives_next_build(monkeypatch):
    """Building the next step must not change a step's still-queued uploads."""
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    device = torch.device("cuda")
    config = create_vllm_config(max_model_len=8192)
    config.model_config.hf_text_config.index_topk = 2048
    config.compilation_config.static_forward_context = {
        "indexer": SimpleNamespace(use_shardq=True)
    }
    monkeypatch.setattr(
        shardq, "get_tp_group", lambda: SimpleNamespace(world_size=2, rank_in_group=0)
    )
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=MLAAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
        ),
        layer_names=["indexer"],
        vllm_config=config,
        device=device,
        block_table_width=128,
    )
    # On the device up front: a blocking copy would drain the queue below.
    steps = [
        create_common_attn_metadata(
            BatchSpec(seq_lens=[8192, 4096], query_lens=query_lens), 64, device
        )
        for query_lens in ([3000, 1200], [700, 900])
    ]

    def build_and_read(step):
        """What the step's forward reads, queued right behind its build."""
        prefill = builder.build(0, step).prefill
        assert prefill is not None
        indices = prefill.shardq_gather_indices
        assert indices is not None
        chunk = prefill.chunks[0]
        return [t.clone() for t in (indices, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke)]

    expected = [t.cpu() for t in build_and_read(steps[0])]
    torch.cuda._sleep(1 << 30)  # keep the uploads queued
    queued = build_and_read(steps[0])
    builder.build(0, steps[1])
    torch.accelerator.synchronize()
    for got, want in zip(queued, expected):
        torch.testing.assert_close(got.cpu(), want)


def _check_builder_shardq_route(rank, world, port, hisparse=False):
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
    from vllm.v1.attention.backends.mla.sparse_utils import SparseMLAPrefillRoute
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V3.2",
        tensor_parallel_size=world,
        max_model_len=4096,
        max_num_batched_tokens=2112,
        max_num_seqs=2,
        num_gpu_blocks=128,
        block_size=64,
        dtype="bfloat16",
    )
    config.cache_config.cache_dtype = "fp8_ds_mla"
    if hisparse:
        config.attention_config.hisparse_config = HiSparseConfig(
            device_buffer_size=4096,
        )
    config.model_config = ModelConfig(
        model="deepseek-ai/DeepSeek-V3.2",
        hf_overrides={"num_attention_heads": 32},
        dtype="bfloat16",
        max_model_len=4096,
    )
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend=current_platform.dist_backend,
        )
        ensure_model_parallel_initialized(world, 1)
        init_workspace_manager(device)
        try:
            # The indexer shards the long 64-query request, while MLA classifies
            # it as decode. MLA's remaining prefill fits within the top-k.
            common = create_common_attn_metadata(
                BatchSpec(seq_lens=[4096, 2048], query_lens=[64, 2048]),
                64,
                device,
                arange_block_indices=True,
            )
            for enabled in (False, True):
                prefix = f"layer_{enabled}"
                with torch.device(device):
                    layer = MLAAttention(
                        num_heads=32 // world,
                        scale=192**-0.5,
                        qk_nope_head_dim=128,
                        qk_rope_head_dim=64,
                        v_head_dim=128,
                        q_lora_rank=1536,
                        kv_lora_rank=512,
                        kv_b_proj=ColumnParallelLinear(512, 32 * 256, bias=False),
                        cache_config=config.cache_config,
                        prefix=prefix,
                        attn_backend=FlashMLASparseBackend,
                        use_sparse=True,
                        topk_indices_buffer=torch.empty(
                            (2112, 2048), dtype=torch.int32
                        ),
                        shardq_eligible=enabled,
                    )
                builder = FlashMLASparseBackend.get_builder_cls()(
                    layer.get_kv_cache_config(config).spec, [prefix], config, device
                )
                assert layer.use_shardq == enabled
                metadata = builder.build(0, common)
                assert (metadata.shardq is not None) == enabled
                assert metadata.prefill is not None
                expected = (
                    SparseMLAPrefillRoute.SPARSE_MQA
                    if enabled or hisparse
                    else SparseMLAPrefillRoute.DENSE_MHA
                )
                assert metadata.prefill.route is expected, metadata.prefill.route
        finally:
            reset_workspace_manager()
            destroy_model_parallel()
            destroy_distributed_environment()


def test_builder_selects_mqa_after_shardq_split():
    """The real builder must select the route after computing the ShardQ split."""
    world = 2
    if not current_platform.is_cuda() or torch.accelerator.device_count() < world:
        pytest.skip("requires two CUDA GPUs for FlashMLA ShardQ")
    from vllm.v1.attention.ops import flashmla

    supported, reason = flashmla.is_flashmla_sparse_supported()
    if not supported:
        pytest.skip(reason)
    mp.spawn(_check_builder_shardq_route, args=(world, get_open_port()), nprocs=world)


def test_builder_selects_shardq_under_hisparse():
    """HiSparse + quantized cache + TP>1 must keep the ShardQ route enabled."""
    world = 2
    if not current_platform.is_cuda() or torch.accelerator.device_count() < world:
        pytest.skip("requires two CUDA GPUs for FlashMLA ShardQ")
    from vllm.v1.attention.ops import flashmla

    supported, reason = flashmla.is_flashmla_sparse_supported()
    if not supported:
        pytest.skip(reason)
    mp.spawn(
        _check_builder_shardq_route,
        args=(world, get_open_port(), True),
        nprocs=world,
    )
