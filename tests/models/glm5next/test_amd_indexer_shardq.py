# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ pool-id all-gather restores token order across TP ranks."""

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

import vllm.models.glm5next.common.sparse_indexer as sparse_indexer
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.ops.shardq import zigzag_split

SELECT_K = 512
KPOOL = 4
PREFILL_LENS = [40, 70, 33, 128]


class _FakeGroup:
    """TP group whose all_gatherv returns the ranks' locals in rank order."""

    def __init__(self, rank: int, locals_: list[torch.Tensor]):
        self.rank_in_group = rank
        self.locals = locals_
        self.element_size = 0

    def all_gatherv(self, local, dim=0, sizes=None):
        assert dim == 0
        assert sizes == [len(t) for t in self.locals]
        self.element_size = local.element_size()
        # Narrowed ids arrive bit-cast to float16, where a -1 id reads as NaN.
        narrow = local.dtype == torch.float16
        sent = [t.to(torch.int16) for t in self.locals] if narrow else self.locals
        mine = local.view(torch.int16) if narrow else local
        assert torch.equal(mine, sent[self.rank_in_group])
        gathered = torch.cat(sent)
        return gathered.view(torch.float16) if narrow else gathered


def _split_by_rank(expected, token_ids, per_rank, world):
    offsets = np.concatenate(([0], np.cumsum(per_rank)))
    return [
        expected[token_ids[offsets[r] : offsets[r + 1]]].clone() for r in range(world)
    ]


@pytest.mark.parametrize("world", [2, 8])
def test_all_gather_restores_token_order(monkeypatch, world):
    token_ids, per_rank = zigzag_split(np.array(PREFILL_LENS), world)
    rows = sum(PREFILL_LENS)
    expected = torch.arange(rows * SELECT_K, dtype=torch.int32).view(rows, SELECT_K)
    locals_ = _split_by_rank(expected, token_ids, per_rank, world)
    ids = torch.from_numpy(token_ids.astype(np.int64))

    for rank in range(world):
        group = _FakeGroup(rank, locals_)
        monkeypatch.setattr(sparse_indexer, "get_tp_group", lambda g=group: g)
        got = sparse_indexer.all_gather_shardq_pool_ids(
            locals_[rank], per_rank.tolist(), ids, rows * SELECT_K
        )
        assert torch.equal(got, expected)
        assert group.element_size == 4


def test_narrow_exchange_keeps_ids_and_sentinel(monkeypatch):
    world = 4
    token_ids, per_rank = zigzag_split(np.array(PREFILL_LENS), world)
    rows, width = sum(PREFILL_LENS), 16
    expected = torch.arange(rows, dtype=torch.int32)[:, None] * 10 + torch.arange(
        width, dtype=torch.int32
    )
    expected[:, -3:] = -1
    locals_ = _split_by_rank(expected, token_ids, per_rank, world)
    ids = torch.from_numpy(token_ids.astype(np.int64))

    for rank in range(world):
        group = _FakeGroup(rank, locals_)
        monkeypatch.setattr(sparse_indexer, "get_tp_group", lambda g=group: g)
        got = sparse_indexer.all_gather_shardq_pool_ids(
            locals_[rank], per_rank.tolist(), ids, int(expected.max()) + 1
        )
        assert torch.equal(got, expected)
        assert got.dtype == torch.int32
        assert group.element_size == 2


def _top_pools(scores, num_valid, num_pools):
    """The selection kernel's layout: ascending pool ids, -1 past a row's pools."""
    pools = torch.arange(num_pools, device=scores.device)
    masked = scores.masked_fill(pools[None, :] >= num_valid[:, None], -float("inf"))
    ids = masked.topk(SELECT_K, dim=-1).indices.to(torch.int32)
    kept = torch.arange(SELECT_K, device=scores.device)[None, :] < num_valid[:, None]
    ids = torch.where(kept, ids, num_pools).sort(dim=-1).values
    return torch.where(ids == num_pools, -1, ids)


def _selection_round_trip(rank, world, port, num_pools):
    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
        )
        ensure_model_parallel_initialized(world, 1)
        torch.manual_seed(0)
        query_lens = np.array([601, 64, 200])
        context_lens = np.array([131000, 0, 12000])
        rows = int(query_lens.sum())
        # The indexer is replicated, so every rank scores a given row identically.
        scores = torch.randn(rows, num_pools, device="cuda")
        positions = torch.cat(
            [
                torch.arange(c, c + q, device="cuda")
                for c, q in zip(context_lens, query_lens)
            ]
        )
        num_valid = (positions // KPOOL + 1).clamp(max=num_pools)
        expected = _top_pools(scores, num_valid, num_pools)

        token_ids, per_rank = zigzag_split(query_lens, world)
        start = int(per_rank[:rank].sum())
        mine = torch.from_numpy(token_ids[start : start + per_rank[rank]]).cuda()
        got = sparse_indexer.all_gather_shardq_pool_ids(
            _top_pools(scores[mine], num_valid[mine], num_pools),
            per_rank.tolist(),
            torch.from_numpy(token_ids.astype(np.int64)).cuda(),
            num_pools,
        )
        assert torch.equal(got, expected)
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.parametrize("num_pools", [32767, 40000])
def test_shardq_selection_matches_unsharded(num_pools):
    """Scoring only a rank's own tokens and gathering selects the same pools."""
    world = 2
    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    mp.spawn(
        _selection_round_trip, args=(world, get_open_port(), num_pools), nprocs=world
    )


def _cdiv(a, b):
    return -(-a // b)


def _shared_split_round_trip(rank, world, port, num_spec):
    from types import SimpleNamespace

    from transformers import Glm5NextTextConfig

    from vllm.config import CacheConfig, ParallelConfig, SchedulerConfig
    from vllm.forward_context import set_forward_context
    from vllm.models.glm5next.amd.sparse_indexer import (
        _SHARDQ_MIN_SEQ_LEN,
        SparseAttnIndexerKpool,
    )
    from vllm.models.glm5next.common.attention import Glm5NextIndexerCache
    from vllm.models.glm5next.common.model import get_sparse_topk_buffer_width
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
        get_max_prefill_buffer_size,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        indexer_k_quant_and_cache_triton,
    )
    from vllm.v1.kv_cache_interface import get_block_table_width
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    text_cfg = Glm5NextTextConfig(
        index_topk=2048, index_kpool=KPOOL, index_n_heads=32, index_head_dim=128
    )
    head_dim, block_size, max_model_len = 128, 256, 32768
    cfg = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=1024,
            max_model_len=max_model_len,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(block_size=block_size),
        parallel_config=ParallelConfig(tensor_parallel_size=world),
    )
    cfg.attention_config.sparse_mla_shard_q_min_tokens = 64
    cfg.attention_config.sparse_mla_prefill_workspace_factor = 5
    with set_current_vllm_config(cfg):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
        )
        ensure_model_parallel_initialized(world, 1)
        cfg.model_config = SimpleNamespace(
            hf_text_config=text_cfg, max_model_len=max_model_len
        )
        cfg.speculative_config = SimpleNamespace(
            num_speculative_tokens=num_spec, enable_adaptive_verification=False
        )
        init_workspace_manager(device)

        # A decode, then ragged prefills past the floor. With speculation, only
        # ShardQ counts the 1-token request as a decode.
        reqs = [(17000, 1), (16384, 96), (20000, 161)]
        if num_spec:
            reqs = [(18000, 2), *reqs]
        context_lens, query_lens = map(list, zip(*reqs))
        seq_lens = [c + q for c, q in zip(context_lens, query_lens)]
        num_tokens = sum(query_lens)
        width = get_sparse_topk_buffer_width(text_cfg)
        # A layer that opted in, one that did not (MTP), and a reference.
        prefixes = [f"model.layers.{i}.self_attn.indexer.k_cache" for i in (3, 45, 7)]
        ops = []
        for prefix in prefixes:
            k_cache = Glm5NextIndexerCache(
                head_dim=head_dim + 4,
                dtype=torch.uint8,
                prefix=prefix,
                cache_config=cfg.cache_config,
                index_kpool=KPOOL,
            )
            ops.append(
                SparseAttnIndexerKpool(
                    k_cache,
                    head_dim,
                    "ue8m0",
                    text_cfg.index_topk,
                    head_dim,
                    max_model_len // KPOOL,
                    get_max_prefill_buffer_size(cfg) // KPOOL,
                    torch.full(
                        (num_tokens, width), -7, dtype=torch.int32, device=device
                    ),
                    skip_k_cache_insert=True,
                )
            )
        assert ops[0].enable_shardq()

        # The same random keys on every rank, written per pool state.
        per_block = block_size // KPOOL
        num_blocks = [_cdiv(_cdiv(s, KPOOL), per_block) for s in seq_lens]
        first = np.concatenate(([0], np.cumsum(num_blocks)[:-1]))
        kv_cache = torch.zeros(
            (sum(num_blocks), per_block, head_dim + 4), dtype=torch.uint8, device=device
        )
        bt_width = get_block_table_width(max_model_len // block_size, block_size)
        block_table = torch.zeros(
            (len(reqs), bt_width), dtype=torch.int32, device=device
        )
        state_slots, token_slots = [], []
        for r, (c, s) in enumerate(zip(context_lens, seq_lens)):
            block_table[r, : num_blocks[r]] = torch.arange(
                first[r], first[r] + num_blocks[r]
            )
            states = torch.arange(_cdiv(s, KPOOL))
            state_slots.append(
                (first[r] + states // per_block) * per_block + states % per_block
            )
            pos = torch.arange(c, s)
            token_slots.append(
                (first[r] + pos // block_size) * block_size + pos % block_size
            )
        torch.manual_seed(0)
        slots = torch.cat(state_slots).to(device)
        k_states = torch.randn((slots.numel(), head_dim), device=device).bfloat16()
        indexer_k_quant_and_cache_triton(k_states, kv_cache, slots, head_dim, "ue8m0")
        for op in ops:
            op.k_cache.kv_cache = kv_cache

        def make_common(reqs, block_table, slot_mapping):
            qsl = torch.tensor([0, *np.cumsum([q for _, q in reqs])], dtype=torch.int32)
            seq = torch.tensor([c + q for c, q in reqs], dtype=torch.int32)
            return CommonAttentionMetadata(
                query_start_loc=qsl.to(device),
                query_start_loc_cpu=qsl,
                seq_lens=seq.to(device),
                seq_lens_cpu_upper_bound=seq,
                num_reqs=len(reqs),
                num_actual_tokens=int(qsl[-1]),
                max_query_len=max(q for _, q in reqs),
                max_seq_len=int(seq.max()),
                block_table_tensor=block_table,
                slot_mapping=slot_mapping,
                causal=True,
            )

        spec = ops[0].k_cache.get_kv_cache_config(cfg).spec

        def build(layer_names, common):
            return DeepseekV32IndexerMetadataBuilder(
                kv_cache_spec=spec,
                layer_names=layer_names,
                vllm_config=cfg,
                device=device,
                block_table_width=bt_width,
            ).build(0, common)

        common = make_common(reqs, block_table, torch.cat(token_slots).to(device))
        shared, replicated = build(prefixes[:2], common), build(prefixes[2:], common)
        split = shared.prefill.shardq_split
        assert split is not None and replicated.prefill.shardq_split is None
        assert (split.num_decode_tokens > shared.num_decode_tokens) == bool(num_spec)
        # The split starts at the floor, not one token below it.
        at_floor = [
            build(
                prefixes[:1],
                make_common([(s - 96, 96)], block_table[-1:], common.slot_mapping[:96]),
            ).prefill.shardq_split
            is not None
            for s in (_SHARDQ_MIN_SEQ_LEN - 1, _SHARDQ_MIN_SEQ_LEN)
        ]
        assert at_floor == [False, True]

        q = torch.randn((num_tokens, 32, head_dim), device=device)
        q = q.to(current_platform.fp8_dtype())
        weights = torch.rand((num_tokens, 32), device=device)
        hidden = torch.zeros((num_tokens, 64), dtype=torch.bfloat16, device=device)
        k = torch.zeros((num_tokens, head_dim), dtype=torch.bfloat16, device=device)
        positions = torch.cat(
            [torch.arange(c, s) for c, s in zip(context_lens, seq_lens)]
        ).to(device)
        metadata = dict(zip(prefixes, (shared, shared, replicated)))
        with set_forward_context(metadata, cfg):
            outs = [
                op(hidden, q, k, weights, index_kpool=KPOOL, positions=positions)
                for op in ops
            ]
        expected = outs[2].sort(dim=-1).values
        assert (expected[:, -1] >= 0).all()
        differing = [
            int((out.sort(dim=-1).values != expected).any(dim=-1).sum())
            for out in outs[:2]
        ]
        assert differing == [0, 0], f"rows differing (opted in, not): {differing}"
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm kpool indexer")
@pytest.mark.parametrize("num_spec", [0, 1])
def test_layer_reading_a_shared_split_restores_token_order(monkeypatch, num_spec):
    """A layer that did not opt in, like MTP on draft prefill, still gathers."""
    world = 2
    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    mp.spawn(
        _shared_split_round_trip,
        args=(world, get_open_port(), num_spec),
        nprocs=world,
    )
