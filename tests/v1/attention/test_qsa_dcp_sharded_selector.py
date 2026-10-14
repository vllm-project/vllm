# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in sharded QSA selector under DCP: rank-local side-cache placement,
the exact global top-k merge (CPU reference, Triton parity on CUDA) and the
hybrid cache coordinator."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager
from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.nvidia.ops import qsa_dcp_selector
from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import expand_qsa_block_indices
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

COMPRESS_RATIO = 4
BLOCK_SIZE = 400
STATES_PER_BLOCK = BLOCK_SIZE // COMPRESS_RATIO
SEQ_LENS = (1700, 2403)
QUERY_LENS = (1700, 5)
INT64_MIN = torch.iinfo(torch.int64).min
LOW32 = (1 << 32) - 1

requires_cuda_triton = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA DCP kernels require CUDA and Triton",
)


def _sharded_config(dcp_size, interleave):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=interleave,
        ),
        attention_config=SimpleNamespace(qsa_dcp_selector="sharded"),
    )


def _reference(dcp_rank, dcp_size, interleave, block_table):
    """Runner token placement (block_table.py) applied to each group's last token."""
    slots, local_visible = [], []
    for request, (seq_len, query_len) in enumerate(zip(SEQ_LENS, QUERY_LENS)):
        for position in range(seq_len - query_len, seq_len):
            visible = min((position + 1) // COMPRESS_RATIO, seq_len // COMPRESS_RATIO)
            local_visible.append(
                sum(
                    state * COMPRESS_RATIO // interleave % dcp_size == dcp_rank
                    for state in range(visible)
                )
            )
            owner = position // interleave % dcp_size
            if (position + 1) % COMPRESS_RATIO or owner != dcp_rank:
                slots.append(-1)
                continue
            virtual_block, offset = divmod(position, BLOCK_SIZE * dcp_size)
            local_offset = (
                offset // (interleave * dcp_size) * interleave + offset % interleave
            )
            local_state = (virtual_block * BLOCK_SIZE + local_offset) // COMPRESS_RATIO
            block = int(block_table[request, local_state // STATES_PER_BLOCK])
            slots.append(block * STATES_PER_BLOCK + local_state % STATES_PER_BLOCK)
    return slots, local_visible


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(qsa_cache._build_qsa_metadata_torch, marks=pytest.mark.cpu_test),
        pytest.param(
            qsa_cache.build_qsa_metadata_triton,
            marks=pytest.mark.skipif(
                not current_platform.is_cuda() or not HAS_TRITON,
                reason="QSA metadata kernel requires CUDA and Triton",
            ),
        ),
    ],
)
@pytest.mark.parametrize("dcp_size", [1, 2, 4])
@pytest.mark.parametrize("interleave", [4, 16, 100])
def test_sharded_metadata_matches_runner_dcp_placement(build, dcp_size, interleave):
    """Each compressed state is written once, by the rank that owns its tokens,
    at the local slot the runner gives its last token; visible-state counts
    split the same way."""
    device = "cuda" if build is qsa_cache.build_qsa_metadata_triton else "cpu"
    num_tokens = sum(QUERY_LENS)
    query_start_loc = torch.tensor([0, QUERY_LENS[0], num_tokens], dtype=torch.int32)
    # Block 0 is the null block, which never receives a write.
    block_table = (torch.randperm(64, dtype=torch.int32) + 1).view(2, 32)
    token_to_req = torch.repeat_interleave(torch.arange(2), torch.tensor(QUERY_LENS))
    common = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc.to(device),
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64, device=device),
        block_table_tensor=block_table.to(device),
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )
    for dcp_rank in range(dcp_size):
        shard = qsa_cache.QSADCPShard(dcp_rank, dcp_size, interleave // COMPRESS_RATIO)
        local_visible = torch.empty(num_tokens, dtype=torch.int32, device=device)
        _, _, visible, slots = build(
            common,
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
            storage_block_size=STATES_PER_BLOCK,
            compress_ratio=COMPRESS_RATIO,
            selector_shard=shard,
            local_visible_blocks_buffer=local_visible,
        )
        if dcp_size == 1:
            local_visible = visible
        expected_slots, expected_visible = _reference(
            dcp_rank, dcp_size, interleave, block_table
        )
        assert slots.tolist() == expected_slots
        assert local_visible.tolist() == expected_visible
        assert max(expected_visible) <= shard.max_local_length(max(visible.tolist()))


@pytest.mark.cpu_test
@pytest.mark.parametrize("dcp_size", [1, 4])
def test_sharded_qsa_groups_pass_hybrid_coordinator_under_dcp(dcp_size):
    """Full attention, sharded compressed key, raw-key ring and Mamba groups
    coexist under DCP with fine-grained prefix hits; only the token-sharded
    groups scale their block size."""
    block_size = 16
    cache_args = dict(
        head_size=64,
        dtype=torch.bfloat16,
        cache_config=SimpleNamespace(block_size=block_size),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
        compress_ratio=COMPRESS_RATIO,
    )
    ring = qsa_cache.QSAKeyStateCache(prefix="ring", **cache_args).get_kv_cache_spec(
        SimpleNamespace(num_speculative_tokens=3)
    )
    compressed = qsa_cache.QSACompressedKeyCache(
        prefix="compressed", **cache_args
    ).get_kv_cache_spec(_sharded_config(dcp_size, COMPRESS_RATIO))
    full = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
    )
    mamba = MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    groups = {"full": full, "compressed": compressed, "ring": ring, "mamba": mamba}
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([name], g) for name, g in groups.items()],
    )

    manager = make_kv_cache_manager(
        kv_cache_config=config,
        max_model_len=1024,
        enable_caching=True,
        dcp_world_size=dcp_size,
        scheduler_block_size=block_size * dcp_size,
        hash_block_size=block_size // 2,
    )

    sharded = block_size * dcp_size
    assert compressed.dcp_sharded and not ring.dcp_sharded
    assert manager.coordinator.group_block_sizes == (sharded, sharded, 8, block_size)
    assert manager.coordinator.enable_partial_hash_hits


def _pack_reference(logits, local_indices, shard):
    """PyTorch form of the Triton pack: ``[rows, k, 2]`` (score, global state),
    ``(-inf, -1)`` for padding."""
    valid = local_indices >= 0
    safe_indices = local_indices.clamp_min(0).long()
    scores = logits.gather(1, safe_indices).masked_fill(~valid, float("-inf"))
    global_ids = torch.where(valid, shard.global_positions(safe_indices), -1)
    return torch.stack((scores, global_ids.to(torch.float32)), dim=-1)


def _select_reference(gathered, topk):
    """Global top-k of dim-1 gathered ``[rows, candidates, 2]`` rows with
    torch.topk: score descending, then global id ascending, padding last."""
    global_ids = gathered[..., 1].to(torch.int64)
    score_bits = gathered[..., 0].contiguous().view(torch.int32).to(torch.int64)
    score_key = torch.where(score_bits < 0, score_bits ^ 0x7FFFFFFF, score_bits)
    key = score_key * (1 << 32) + (LOW32 - global_ids)
    key.masked_fill_(global_ids < 0, INT64_MIN)
    top_keys, columns = torch.topk(key, topk, dim=1, sorted=True)
    selected = global_ids.gather(1, columns)
    return selected.masked_fill_(top_keys == INT64_MIN, -1).to(torch.int32)


def _select_rank_major_reference(rank_major):
    """The algorithm of ``_select_qsa_dcp_topk_kernel``: sort int64 keys of
    flipped score bits and inverted ids over a ``[world, rows, topk, 2]``
    gather."""
    world, rows, topk, _ = rank_major.shape
    bits = rank_major.view(torch.int32).permute(1, 0, 2, 3).reshape(rows, -1, 2)
    score_bits = bits[..., 0].long()
    ids = bits[..., 1].contiguous().view(torch.float32).long()
    score_key = torch.where(score_bits < 0, score_bits ^ 0x7FFFFFFF, score_bits)
    key = torch.where(ids >= 0, score_key * (1 << 32) + (LOW32 - ids), INT64_MIN)
    key = key.sort(dim=1, descending=True).values[:, :topk]
    return torch.where(key == INT64_MIN, -1, LOW32 - (key & LOW32)).int()


def _stable_topk(scores, count, topk):
    order = sorted(range(count), key=lambda column: (-scores[column], column))
    return order[:topk] + [-1] * (topk - min(topk, count))


def _sharded_topk_case(dcp_size, topk=32, interleave=4, width=400):
    """Dense logits with ties, the DCP=1 stable top-k, and each rank's local
    logits and unordered local top-k (as persistent_topk returns it)."""
    visible = torch.tensor([0, 5, 31, 32, 100, 400])
    generator = torch.Generator().manual_seed(dcp_size)
    logits = torch.randint(-4, 4, (len(visible), width), generator=generator).float()
    expected = [
        _stable_topk(row.tolist(), int(n), topk) for row, n in zip(logits, visible)
    ]
    ranks = []
    for rank in range(dcp_size):
        shard = qsa_cache.QSADCPShard(rank, dcp_size, interleave)
        local_logits = logits[
            :, shard.global_positions(torch.arange(width // dcp_size))
        ]
        local_visible = shard.local_lengths(visible).tolist()
        local_topk = torch.tensor(
            [
                _stable_topk(row.tolist(), n, topk)
                for row, n in zip(local_logits, local_visible)
            ]
        )
        shuffle = torch.randperm(topk, generator=generator)
        ranks.append((shard, local_logits, local_topk[:, shuffle].int()))
    return expected, ranks


@pytest.mark.cpu_test
@pytest.mark.parametrize("dcp_size", [1, 2, 4])
def test_sharded_topk_merge_matches_dcp1_stable_topk(dcp_size):
    """Merging per-rank local top-k candidates gives the DCP=1 top-k in stable
    order (score descending, then state ascending), ties and short rows too,
    for the dim-1 and the rank-major gather."""
    topk = 32
    expected, ranks = _sharded_topk_case(dcp_size, topk)
    candidates = [_pack_reference(*rank[1:], rank[0]) for rank in ranks]

    merged = _select_reference(torch.cat(candidates, dim=1), topk)
    rank_major = _select_rank_major_reference(torch.stack(candidates))

    assert merged.tolist() == expected
    assert rank_major.tolist() == expected


def _packed_candidates(rows, topk, world, generator):
    """Per-rank packed ``[rows, topk, 2]`` candidates with tied scores, -inf,
    -0.0, distinct ids up to 2**24 - 1, an empty row 0 and short rows."""
    order = torch.rand(rows, world * topk, generator=generator).argsort(1)
    offset = torch.randint(0, 1 << 24, (rows, 1), generator=generator)
    # An odd multiplier is a bijection modulo 2**24, so ids stay distinct.
    ids = ((order * 4099 + offset) % (1 << 24)).view(rows, world, topk)
    scores = torch.randint(-4, 5, ids.shape, generator=generator).float() * 0.5
    pick = torch.rand(ids.shape, generator=generator)
    scores[pick < 0.03] = float("-inf")
    scores[(pick >= 0.03) & (pick < 0.06)] = -0.0
    count = torch.where(
        torch.rand(rows, world, generator=generator) < 0.33,
        torch.randint(0, topk, (rows, world), generator=generator),
        topk,
    )
    count[0] = 0
    valid = torch.arange(topk) < count[..., None]
    scores = scores.masked_fill(~valid, float("-inf"))
    packed = torch.stack((scores, ids.masked_fill(~valid, -1).float()), dim=-1)
    return [packed[:, rank].contiguous() for rank in range(world)]


@pytest.mark.cpu_test
@pytest.mark.parametrize("rows", [80, 1024])
@pytest.mark.parametrize(("world", "topk"), [(2, 128), (4, 128), (4, 512)])
def test_rank_major_select_matches_dim1_select(rows, world, topk):
    """The Triton selector's key sort over the rank-major (dim-0) gather
    selects the same ids as torch.topk over the dim-1 gather."""
    generator = torch.Generator().manual_seed(rows * 31 + world * 7 + topk)
    candidates = _packed_candidates(rows, topk, world, generator)

    expected = _select_reference(torch.cat(candidates, 1), topk)

    assert (expected[0] == -1).all()
    assert torch.equal(_select_rank_major_reference(torch.stack(candidates)), expected)


@requires_cuda_triton
@pytest.mark.parametrize("topk", [128, 512, 2048])
@pytest.mark.parametrize(("world", "interleave"), [(4, 4), (4, 1), (2, 16)])
def test_triton_pack_is_bit_identical(topk, world, interleave):
    """The Triton candidate pack writes the bits of the PyTorch pack."""
    rows, width = 80, topk * 4
    generator = torch.Generator().manual_seed(topk + world + interleave)
    logits = torch.randint(-4, 5, (rows, width), generator=generator).float() / 2
    logits[torch.rand(rows, width, generator=generator) < 0.03] = float("-inf")
    order = torch.rand(rows, width, generator=generator).argsort(1)[:, :topk]
    keep = torch.randint(0, topk + 1, (rows, 1), generator=generator)
    keep[0], keep[1] = 0, topk
    local = torch.where(torch.arange(topk) < keep, order, -1).int()
    logits, local = logits.cuda(), local.cuda()
    for rank in range(world):
        shard = qsa_cache.QSADCPShard(rank, world, interleave)
        expected = _pack_reference(logits, local, shard)
        actual = qsa_dcp_selector.pack_qsa_dcp_candidates(logits, local, shard)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@requires_cuda_triton
@pytest.mark.parametrize("rows", [80, 1024])
@pytest.mark.parametrize("topk", [128, 512, 2048])
@pytest.mark.parametrize("world", [2, 4])
def test_triton_select_matches_torch_select(rows, topk, world):
    """The Triton selector over the rank-major gather equals torch.topk over
    the dim-1 gather of the same candidates."""
    generator = torch.Generator().manual_seed(rows * 3 + topk + world)
    candidates = [c.cuda() for c in _packed_candidates(rows, topk, world, generator)]
    expected = _select_reference(torch.cat(candidates, 1), topk)
    out = torch.full((rows, topk), 7, dtype=torch.int32, device="cuda")

    qsa_dcp_selector.select_qsa_dcp_topk(torch.stack(candidates), out)

    assert torch.equal(out, expected)


@requires_cuda_triton
@pytest.mark.parametrize("dcp_size", [2, 4])
def test_merge_gives_every_rank_the_dcp1_topk(dcp_size, monkeypatch):
    """merge_qsa_dcp_topk (Triton pack, dim-0 all-gather, Triton select)
    turns each rank's local top-k into the DCP=1 top-k. The all-gather is
    simulated on one GPU from the packs of all ranks."""
    topk = 32
    expected, ranks = _sharded_topk_case(dcp_size, topk)
    ranks = [(shard, logits.cuda(), local.cuda()) for shard, logits, local in ranks]
    gathered = torch.cat(
        [
            qsa_dcp_selector.pack_qsa_dcp_candidates(logits, local, shard)
            for shard, logits, local in ranks
        ]
    )
    group = SimpleNamespace(all_gather=lambda packed, dim: gathered)
    monkeypatch.setattr(qsa_dcp_selector, "get_dcp_group", lambda: group)
    for shard, logits, local in ranks:
        qsa_dcp_selector.merge_qsa_dcp_topk(logits, local, shard)
        assert local.tolist() == expected


def _expand_inputs(rows, compress_ratio, positions_dtype, token_topk=2048):
    """Block indices with -1 padding, query positions with -1 rows and open
    and closed compression groups, and visible counts with zero rows."""
    block_topk = token_topk // compress_ratio
    num_blocks = block_topk * 3
    generator = torch.Generator().manual_seed(rows + compress_ratio)
    blocks = torch.rand(rows, num_blocks, generator=generator).argsort(1)
    keep = torch.randint(0, block_topk + 1, (rows, 1), generator=generator)
    keep[: rows // 2] = block_topk
    block_indices = torch.where(
        torch.arange(block_topk) < keep, blocks[:, :block_topk], -1
    )
    positions = torch.randint(
        0, num_blocks * compress_ratio, (rows,), generator=generator
    )
    positions[::7] = -1
    positions[1::5] = (positions[1::5] // compress_ratio + 1) * compress_ratio - 1
    visible = torch.randint(0, block_topk * 2, (rows,), generator=generator)
    visible[::9] = 0
    return [
        block_indices.int().cuda(),
        positions.to(positions_dtype).cuda(),
        visible.int().cuda(),
    ]


@requires_cuda_triton
@pytest.mark.parametrize("rows", [80, 1024])
@pytest.mark.parametrize("compress_ratio", [2, 4, 8])
@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
def test_fused_expand_localize_matches_expand_then_localize(
    rows, compress_ratio, positions_dtype
):
    """The fused kernel writes what expand_qsa_block_indices followed by the
    in-place qsa_localize_dcp_indices writes, on every rank."""
    token_topk, world, interleave = 2048, 4, 16
    inputs = _expand_inputs(rows, compress_ratio, positions_dtype, token_topk)
    width = token_topk + compress_ratio - 1
    expanded = torch.full((rows, width + 1), 9, dtype=torch.int32, device="cuda")
    expand_qsa_block_indices(*inputs, compress_ratio, token_topk, expanded)
    for rank in range(world):
        fused = torch.full_like(expanded, 9)
        qsa_dcp_selector.expand_localize_qsa_block_indices(
            *inputs,
            compress_ratio,
            token_topk,
            fused,
            qsa_cache.QSADCPShard(rank, world, interleave),
        )
        reference = expanded.clone()
        qsa_localize_dcp_indices(reference, reference, world, rank, interleave)

        assert torch.equal(fused, reference)


@requires_cuda_triton
@pytest.mark.parametrize(("world", "token_interleave"), [(2, 4), (4, 16)])
def test_sharded_selector_warmup_compiles(world, token_interleave):
    """The warmup compiles the pack, select and fused expand+localize
    kernels for a sharded deployment; the world-1 shard warms nothing."""
    token_topk, compress_ratio = 2048, 4
    token_shard = qsa_cache.QSADCPShard(world - 1, world, token_interleave)
    selector_shard = qsa_cache.QSADCPShard(
        world - 1, world, token_interleave // compress_ratio
    )
    qsa_dcp_selector.warmup_qsa_dcp_selector(
        logits_width=4096,
        token_topk=token_topk,
        compress_ratio=compress_ratio,
        selector_shard=selector_shard,
        token_shard=token_shard,
    )
    qsa_dcp_selector.warmup_qsa_dcp_selector(
        logits_width=4096,
        token_topk=token_topk,
        compress_ratio=compress_ratio,
        selector_shard=qsa_cache.QSADCPShard(),
        token_shard=qsa_cache.QSADCPShard(),
    )
