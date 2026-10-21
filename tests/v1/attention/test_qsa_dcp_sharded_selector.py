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
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_dcp, qsa_dcp_selector
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops
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
@pytest.mark.parametrize("rows", [1, 80, 1920])
@pytest.mark.parametrize("case", ["plain", "ties", "padded"])
@pytest.mark.parametrize("interleave", [1, 4])
@pytest.mark.parametrize("world", [2, 4])
def test_sorted_key_merge_matches_select(world, interleave, case, rows):
    """The sorted-key pack and bitonic merge give the ids, order and -1
    padding of the candidate pack and full select: on a 1/256 score grid
    (ties), and with padded local top-k rows (one row all padding on ranks
    other than 0)."""
    topk, width = 512, 4096
    assert qsa_dcp_selector.uses_sorted_merge(world, topk)
    assert not qsa_dcp_selector.uses_sorted_merge(3, topk)
    assert not qsa_dcp_selector.uses_sorted_merge(4, 500)
    generator = torch.Generator().manual_seed(rows * 31 + world * 7 + interleave)
    candidates, keys = [], []
    for rank in range(world):
        logits = torch.randn(rows, width, generator=generator)
        if case == "ties":
            logits = (logits * 256).round() / 256
        shuffle = torch.randperm(topk, generator=generator)
        local = logits.topk(topk, dim=1).indices[:, shuffle].int()
        if case == "padded":
            local[torch.rand(rows, topk, generator=generator) < 0.25] = -1
            if rank > 0:
                local[0] = -1
        shard = qsa_cache.QSADCPShard(rank, world, interleave)
        logits, local = logits.cuda(), local.cuda()
        candidates.append(
            qsa_dcp_selector.pack_qsa_dcp_candidates(logits, local, shard)
        )
        keys.append(qsa_dcp_selector.pack_sorted_qsa_dcp_keys(logits, local, shard))
    expected = torch.full((rows, topk), 7, dtype=torch.int32, device="cuda")
    actual = torch.full_like(expected, 7)

    qsa_dcp_selector.select_qsa_dcp_topk(torch.stack(candidates), expected)
    qsa_dcp_selector.merge_sorted_qsa_dcp_keys(torch.stack(keys), actual)

    assert torch.equal(actual, expected)


@requires_cuda_triton
@pytest.mark.parametrize("topk", [24, 32])
@pytest.mark.parametrize("dcp_size", [2, 4])
def test_merge_gives_every_rank_the_dcp1_topk(dcp_size, topk, monkeypatch):
    """merge_qsa_dcp_topk (Triton pack, dim-0 all-gather, Triton select or
    sorted-key merge) turns each rank's local top-k into the DCP=1 top-k.
    The all-gather is simulated on one GPU from the packs of all ranks."""
    expected, ranks = _sharded_topk_case(dcp_size, topk)
    ranks = [(shard, logits.cuda(), local.cuda()) for shard, logits, local in ranks]
    if qsa_dcp_selector.uses_sorted_merge(dcp_size, topk):
        pack = qsa_dcp_selector.pack_sorted_qsa_dcp_keys
    else:
        pack = qsa_dcp_selector.pack_qsa_dcp_candidates
    gathered = torch.cat([pack(logits, local, shard) for shard, logits, local in ranks])
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
    """The warmup compiles the merge and fused expand+localize kernels for a
    sharded deployment; the world-1 shard warms nothing."""
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


# --- simulated N-rank selection, end to end ----------------------------------

# persistent_topk supports block top-k 512, 1024 or 2048; 2048 tokens / ratio 4
# = 512 blocks, the selector of the served config.
E2E_TOKEN_TOPK = 2048
E2E_STATES_PER_BLOCK = 16
E2E_INDEX_HEADS, E2E_INDEX_DIM = 4, 64
# (sequence length, query length): three uniform decodes (MTP verify rows),
# then two prefill chunks.
E2E_REQUESTS = ((3000, 4), (2600, 4), (77, 4), (2300, 37), (4100, 120))
E2E_NUM_DECODES, E2E_DECODE_QUERY_LEN = 3, 4


class _TwoPassDCPGroup:
    """Simulated dim-0 all-gather of the ranks' candidate or key packs in one
    process.

    Pass 1 runs every rank and records its packs (the merge result is thrown
    away). Pass 2 runs every rank again, checks that each pack holds the
    candidates of its pass-1 record and returns all ranks' packs, this
    rank's current one in its own slot. The radix top-k may order a row's
    candidates differently on each run; the merge does not depend on that
    order.
    """

    def __init__(self, world):
        self.world = world
        self.records = [[] for _ in range(world)]
        self.calls = [0] * world
        self.rank = 0
        self.replay = False

    def all_gather(self, tensor, dim):
        assert dim == 0
        index = self.calls[self.rank]
        self.calls[self.rank] += 1
        if not self.replay:
            self.records[self.rank].append(tensor.clone())
            return torch.zeros(
                (self.world * tensor.shape[0], *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
        assert torch.equal(_by_state(tensor), _by_state(self.records[self.rank][index]))
        packs = [self.records[r][index] for r in range(self.world)]
        packs[self.rank] = tensor
        return torch.cat(packs)

    def start(self, rank, replay):
        self.rank, self.replay = rank, replay
        if replay and rank == 0:
            self.calls = [0] * self.world


def _by_state(packed):
    """A ``[rows, k, 2]`` candidate pack with each row ordered by state id. A
    ``[rows, k]`` sorted-key pack is already in a fixed order."""
    if packed.dim() == 2:
        return packed
    order = packed[..., 1].argsort(dim=1, stable=True)
    return packed.gather(1, order.unsqueeze(-1).expand_as(packed))


def _e2e_batch(generator):
    """Global compressed keys per request, the query rows and their metadata."""
    keys = [
        torch.randn(seq_len // COMPRESS_RATIO, E2E_INDEX_DIM, generator=generator)
        .bfloat16()
        .cuda()
        for seq_len, _ in E2E_REQUESTS
    ]
    positions = torch.cat([torch.arange(s - q, s) for s, q in E2E_REQUESTS]).cuda()
    query_lens = torch.tensor([q for _, q in E2E_REQUESTS])
    token_to_req = torch.repeat_interleave(
        torch.arange(len(E2E_REQUESTS)), query_lens
    ).cuda()
    query_start_loc = (
        torch.cat((torch.zeros(1, dtype=torch.int64), query_lens.cumsum(0)))
        .int()
        .cuda()
    )
    visible = ((positions + 1) // COMPRESS_RATIO).int()
    q = (
        torch.randn(
            positions.numel(), E2E_INDEX_HEADS, E2E_INDEX_DIM, generator=generator
        )
        .bfloat16()
        .cuda()
    )
    return keys, positions, token_to_req, query_start_loc, visible, q


def _e2e_cache(keys, shard):
    """A compressed key cache written through the (sharded) slot mapping: every
    token of every request is offered; only group-closing tokens whose state
    this rank owns get a slot."""
    max_states = max(k.shape[0] for k in keys)
    local_states = shard.max_local_length(max_states)
    blocks_per_request = -(-local_states // E2E_STATES_PER_BLOCK)
    # Block 0 is the null block.
    table = (
        (torch.arange(len(keys) * blocks_per_request, dtype=torch.int32) + 1)
        .view(len(keys), blocks_per_request)
        .cuda()
    )
    cache = torch.zeros(
        len(keys) * blocks_per_request + 1,
        E2E_STATES_PER_BLOCK,
        1,
        E2E_INDEX_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    for request, request_keys in enumerate(keys):
        positions = torch.arange(request_keys.shape[0] * COMPRESS_RATIO).cuda()
        slots = qsa_cache.compressed_qsa_slot_mapping(
            table,
            torch.full_like(positions, request),
            positions,
            E2E_STATES_PER_BLOCK,
            COMPRESS_RATIO,
            selector_shard=shard,
        )
        rows = request_keys.repeat_interleave(COMPRESS_RATIO, dim=0)
        qsa_ops.qsa_store_cache_rows(cache, slots, rows)
    return cache, table


def _e2e_select(q, cache, table, query_start_loc, visible, max_seq_len, shard):
    """The indexer's decode and prefill select calls on one rank."""
    rows = q.shape[0]
    decode_rows = E2E_NUM_DECODES * E2E_DECODE_QUERY_LEN
    block_indices = torch.empty(
        rows, E2E_TOKEN_TOPK // COMPRESS_RATIO, dtype=torch.int32, device="cuda"
    )
    qsa_indexer_ops.qsa_select_paged_decode(
        q[:decode_rows],
        cache,
        table[:E2E_NUM_DECODES],
        visible[:decode_rows],
        E2E_TOKEN_TOPK,
        COMPRESS_RATIO,
        E2E_DECODE_QUERY_LEN,
        block_indices[:decode_rows],
        shard,
    )
    qsa_indexer_ops.qsa_select_paged_prefill(
        q[decode_rows:],
        cache,
        table[E2E_NUM_DECODES:],
        query_start_loc[E2E_NUM_DECODES:],
        visible[decode_rows:],
        E2E_TOKEN_TOPK,
        COMPRESS_RATIO,
        max(q for _, q in E2E_REQUESTS[E2E_NUM_DECODES:]),
        block_indices[decode_rows:],
        max_seq_len,
        shard,
    )
    return block_indices


def _sorted_rows(rows):
    return [sorted(row) for row in rows.tolist()]


@requires_cuda_triton
@pytest.mark.parametrize("through_indexer", [False, True])
@pytest.mark.parametrize(("world", "token_interleave"), [(2, 4), (4, 4), (4, 16)])
def test_sharded_selection_matches_replicated_expand_and_localize(
    world, token_interleave, through_indexer, monkeypatch
):
    """Per-rank selector caches written through the sharded slot mapping,
    scored by the real decode and prefill selectors, merged by a simulated
    dim-0 all-gather (one call per select call) and expanded+localized by
    the fused kernel give, on every rank, the rows that the replicated
    selector on the full cache gives after expand and localize. Order inside
    a row may differ (the merge orders by score, the radix top-k does not),
    so rows compare as sorted lists; the counts compare exactly."""
    generator = torch.Generator().manual_seed(world * 100 + token_interleave)
    keys, positions, token_to_req, query_start_loc, visible, q = _e2e_batch(generator)
    max_seq_len = max(s for s, _ in E2E_REQUESTS)
    output_width = E2E_TOKEN_TOPK + COMPRESS_RATIO - 1

    full_cache, full_table = _e2e_cache(keys, qsa_cache.QSADCPShard())
    full_blocks = _e2e_select(
        q, full_cache, full_table, query_start_loc, visible, max_seq_len, None
    )
    full_rows = torch.empty(
        q.shape[0], output_width + 1, dtype=torch.int32, device="cuda"
    )
    qsa_indexer_ops.expand_qsa_block_indices(
        full_blocks, positions, visible, COMPRESS_RATIO, E2E_TOKEN_TOPK, full_rows
    )

    group = _TwoPassDCPGroup(world)
    monkeypatch.setattr(qsa_dcp_selector, "get_dcp_group", lambda: group)
    selector_interleave = token_interleave // COMPRESS_RATIO
    ranks = []
    for rank in range(world):
        shard = qsa_cache.QSADCPShard(rank, world, selector_interleave)
        cache, table = _e2e_cache(keys, shard)
        ranks.append((shard, cache, table, shard.local_lengths(visible)))

    def run_rank(rank):
        shard, cache, table, local_visible = ranks[rank]
        token_shard = qsa_cache.QSADCPShard(rank, world, token_interleave)
        local_rows = torch.full(
            (q.shape[0], output_width + 1), 7, dtype=torch.int32, device="cuda"
        )
        if through_indexer:
            _select_through_indexer(
                monkeypatch,
                q,
                cache,
                table,
                positions,
                token_to_req,
                query_start_loc,
                visible,
                local_visible,
                max_seq_len,
                shard,
                token_shard,
                local_rows,
            )
            return None, local_rows
        blocks = _e2e_select(
            q, cache, table, query_start_loc, local_visible, max_seq_len, shard
        )
        qsa_dcp_selector.expand_localize_qsa_block_indices(
            blocks,
            positions,
            visible,
            COMPRESS_RATIO,
            E2E_TOKEN_TOPK,
            local_rows,
            token_shard,
        )
        return blocks, local_rows

    for rank in range(world):
        group.start(rank, replay=False)
        run_rank(rank)
    for rank in range(world):
        group.start(rank, replay=True)
        blocks, local_rows = run_rank(rank)
        # One all-gather per select call: decode and prefill.
        assert group.calls[rank] == 2
        expected = qsa_dcp.qsa_localize_dcp_indices(
            full_rows, torch.empty_like(full_rows), world, rank, token_interleave
        )
        if blocks is not None:
            assert _sorted_rows(blocks) == _sorted_rows(full_blocks)
        assert torch.equal(local_rows[:, -1], expected[:, -1])
        for row in range(q.shape[0]):
            count = int(expected[row, -1])
            assert sorted(local_rows[row, :count].tolist()) == sorted(
                expected[row, :count].tolist()
            )
            assert bool((local_rows[row, count:output_width] == -1).all())


def _select_through_indexer(
    monkeypatch,
    q,
    cache,
    table,
    positions,
    token_to_req,
    query_start_loc,
    visible,
    local_visible,
    max_seq_len,
    shard,
    token_shard,
    out,
):
    """QSAIndexer.forward on one rank, sharded branch, real kernels.

    The fused prepare is replaced by a copy of the index queries (the
    caches are written beforehand)."""
    from vllm.models.qwen4_exp.nvidia import indexer_qsa

    def prepare(*args, **kwargs):
        args[7].copy_(q)

    monkeypatch.setattr(indexer_qsa, "qsa_prepare", prepare)
    rows = q.shape[0]
    decode_rows = E2E_NUM_DECODES * E2E_DECODE_QUERY_LEN
    raw_metadata = SimpleNamespace(
        num_actual_tokens=rows,
        slot_mapping=torch.full((rows,), -1, device="cuda"),
        block_table=table,
        query_start_loc=query_start_loc,
        logical_positions=positions,
    )
    compressed_metadata = SimpleNamespace(
        num_actual_tokens=rows,
        slot_mapping=torch.full((rows,), -1, device="cuda"),
        k_work_metadata=torch.empty(0, 2, dtype=torch.int32, device="cuda"),
        num_decodes=E2E_NUM_DECODES,
        num_decode_tokens=decode_rows,
        decode_query_len=E2E_DECODE_QUERY_LEN,
        visible_blocks=visible,
        local_visible_blocks=local_visible,
        selector_shard=shard,
        block_table=table,
        query_start_loc=query_start_loc,
        logical_positions=positions,
        max_query_len=max(q for _, q in E2E_REQUESTS),
        max_seq_len=max_seq_len,
    )
    indexer = SimpleNamespace(
        skip_topk=False,
        _metadata=lambda: (raw_metadata, compressed_metadata),
        index_n_heads=E2E_INDEX_HEADS,
        index_kv_heads=1,
        index_head_dim=E2E_INDEX_DIM,
        indexer_dtype=torch.bfloat16,
        raw_key_cache=SimpleNamespace(
            kv_cache=torch.empty(0, device="cuda"),
            rope_position_cache=None,
            rope_position_offset=0,
        ),
        compressed_key_cache=SimpleNamespace(kv_cache=cache),
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.empty(0, device="cuda")),
        q_layernorm=SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-6),
        k_layernorm=SimpleNamespace(weight=torch.ones(1)),
        compress_ratio=COMPRESS_RATIO,
        token_topk=E2E_TOKEN_TOPK,
        packed_output_width=out.shape[1],
    )
    attn = SimpleNamespace(
        use_fused_qsa_prepare=True,
        kv_cache=torch.empty(0, 1, 1, 2, device="cuda"),
        kv_cache_dtype="auto",
        q_norm=SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.ones(1)),
        _k_scale_float=1.0,
        _v_scale_float=1.0,
        impl=SimpleNamespace(dcp_token_shard=token_shard),
    )
    projected_qk = torch.zeros(
        rows, (E2E_INDEX_HEADS + 1) * E2E_INDEX_DIM, device="cuda"
    )
    returned, _ = indexer_qsa.QSAIndexer.forward(
        indexer,
        projected_qk,
        positions,
        out,
        attn=attn,
        qkv=torch.zeros(rows, 4, device="cuda"),
        slot_mapping=torch.full((rows,), -1, device="cuda"),
    )
    assert returned is out
