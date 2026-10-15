# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

import multiprocess as mp
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("the MiniMax-M3 MSA indexer is AMD-only", allow_module_level=True)

import vllm.platforms.rocm as rocm_platform  # noqa: E402
from tests.utils import ensure_current_vllm_config  # noqa: E402
from vllm.distributed.parallel_state import (  # noqa: E402
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.models.minimax_m3.amd.indexer_msa import (  # noqa: E402
    MAX_SUPPORTED_BLOCKS,
    MSA_INDEX_HEAD_DIM,
    MSA_SCORE_TYPE,
    MSA_SPARSE_BLOCK_SIZE,
    MSA_TOPK_BLOCKS,
    SLOTS_MAX,
    WAVE_SIZE,
)
from vllm.models.minimax_m3.amd.ops.sparse_pa import (  # noqa: E402
    ASM_PAGE_SIZE,
    PAGE16_SIDES_PER_BLOCK,
)
from vllm.utils.network_utils import get_open_port  # noqa: E402
from vllm.utils.system_utils import update_environment_variables  # noqa: E402

mp.set_start_method("spawn", force=True)

# No head count: the gate takes one only to describe a CP shape, and the
# default cp_world of 1 is the unsharded indexer, which does not turn on one.
_ADMITTED = dict(
    topk_blocks=MSA_TOPK_BLOCKS,
    sparse_block_size=MSA_SPARSE_BLOCK_SIZE,
    index_head_dim=MSA_INDEX_HEAD_DIM,
    indexer_kv_dtype="fp8",
    max_model_len=8192,
    score_type=MSA_SCORE_TYPE,
)

# The same config sharded, at MiniMax-M3's own shape: 4 index heads over 4
# ranks is one apiece, and 4 x top-16 fills the merge exactly.
_ADMITTED_CP = dict(
    _ADMITTED,
    cp_world=4,
    num_index_heads=4,
    num_kv_heads=1,
    max_query_len=1,
    dcp_size=1,
)

# The longest context the top-k has register slots for.
_MAX_LEN = MAX_SUPPORTED_BLOCKS * MSA_SPARSE_BLOCK_SIZE


@pytest.fixture
def aiter_msa_indexer_gate(monkeypatch):
    import vllm.models.minimax_m3.amd.indexer_msa as indexer_msa_mod

    def probe(rocm=True, gfx950=True, aiter_attend=True, base=_ADMITTED, **overrides):
        monkeypatch.setattr(indexer_msa_mod.current_platform, "is_rocm", lambda: rocm)
        monkeypatch.setattr(
            indexer_msa_mod,
            "_minimax_m3_aiter_sparse_pa_requested",
            lambda: aiter_attend,
        )
        monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: gfx950)
        return indexer_msa_mod.msa_indexer_unsupported_reason(**{**base, **overrides})

    return probe


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({}, None),
        # Two ranks owning two index heads each, matching the model's kv split.
        ({"cp_world": 2, "num_kv_heads": 2}, None),
        # Sharding does not excuse the unsharded contract: the same checks run
        # first, so one gate cannot admit at one width what it refuses at the
        # other.
        ({"indexer_kv_dtype": "bf16"}, "fp8 e4m3 index cache"),
        ({"gfx950": False}, "gfx950"),
        ({"max_model_len": _MAX_LEN + 1}, "the top-k is compiled for"),
        # And then the conditions only sharding runs into.
        ({"dcp_size": 2}, "decode_context_parallel_size"),
        ({"cp_world": 3}, "divisible"),
        ({"cp_world": 8, "num_index_heads": 8}, "candidates the merge holds"),
        ({"max_query_len": 5}, "MFMA columns"),
        ({"num_kv_heads": 2}, "1:1"),
    ],
)
def test_msa_indexer_gate_rejects_unshardable_configs(
    aiter_msa_indexer_gate, override, expected
):
    reason = aiter_msa_indexer_gate(base=_ADMITTED_CP, **override)
    if expected is None:
        assert reason is None, f"{override} should be admitted, got {reason!r}"
    else:
        assert reason is not None, f"{override} should be refused"
        assert expected in reason, f"{override} refused for the wrong reason: {reason}"


def test_msa_slot_cap_matches_what_aiter_compiled():
    msa_block_select = pytest.importorskip("aiter.ops.msa_block_select")

    assert SLOTS_MAX == msa_block_select.TOPK_MAX_SLOTS
    assert WAVE_SIZE == msa_block_select.WAVE_SIZE

    def aiter_accepts(blocks: int) -> bool:
        strips = -(-blocks // WAVE_SIZE)
        return 1 << (strips - 1).bit_length() <= SLOTS_MAX

    assert aiter_accepts(MAX_SUPPORTED_BLOCKS)
    assert not aiter_accepts(MAX_SUPPORTED_BLOCKS + 1)


_INDEX_HEADS = 4  # MiniMax-M3's own count, which is also the CP ceiling
_QUERY_LEN = 1
_INIT_BLOCKS = 1
_LOCAL_BLOCKS = 2
_PAGES_PER_BLOCK = MSA_SPARSE_BLOCK_SIZE // ASM_PAGE_SIZE
_BATCH_SIZES = (32, 64)
_CONTEXT_LENS = (16 * 1024, 64 * 1024, 128 * 1024)
_SHAPES = [(batch, ctx) for batch in _BATCH_SIZES for ctx in _CONTEXT_LENS]
_MAX_ROWS = max(_BATCH_SIZES) * _QUERY_LEN
_FP8 = torch.float8_e4m3fn
_UNWRITTEN = -7


def _shape_id(shape) -> str:
    batch, ctx = shape
    return f"b{batch}-{ctx // 1024}k"


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _score_cols(max_seq_len: int, world: int = 1) -> int:
    """Block-axis width both kernels demand of a shard of ``world``.

    Their contract, not a copy of the indexer's helper: lanes read whole
    wave-wide strips and hold a power-of-two count of them.
    """
    blocks = _cdiv(_cdiv(max_seq_len, MSA_SPARSE_BLOCK_SIZE), world)
    strips = _cdiv(blocks, WAVE_SIZE)
    return (1 << (strips - 1).bit_length()) * WAVE_SIZE


def _require_msa_kernels():
    if not rocm_platform.on_gfx950():
        pytest.skip("the MSA indexer's kernels are gfx950-only")
    pytest.importorskip("aiter.ops.msa_attention")


def _decode_inputs(batch: int, ctx: int, device: torch.device):
    """A decode batch, identical on every rank that builds it.

    Drawn from a seeded CPU generator and then copied over rather than drawn
    on the device, so the ranks agree by construction instead of by trusting
    that per-device RNG streams line up.

    The index cache holds one row's worth of pages and each request gets its
    own permutation of them. The requests still read different keys, but the
    cache stops growing with the batch -- at 128K over 64 requests the honest
    allocation is about a gigabyte.
    """
    gen = torch.Generator().manual_seed(0x5EED)
    blocks = _cdiv(ctx, MSA_SPARSE_BLOCK_SIZE)
    rows = batch * _QUERY_LEN

    query = torch.randn(rows, _INDEX_HEADS, MSA_INDEX_HEAD_DIM, generator=gen)
    cache = torch.randn(
        blocks, MSA_SPARSE_BLOCK_SIZE, MSA_INDEX_HEAD_DIM, generator=gen
    )
    block_table = torch.stack(
        [torch.randperm(blocks, generator=gen) for _ in range(batch)]
    )
    # Long enough that the forced blocks and the top-k never cover the whole
    # row, or every selection would be the trivial one.
    floor = MSA_SPARSE_BLOCK_SIZE * (MSA_TOPK_BLOCKS + _INIT_BLOCKS + _LOCAL_BLOCKS)
    seq_lens = torch.randint(floor, ctx + 1, (batch,), generator=gen)
    seq_lens[0] = ctx  # one row pinned to the full context

    return (
        query.to(device).to(_FP8),
        cache.to(device).to(_FP8),
        block_table.to(device).to(torch.int32),
        seq_lens.to(device).to(torch.int32),
    )


@pytest.mark.parametrize("cp_world", [2, 4])
@pytest.mark.parametrize("shape", _SHAPES, ids=_shape_id)
def test_cp_score_shard_holds_the_unsharded_scores(cp_world, shape):
    """Each shard's scores are the unsharded ones at its round-robin columns.

    One GPU and no exchange: this is the half of the chain that can be
    checked without peers, and the multi-rank test below leans on it.
    """
    _require_msa_kernels()
    from aiter.ops.msa_attention import pa_sparse_block_score_decode

    batch, ctx = shape
    device = torch.device("cuda")
    query, cache, block_table, seq_lens = _decode_inputs(batch, ctx, device)
    blocks = _cdiv(ctx, MSA_SPARSE_BLOCK_SIZE)
    rows = batch * _QUERY_LEN

    def score(cols: int, **shard) -> torch.Tensor:
        # Blocks past a row's causal reach are left untouched, so the fill is
        # what makes the two sides comparable there as well.
        out = torch.full(
            (_INDEX_HEADS, rows, cols),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        pa_sparse_block_score_decode(
            query,
            cache,
            out,
            block_table,
            seq_lens,
            init_blocks=_INIT_BLOCKS,
            local_blocks=_LOCAL_BLOCKS,
            query_len=_QUERY_LEN,
            max_seq_len=ctx,
            **shard,
        )
        return out

    unsharded = score(_score_cols(ctx))
    # Row 0 spans the whole context, so every block of it is in reach and the
    # comparison below cannot be passing on a buffer of untouched -inf.
    assert torch.isfinite(unsharded[:, 0, :blocks]).all()

    for rank in range(cp_world):
        shard = score(_score_cols(ctx, cp_world), rank=rank, world=cp_world)
        owned = _cdiv(blocks - rank, cp_world)
        assert torch.equal(
            shard[:, :, :owned],
            unsharded[:, :, rank : rank + owned * cp_world : cp_world],
        ), f"rank {rank}/{cp_world} does not hold its round-robin columns"


_JOIN_TIMEOUT_S = 300


def _check_cp_selection(shape, rank: int, world: int, device, peers) -> None:
    """One shape's CP decode against an unsharded decode of the same inputs."""
    from aiter.ops.msa_attention import (
        pa_sparse_block_score_decode,
        pa_sparse_block_topk,
        pa_sparse_block_topk_cp,
    )

    batch, ctx = shape
    query, cache, block_table, seq_lens = _decode_inputs(batch, ctx, device)
    page16 = block_table * PAGE16_SIDES_PER_BLOCK
    rows = batch * _QUERY_LEN
    owned = _INDEX_HEADS // world
    lo = rank * owned

    def outputs():
        return (
            torch.full(
                (owned, rows, MSA_TOPK_BLOCKS),
                _UNWRITTEN,
                dtype=torch.int32,
                device=device,
            ),
            torch.full(
                (rows * owned, MSA_TOPK_BLOCKS * _PAGES_PER_BLOCK),
                _UNWRITTEN,
                dtype=torch.int32,
                device=device,
            ),
            torch.full((rows * owned,), _UNWRITTEN, dtype=torch.int32, device=device),
        )

    shard = torch.full(
        (_INDEX_HEADS, rows, _score_cols(ctx, world)),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    pa_sparse_block_score_decode(
        query,
        cache,
        shard,
        block_table,
        seq_lens,
        init_blocks=_INIT_BLOCKS,
        local_blocks=_LOCAL_BLOCKS,
        query_len=_QUERY_LEN,
        max_seq_len=ctx,
        rank=rank,
        world=world,
    )
    cp_topk, cp_bt, cp_ctx = outputs()
    pa_sparse_block_topk_cp(
        shard,
        peers.cand_ptrs,
        peers.cp_gen,
        cp_topk,
        seq_lens,
        page16,
        cp_bt,
        cp_ctx,
        max_seq_len=ctx,
        block_size=MSA_SPARSE_BLOCK_SIZE,
        cand_rows=_MAX_ROWS,
        query_len=_QUERY_LEN,
        init_blocks=_INIT_BLOCKS,
        local_blocks=_LOCAL_BLOCKS,
        num_kv_heads=owned,
        pages_per_block=_PAGES_PER_BLOCK,
        rank=rank,
        world=world,
    )

    unsharded = torch.full(
        (_INDEX_HEADS, rows, _score_cols(ctx)),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    pa_sparse_block_score_decode(
        query,
        cache,
        unsharded,
        block_table,
        seq_lens,
        init_blocks=_INIT_BLOCKS,
        local_blocks=_LOCAL_BLOCKS,
        query_len=_QUERY_LEN,
        max_seq_len=ctx,
    )
    ref_topk, ref_bt, ref_ctx = outputs()
    pa_sparse_block_topk(
        unsharded[lo : lo + owned].contiguous(),
        ref_topk,
        page16,
        seq_lens,
        sparse_bt=ref_bt,
        sparse_ctx=ref_ctx,
        max_seq_len=ctx,
        block_size=MSA_SPARSE_BLOCK_SIZE,
        query_len=_QUERY_LEN,
        num_kv_heads=owned,
        pages_per_block=_PAGES_PER_BLOCK,
    )

    torch.accelerator.synchronize()
    where = f"rank {rank}/{world} at {_shape_id(shape)}"
    assert (cp_topk != _UNWRITTEN).any(), f"{where}: the merge wrote no selection"
    assert torch.equal(cp_topk, ref_topk), f"{where}: selected different blocks"
    assert torch.equal(cp_bt, ref_bt), f"{where}: emitted a different page table"
    assert torch.equal(cp_ctx, ref_ctx), f"{where}: emitted different page counts"


def _cp_worker(env) -> None:
    update_environment_variables(env)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    torch.accelerator.set_device_index(device)

    from vllm.models.minimax_m3.amd.indexer_cp import get_indexer_cp_peers

    with ensure_current_vllm_config():
        init_distributed_environment()
        ensure_model_parallel_initialized(world, 1)

        peers = get_indexer_cp_peers(_INDEX_HEADS // world, _MAX_ROWS, MSA_TOPK_BLOCKS)
        for shape in _SHAPES:
            _check_cp_selection(shape, rank, world, device, peers)


def _distributed_run(fn, world_size: int) -> None:
    port = str(get_open_port())
    procs = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
        }
        proc = mp.Process(target=fn, args=(env,))
        proc.start()
        procs.append(proc)

    deadline = time.monotonic() + _JOIN_TIMEOUT_S
    for proc in procs:
        proc.join(max(0.0, deadline - time.monotonic()))
    hung = [i for i, proc in enumerate(procs) if proc.is_alive()]
    for proc in procs:
        if proc.is_alive():
            proc.kill()
        proc.join()

    assert not hung, f"ranks {hung} did not finish within {_JOIN_TIMEOUT_S}s"
    failed = {i: proc.exitcode for i, proc in enumerate(procs) if proc.exitcode != 0}
    assert not failed, f"ranks exited non-zero: {failed}"


@pytest.mark.parametrize("cp_world", [2, 4])
def test_indexer_cp_selects_what_an_unsharded_indexer_selects(cp_world):
    """The real thing: ``cp_world`` processes, real IPC, real exchange.

    Each rank runs the production CP decode over its block shard and compares
    the selection against an unsharded decode of the same inputs over the
    heads it owns. The gating lives here rather than in the worker, where a
    skip reads back to the parent as a failed exit.
    """
    _require_msa_kernels()
    if torch.accelerator.device_count() < cp_world:
        pytest.skip(f"needs {cp_world} GPUs")

    _distributed_run(_cp_worker, cp_world)
