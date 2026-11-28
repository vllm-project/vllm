# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCm Lamport P2P fused RS + short-conv + AG + add + RMSNorm.

Covers `vllm/models/inkling/amd/ops/lamport.py`:

* `initialize_lamport_rs_conv` / `get_lamport_rs_conv` (global state),
* `LamportRSConv.usable` / `rs_sconv_ag_add_norm` / `close`.

The multi-GPU tests spawn one process per TP rank and compare every rank's
output against a pure-torch reference that follows the HIP kernels' rounding
points. Shapes use the Inkling and Inkling-Small configs.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import vllm.models.inkling.amd.ops.lamport as lamport
from vllm.models.inkling.amd.ops.lamport import (
    _MAX_TOKENS,
    LamportRSConv,
    get_lamport_rs_conv,
    initialize_lamport_rs_conv,
)


@dataclass(frozen=True)
class ModelShape:
    name: str
    hidden_size: int
    window_size: int  # text_config.sconv_kernel_size
    max_tokens: int  # min(_MAX_TOKENS, max_num_batched_tokens)
    # Global-layer conv-state cache layout (InklingConvState).
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6


INKLING = ModelShape(
    "Inkling",
    hidden_size=6144,
    window_size=4,
    max_tokens=_MAX_TOKENS,
    num_kv_heads=8,
    head_dim=128,
)
INKLING_SMALL = ModelShape(
    "Inkling-Small",
    hidden_size=4096,
    window_size=4,
    max_tokens=_MAX_TOKENS,
    num_kv_heads=8,
    head_dim=128,
)
MODELS = [pytest.param(INKLING, id="inkling"), pytest.param(INKLING_SMALL, id="small")]
TP_SIZES = [2, 4, 8]
_SENTINEL = -0x8000  # bf16 -0.0, Lamport "empty"
_WORKER_TIMEOUT_S = 900


@pytest.fixture
def clean_lamport_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LAMPORT_RS_SCONV", raising=False)
    monkeypatch.setattr(lamport, "_STATE", None)
    monkeypatch.setattr(lamport, "_STATE_FAILED", False)
    yield


# ---------------------------------------------------------------------------
# Single-process tests (no collectives)
# ---------------------------------------------------------------------------


class _FakeLamport:
    """Stand-in for LamportRSConv that records its constructor arguments."""

    instances: list[_FakeLamport] = []

    def __init__(self, hidden_size: int, window_size: int, max_tokens: int) -> None:
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.max_tokens = max_tokens
        _FakeLamport.instances.append(self)


@pytest.fixture
def fake_lamport(monkeypatch: pytest.MonkeyPatch, clean_lamport_state):
    _FakeLamport.instances = []
    monkeypatch.setattr(lamport, "LamportRSConv", _FakeLamport)
    return _FakeLamport


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_num_batched_tokens", [1, 8192, 16384, 65536])
def test_initialize_clamps_max_tokens_and_is_idempotent(
    fake_lamport, model: ModelShape, max_num_batched_tokens: int
):
    assert get_lamport_rs_conv(model.hidden_size, model.window_size) is None

    initialize_lamport_rs_conv(
        model.hidden_size, model.window_size, max_num_batched_tokens
    )
    state = get_lamport_rs_conv(model.hidden_size, model.window_size)
    assert isinstance(state, _FakeLamport)
    assert state.max_tokens == min(_MAX_TOKENS, max_num_batched_tokens)

    # A second (e.g. MTP draft) model with the same sizes reuses the state.
    initialize_lamport_rs_conv(model.hidden_size, model.window_size, 123)
    assert get_lamport_rs_conv(model.hidden_size, model.window_size) is state
    assert len(fake_lamport.instances) == 1


@pytest.mark.parametrize("model", MODELS)
def test_mismatched_sizes_raise(fake_lamport, model: ModelShape):
    initialize_lamport_rs_conv(model.hidden_size, model.window_size, 8192)
    other_hidden = (
        INKLING_SMALL.hidden_size if model is INKLING else INKLING.hidden_size
    )
    with pytest.raises(RuntimeError, match="share hidden and window"):
        initialize_lamport_rs_conv(other_hidden, model.window_size, 8192)
    with pytest.raises(RuntimeError, match="share hidden and window"):
        initialize_lamport_rs_conv(model.hidden_size, 3, 8192)
    with pytest.raises(RuntimeError, match="share hidden and window"):
        get_lamport_rs_conv(other_hidden, model.window_size)
    with pytest.raises(RuntimeError, match="share hidden and window"):
        get_lamport_rs_conv(model.hidden_size, 3)


@pytest.mark.parametrize("model", MODELS)
def test_initialize_disabled_by_env(
    fake_lamport, monkeypatch: pytest.MonkeyPatch, model: ModelShape
):
    monkeypatch.setenv("LAMPORT_RS_SCONV", "0")
    initialize_lamport_rs_conv(model.hidden_size, model.window_size, 8192)
    assert get_lamport_rs_conv(model.hidden_size, model.window_size) is None
    assert not fake_lamport.instances


@pytest.mark.parametrize("model", MODELS)
def test_initialize_failure_falls_back_without_retry(
    clean_lamport_state, monkeypatch: pytest.MonkeyPatch, model: ModelShape
):
    calls = 0

    def broken(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("hipIpcGetMemHandle failed")

    monkeypatch.setattr(lamport, "LamportRSConv", broken)
    initialize_lamport_rs_conv(model.hidden_size, model.window_size, 8192)
    assert lamport._STATE_FAILED
    assert get_lamport_rs_conv(model.hidden_size, model.window_size) is None
    # The failure is sticky: later layers/models must not re-enter a
    # half-initialized collective on some ranks.
    initialize_lamport_rs_conv(model.hidden_size, model.window_size, 8192)
    assert calls == 1


def _bare_lamport(model: ModelShape, world_size: int) -> LamportRSConv:
    """A LamportRSConv with only the attributes used before any collective."""
    mm = LamportRSConv.__new__(LamportRSConv)
    mm.hidden_size = model.hidden_size
    mm.window_size = model.window_size
    mm.max_tokens = model.max_tokens
    mm.world_size = world_size
    mm._ipc_local, mm._ipc_peers = [], []
    return mm


@pytest.mark.parametrize("model", MODELS)
def test_usable_bounds(model: ModelShape):
    mm = _bare_lamport(model, world_size=4)
    assert not mm.usable(0)
    assert not mm.usable(-1)
    assert mm.usable(1)
    assert mm.usable(model.max_tokens)
    assert not mm.usable(model.max_tokens + 1)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp", TP_SIZES)
def test_rs_sconv_ag_add_norm_rejects_bad_inputs(model: ModelShape, tp: int):
    """Every argument check fires before any rank publishes to a peer."""
    mm = _bare_lamport(model, world_size=tp)
    case = _Case(query_lens=(1, 2), ctx_lens=(5, 0), block_size=32)
    args = _make_rank_args(model, tp, 0, case, stream=0, seed=0, device="cuda")

    def call(**overrides):
        kw = dict(args.kwargs, **overrides)
        return mm.rs_sconv_ag_add_norm(**kw)

    T, H = args.kwargs["residual"].shape
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    shard = H // tp
    with pytest.raises(ValueError, match="num_tokens"):
        call(
            residual=torch.empty(0, H, **bf16),
            input_tensor=torch.empty(0, H, **bf16),
        )
    with pytest.raises(ValueError, match="num_tokens"):
        n = model.max_tokens + 1
        call(residual=torch.empty(n, H, **bf16), input_tensor=torch.empty(n, H, **bf16))
    with pytest.raises(ValueError, match="residual"):
        call(residual=torch.empty(T, H + 2 * tp, **bf16))
    with pytest.raises(ValueError, match="residual"):
        call(residual=args.kwargs["residual"].float())
    with pytest.raises(ValueError, match="input_tensor"):
        call(input_tensor=torch.empty(H, T, **bf16).t())
    with pytest.raises(ValueError, match="shared_tensor"):
        call(shared_tensor=torch.empty(T, H, dtype=torch.float32, device="cuda"))
    with pytest.raises(ValueError, match="conv_weight"):
        call(conv_weight=torch.empty(shard, 3, **bf16))
    with pytest.raises(ValueError, match="conv_weight"):
        call(conv_weight=torch.empty(4, shard, **bf16).t())
    with pytest.raises(ValueError, match="norm_weight"):
        call(norm_weight=torch.empty(H // 2, **bf16))
    with pytest.raises(ValueError, match="cache"):
        call(cache=args.kwargs["cache"].float())
    with pytest.raises(ValueError, match="ws"):
        call(ws=args.kwargs["ws"] // 2)
    with pytest.raises(ValueError, match="offset"):
        call(off_s=args.kwargs["cache"].shape[3] - args.kwargs["ws"] + 1)
    with pytest.raises(ValueError, match="block_size"):
        call(block_size=model.window_size)


# ---------------------------------------------------------------------------
# Inputs and torch reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Case:
    query_lens: tuple[int, ...]
    ctx_lens: tuple[int, ...]
    block_size: int  # planner-enlarged conv page (cache.shape[2])
    num_pad: int = 0  # trailing PAD tokens (slot_mapping == -1)
    use_norm: bool = True
    use_shared: bool = False


def _cache_layout(model: ModelShape, tp: int, stream: int) -> tuple[int, int, int, int]:
    """(heads per rank, packed head_size, off_s, ws) as in InklingConvState."""
    hidden_per_head = model.hidden_size // model.num_kv_heads
    raw = 2 * model.head_dim + 2 * hidden_per_head
    head_size = 1 << (raw - 1).bit_length()
    off_s = 2 * model.head_dim + stream * hidden_per_head  # 0=_ATTN, 1=_MLP
    return model.num_kv_heads // tp, head_size, off_s, hidden_per_head


@dataclass
class _RankArgs:
    kwargs: dict
    inputs: list[torch.Tensor]  # every rank's input (+ shared) partial
    caches: list[torch.Tensor]  # every rank's pre-forward cache
    conv_weight_full: torch.Tensor


def _make_rank_args(
    model: ModelShape,
    tp: int,
    rank: int,
    case: _Case,
    stream: int,
    seed: int,
    device: str | torch.device,
) -> _RankArgs:
    """Build the op's arguments for `rank`, plus every rank's private inputs.

    All tensors come from seeded generators so each rank can reconstruct its
    peers' inputs and caches to compute the full reference row locally.
    """
    H = model.hidden_size
    bs = case.block_size
    heads, head_size, off_s, ws = _cache_layout(model, tp, stream)
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale=1.0, g=gen):
        return (torch.randn(*shape, generator=g, device=device) * scale).bfloat16()

    # Request metadata (identical on every rank).
    totals = [c + q for c, q in zip(case.ctx_lens, case.query_lens)]
    blocks_per_req = [(t + bs - 1) // bs for t in totals]
    num_blocks = sum(blocks_per_req) + 1
    perm = torch.randperm(num_blocks, generator=gen, device=device).to(torch.int32)
    block_table = torch.zeros(
        len(totals), max(blocks_per_req), dtype=torch.int32, device=device
    )
    positions, seq_idx, slots = [], [], []
    used = 0
    for r, (ctx, q, nb) in enumerate(
        zip(case.ctx_lens, case.query_lens, blocks_per_req)
    ):
        block_table[r, :nb] = perm[used : used + nb]
        used += nb
        pos = torch.arange(ctx, ctx + q, device=device)
        positions.append(pos)
        seq_idx.append(torch.full((q,), r, dtype=torch.int32, device=device))
        slots.append(block_table[r, pos // bs].to(torch.int64) * bs + pos % bs)
    positions.append(torch.zeros(case.num_pad, dtype=torch.int64, device=device))
    seq_idx.append(torch.zeros(case.num_pad, dtype=torch.int32, device=device))
    slots.append(torch.full((case.num_pad,), -1, dtype=torch.int64, device=device))
    positions = torch.cat(positions)
    seq_idx = torch.cat(seq_idx)
    slot_mapping = torch.cat(slots)
    T = positions.numel()

    residual = randn(T, H)
    conv_weight_full = randn(H, model.window_size, scale=0.5)
    norm_weight = randn(H, scale=0.5) + 1.0 if case.use_norm else None
    inputs, caches, own = [], [], {}
    for r in range(tp):
        g = torch.Generator(device=device).manual_seed(seed * 1009 + r + 1)
        x = randn(T, H, g=g)
        shared = randn(T, H, g=g) if case.use_shared else None
        cache = randn(num_blocks, heads, bs, head_size, g=g)
        partial = x if shared is None else (x.float() + shared.float()).bfloat16()
        inputs.append(partial)
        caches.append(cache)
        if r == rank:
            own = dict(input_tensor=x, shared_tensor=shared, cache=cache.clone())

    shard = H // tp
    kwargs = dict(
        input_tensor=own["input_tensor"],
        residual=residual,
        conv_weight=conv_weight_full[rank * shard : (rank + 1) * shard].contiguous(),
        norm_weight=norm_weight,
        eps=model.rms_norm_eps,
        cache=own["cache"],
        positions=positions,
        block_table=block_table,
        seq_idx=seq_idx,
        slot_mapping=slot_mapping,
        off_s=off_s,
        ws=ws,
        block_size=bs,
        shared_tensor=own["shared_tensor"],
    )
    return _RankArgs(kwargs, inputs, caches, conv_weight_full)


def _reference(args: _RankArgs, tp: int, rank: int):
    """Torch model of the four HIP kernels, with their bf16 rounding points.

    Returns (normed, residual_out, this rank's post-forward cache).
    """
    kw = args.kwargs
    residual = kw["residual"]
    T, H = residual.shape
    shard = H // tp
    off_s, ws, bs = kw["off_s"], kw["ws"], kw["block_size"]
    slot, pos = kw["slot_mapping"], kw["positions"]
    valid = slot >= 0
    vslot = slot[valid]

    # 1+2) reduce-scatter in fp32, source order, then bf16.
    acc = torch.zeros(T, H, dtype=torch.float32, device=residual.device)
    for x in args.inputs:
        acc += x.float()
    reduced = acc.bfloat16()

    out = torch.empty_like(residual)
    own_cache = None
    W = args.conv_weight_full.shape[1]
    for r in range(tp):
        cols = slice(r * shard, (r + 1) * shard)
        cache = args.caches[r].clone()
        heads = cache.shape[1]
        # Insert the reduced shard at each valid token's slot.
        cache[vslot // bs, :, vslot % bs, off_s : off_s + ws] = reduced[
            valid, cols
        ].view(-1, heads, ws)
        # 3) 4-tap conv over the cache (incl. this forward's inserts) + skip.
        conv = torch.zeros(T, shard, dtype=torch.float32, device=residual.device)
        cur = torch.zeros_like(conv)
        w = args.conv_weight_full[cols].float()
        for tap in range(W):
            src = pos - (W - 1) + tap
            take = valid & (src >= 0)
            src_c = src.clamp(min=0)
            lblk = (src_c // bs).clamp(max=kw["block_table"].shape[1] - 1)
            blk = kw["block_table"][kw["seq_idx"].long(), lblk].long()
            val = cache[blk, :, src_c % bs, off_s : off_s + ws].reshape(T, shard)
            val = torch.where(take[:, None], val.float(), 0.0)
            conv += val * w[:, tap]
            if tap == W - 1:
                cur = val
        skip = (conv + cur).bfloat16().float()
        out[:, cols] = (residual[:, cols].float() + skip).bfloat16()
        if r == rank:
            own_cache = cache

    normed = None
    if kw["norm_weight"] is not None:
        o = out.float()
        inv_rms = torch.rsqrt(o.square().mean(-1, keepdim=True) + kw["eps"])
        normed = (o * inv_rms * kw["norm_weight"].float()).bfloat16()
    return normed, out, own_cache


def _check_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Kernel vs torch differ only in fp32 FMA contraction / norm reduction
    # order: allow ~1 bf16 ulp.
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=1e-2,
        rtol=1.6e-2,
        msg=lambda m: f"{name}: {m}",
    )


def _check_rearmed(mm: LamportRSConv, label: str) -> None:
    for name, buf in (("buf_in", mm.buf_in), ("buf_out", mm.buf_out)):
        bad = int(buf.view(torch.int16).ne(_SENTINEL).sum())
        assert bad == 0, f"{label}: {name} has {bad} non-sentinel halves after call"


# ---------------------------------------------------------------------------
# Multi-process harness
# ---------------------------------------------------------------------------


def _worker_entry(fn, tp: int, rank: int, port: str, errq, args) -> None:
    try:
        from tests.utils import (
            ensure_model_parallel_initialized,
            init_test_distributed_environment,
        )

        torch.accelerator.set_device_index(rank)
        init_test_distributed_environment(tp, 1, rank, port, local_rank=rank)
        ensure_model_parallel_initialized(tp, 1)
        fn(tp, rank, *args)
        errq.put((rank, None))
    except BaseException:
        errq.put((rank, traceback.format_exc()))
        os._exit(1)


def _run_tp(tp: int, fn, *args) -> None:
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    from tests.utils import _test_rendezvous_path
    from vllm.utils.network_utils import get_open_port

    port = str(get_open_port())
    ctx = mp.get_context("spawn")
    errq = ctx.SimpleQueue()
    procs = [
        ctx.Process(target=_worker_entry, args=(fn, tp, r, port, errq, args))
        for r in range(tp)
    ]
    for p in procs:
        p.start()
    # A rank that fails mid-collective leaves its peers spinning on Lamport
    # sentinels forever, so the harness enforces a deadline and kills them.
    deadline = time.monotonic() + _WORKER_TIMEOUT_S
    timed_out = False
    try:
        while any(p.is_alive() for p in procs):
            if any(p.exitcode not in (None, 0) for p in procs):
                time.sleep(5)  # let the failing rank's traceback land
                break
            if time.monotonic() > deadline:
                timed_out = True
                break
            time.sleep(0.5)
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
            p.join()
        Path(_test_rendezvous_path(port)).unlink(missing_ok=True)

    errors = []
    while not errq.empty():
        rank, tb = errq.get()
        if tb is not None:
            errors.append(f"--- rank {rank} ---\n{tb}")
    if errors:
        pytest.fail("\n".join(sorted(errors)), pytrace=False)
    if timed_out:
        pytest.fail(f"TP={tp} workers timed out (Lamport deadlock?)", pytrace=False)
    codes = [p.exitcode for p in procs]
    assert all(c == 0 for c in codes), f"worker exit codes {codes}"


# ---------------------------------------------------------------------------
# Multi-GPU workers
# ---------------------------------------------------------------------------


def _cases(model: ModelShape) -> list[_Case]:
    # >3 calls so every one of the three Lamport generations is reused.
    return [
        _Case((1,), (0,), block_size=32),  # first token: taps before pos 0
        _Case((1, 1, 1, 1), (5, 130, 2, 77), block_size=32),  # decode batch
        _Case((1, 1), (3, 40), block_size=4, use_shared=True),  # W-sized pages
        _Case((7, 1, 33), (0, 40, 3), block_size=32, use_norm=False),  # mixed
        _Case((2, 1), (9, 1), block_size=32, num_pad=3),  # spec decode + PAD
        _Case((model.max_tokens,), (0,), block_size=32, use_shared=True),  # full
        _Case((1,), (4,), block_size=32),  # decode right after the max call
    ]


def _rs_sconv_worker(tp: int, rank: int, model: ModelShape) -> None:
    from vllm.distributed import get_tp_group

    device = torch.device(f"cuda:{rank}")

    # Constructor argument checks (raise before any IPC allocation).
    with pytest.raises(ValueError, match="window"):
        LamportRSConv(model.hidden_size, 3, model.max_tokens)
    with pytest.raises(ValueError, match="even shard"):
        LamportRSConv(model.hidden_size + 2, model.window_size, model.max_tokens)
    with pytest.raises(ValueError, match="max_tokens"):
        LamportRSConv(model.hidden_size, model.window_size, 0)
    with pytest.raises(ValueError, match="max_tokens"):
        LamportRSConv(model.hidden_size, model.window_size, _MAX_TOKENS + 1)

    mm = LamportRSConv(model.hidden_size, model.window_size, model.max_tokens)
    assert mm.world_size == tp and mm.rank == rank
    assert mm.shard_size == model.hidden_size // tp
    assert mm.buf_in.shape == (3, model.max_tokens, tp, mm.shard_size)
    assert mm.buf_out.shape == (3, model.max_tokens, model.hidden_size)
    _check_rearmed(mm, "init")
    assert mm.usable(model.max_tokens) and not mm.usable(model.max_tokens + 1)

    # Collect mismatches and raise only after every collective, so one rank's
    # assertion cannot strand its peers in a Lamport poll.
    failures: list[str] = []
    for i, case in enumerate(_cases(model)):
        label = f"case {i} {case}"
        args = _make_rank_args(
            model, tp, rank, case, stream=i % 2, seed=1000 + i, device=device
        )
        gen_before = mm.generation
        normed, residual_out = mm.rs_sconv_ag_add_norm(**args.kwargs)
        torch.accelerator.synchronize(device)
        get_tp_group().barrier()
        try:
            assert mm.generation == (gen_before + 1) % 3
            ref_normed, ref_out, ref_cache = _reference(args, tp, rank)
            _check_close(f"{label} residual_out", residual_out, ref_out)
            if case.use_norm:
                _check_close(f"{label} normed", normed, ref_normed)
            else:
                assert normed is None
            # Insert is fp32-reduce in source order then RNE: bit exact.
            assert torch.equal(args.kwargs["cache"], ref_cache), f"{label}: cache"
            _check_rearmed(mm, label)
        except AssertionError as e:
            failures.append(str(e))
        del args
        # Peers must not publish the next call into our buffers until the
        # re-arm check above has finished.
        get_tp_group().barrier()

    # close(): releases IPC mappings and is safe to call again (as __del__ does).
    get_tp_group().barrier()
    mm.close()
    assert mm.buf_in is None and mm.buf_out is None
    assert mm._ipc_local == [] and mm._ipc_peers == []
    mm.close()
    get_tp_group().barrier()
    if failures:
        raise AssertionError(f"rank {rank}:\n" + "\n".join(failures))


def _global_state_worker(tp: int, rank: int, model: ModelShape) -> None:
    from vllm.distributed import get_tp_group

    device = torch.device(f"cuda:{rank}")
    os.environ.pop("LAMPORT_RS_SCONV", None)
    for max_num_batched_tokens in (8192, 4 * _MAX_TOKENS):
        lamport._STATE, lamport._STATE_FAILED = None, False
        assert get_lamport_rs_conv(model.hidden_size, model.window_size) is None
        initialize_lamport_rs_conv(
            model.hidden_size, model.window_size, max_num_batched_tokens
        )
        assert not lamport._STATE_FAILED, "real LamportRSConv init failed"
        mm = get_lamport_rs_conv(model.hidden_size, model.window_size)
        assert isinstance(mm, LamportRSConv)
        assert mm.max_tokens == min(_MAX_TOKENS, max_num_batched_tokens)
        # Re-initialization (e.g. by the MTP draft model) is a no-op.
        initialize_lamport_rs_conv(model.hidden_size, model.window_size, 1)
        assert get_lamport_rs_conv(model.hidden_size, model.window_size) is mm
        with pytest.raises(RuntimeError, match="share hidden and window"):
            get_lamport_rs_conv(model.hidden_size // 2, model.window_size)

        # The shared state is functional.
        case = _Case((1, 1), (6, 0), block_size=32)
        args = _make_rank_args(model, tp, rank, case, stream=0, seed=7, device=device)
        normed, out = mm.rs_sconv_ag_add_norm(**args.kwargs)
        torch.accelerator.synchronize(device)
        get_tp_group().barrier()
        ref_normed, ref_out, _ = _reference(args, tp, rank)
        _check_close("global residual_out", out, ref_out)
        _check_close("global normed", normed, ref_normed)

        mm.close()
        get_tp_group().barrier()
    lamport._STATE, lamport._STATE_FAILED = None, False


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp", TP_SIZES)
def test_rs_sconv_ag_add_norm_matches_reference(model: ModelShape, tp: int):
    _run_tp(tp, _rs_sconv_worker, model)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp", TP_SIZES)
def test_initialize_and_get_lamport_rs_conv_multi_gpu(model: ModelShape, tp: int):
    _run_tp(tp, _global_state_worker, model)
