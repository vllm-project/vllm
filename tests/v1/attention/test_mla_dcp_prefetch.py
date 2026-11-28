# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for generic dense MLA chunked-context DCP KV prefetch (TP+DCP & PCP+DCP)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.ops.cp_common import dcp_scheduled_step
from vllm.v1.attention.ops.kv_ce_overlap import LayerOrder, plan_hash
from vllm.v1.attention.ops.mla_dcp_prefetch import (
    ChunkedDCPKVPrefetch,
    _PendingItem,
    _StepState,
    build_chunked_dcp_plan,
    chunk_local_row_spans,
    classify_kv_cache_kind,
)


def test_classify_kv_cache_kind() -> None:
    cases = [
        ("auto", torch.bfloat16, 576, "unquantized"),
        ("bfloat16", torch.bfloat16, 576, "unquantized"),
        ("float16", torch.float16, 576, "unquantized"),
        ("fp8_ds_mla", torch.bfloat16, 576, "fp8_ds_mla"),
        ("fp8_ds_mla", torch.bfloat16, 512, None),
        ("fp8_ds_mla", torch.float16, 576, None),
        ("fp8", torch.bfloat16, 576, "fp8"),
        ("fp8_e4m3", torch.bfloat16, 512, "fp8"),
        ("fp8", torch.bfloat16, 128, None),
        ("nvfp4_ds_mla", torch.bfloat16, 576, None),
        ("auto", torch.float32, 576, None),
    ]
    for dtype, out_dtype, dim, want in cases:
        assert classify_kv_cache_kind(dtype, out_dtype, dim) == want


@pytest.mark.parametrize(
    "context_lens,dcp,interleave",
    [
        ([128], 4, 1),
        ([101], 4, 1),
        ([103], 4, 4),
        ([128, 77, 1], 4, 1),
        ([65, 130, 19], 8, 2),
        ([3, 5], 8, 1),
    ],
)
def test_chunk_local_row_spans_never_exceeds_historical_rows(
    context_lens: list[int], dcp: int, interleave: int
) -> None:
    """Local reads stop at each rank's exact historical rows."""
    vblk = dcp * interleave
    allranks = get_dcp_local_seq_lens(
        torch.tensor(context_lens, dtype=torch.int32), dcp, None, interleave
    ).tolist()
    padded = [-(-c // vblk) * interleave for c in context_lens]
    for rank in range(dcp):
        aligned, spans = chunk_local_row_spans(
            padded, allranks, [0] * len(padded), rank
        )
        off = 0
        for i, (r_id, dst_off, actual) in enumerate(spans):
            assert (r_id, dst_off, actual) == (i, off, allranks[i][rank])
            assert 0 <= actual <= padded[i]
            off += padded[i]
        assert aligned == all(
            allranks[i][rank] == padded[i] for i in range(len(padded))
        )


def test_chunk_local_row_spans_continuation_chunk() -> None:
    allranks = [[40, 38, 36, 36], [100, 100, 100, 96]]
    local_starts, padded = [36, 64], [4, 36]
    assert chunk_local_row_spans(padded, allranks, local_starts, 0) == (
        True,
        [(0, 0, 4), (1, 4, 36)],
    )
    assert chunk_local_row_spans(padded, allranks, local_starts, 1) == (
        False,
        [(0, 0, 2), (1, 4, 36)],
    )
    assert chunk_local_row_spans(padded, allranks, local_starts, 3) == (
        False,
        [(0, 0, 0), (1, 4, 32)],
    )


def test_build_chunked_dcp_plan_and_hash() -> None:
    p_a = build_chunked_dcp_plan(
        num_layers=47,
        world_size=4,
        cache_kind="unquantized",
        chunk_toks=[16384, 8192],
        max_local_tokens=16384,
    )
    p_b = build_chunked_dcp_plan(
        num_layers=47,
        world_size=4,
        cache_kind="unquantized",
        chunk_toks=[16384, 8192],
        max_local_tokens=16384,
    )
    p_t = build_chunked_dcp_plan(
        num_layers=47,
        world_size=4,
        cache_kind="unquantized",
        chunk_toks=[16384, 4096],
        max_local_tokens=16384,
    )
    p_k = build_chunked_dcp_plan(
        num_layers=47,
        world_size=4,
        cache_kind="fp8_ds_mla",
        chunk_toks=[16384, 8192],
        max_local_tokens=16384,
    )
    assert p_a is not None and p_b is not None and p_t is not None and p_k is not None
    assert plan_hash(p_a.digest()) == plan_hash(p_b.digest())
    assert plan_hash(p_a.digest()) != plan_hash(p_t.digest())
    assert plan_hash(p_a.digest()) != plan_hash(p_k.digest())
    for bad in [
        {"num_layers": 0, "world_size": 4, "chunk_toks": [1024]},
        {"num_layers": 4, "world_size": 1, "chunk_toks": [1024]},
        {"num_layers": 4, "world_size": 4, "chunk_toks": [16385]},
        {"num_layers": 4, "world_size": 4, "chunk_toks": []},
    ]:
        assert (
            build_chunked_dcp_plan(
                cache_kind="unquantized",
                max_local_tokens=16384,
                **bad,  # type: ignore[arg-type]
            )
            is None
        )


def _make_fake_prefetch(
    world_size: int = 4,
    max_gathered_tokens: int = 256,
    row_dim: int = 576,
    cache_kind: str = "unquantized",
) -> tuple[ChunkedDCPKVPrefetch, list[Any]]:
    log: list[Any] = []
    pf = object.__new__(ChunkedDCPKVPrefetch)
    pf.world_size, pf.rank, pf.device = world_size, 0, torch.device("cpu")
    pf.max_gathered_tokens = max_gathered_tokens
    pf.max_local_tokens = max_gathered_tokens // world_size
    pf.row_dim, pf.out_dtype = row_dim, torch.bfloat16
    pf.kv_cache_dtype = "auto" if cache_kind == "unquantized" else cache_kind
    pf.cache_kind = cache_kind
    pf.wire_dtype = torch.bfloat16 if cache_kind == "unquantized" else torch.uint8
    pf.row_bytes = row_dim * 2
    pf.gathered = torch.zeros((2 * max_gathered_tokens, row_dim), dtype=torch.bfloat16)
    pf.gathered_ptr = pf.gathered.data_ptr()
    pf.head_done_owed, pf.order, pf.orders = False, LayerOrder(), {}
    pf.plan = pf.step = pf.pending = pf.position = pf.forward = None
    pf.layers, pf.plan_count = {}, 0
    pf.plan_writers = cast(
        Any,
        [
            SimpleNamespace(
                write=lambda s, st, v: log.append(("plan_write", st, list(v)))
            )
            for _ in range(4)
        ],
    )
    pf.handle = cast(
        Any,
        SimpleNamespace(
            barrier=lambda *, channel, timeout_ms: log.append(("barrier", channel))
        ),
    )
    pf.side = cast(Any, SimpleNamespace(cuda_stream=99))
    pf.side_handle = 99
    pf.wire_peers = [p for p in range(world_size) if p != pf.rank]
    pf.pull_order = [(pf.rank + s) % world_size for s in range(1, world_size)]
    pf.wire_index = {p: i for i, p in enumerate(pf.wire_peers)}
    pf.ready = True

    def fake_launch(layer_pos: int, chunk_idx: int) -> None:
        assert pf.step is not None
        name = pf.order.names[layer_pos]
        toks = int(pf.step.chunks[chunk_idx].num_local_context_tokens)
        gidx = layer_pos * len(pf.step.chunks) + chunk_idx
        if pf.head_done_owed:
            log.append(("barrier_side", "HEAD_DONE"))
            pf.head_done_owed = False
        log.append(("launch", name, layer_pos, chunk_idx, gidx % 2, toks))
        pf.head_done_owed = True
        pf.pending = _PendingItem(name, layer_pos, chunk_idx, gidx, toks)

    vars(pf)["_wait"] = lambda w, s: log.append(
        ("wait", getattr(w, "cuda_stream", 0), s.cuda_stream)
    )
    vars(pf)["_launch_item"] = fake_launch
    return pf, log


class _Fwd:
    def __init__(self, tag: str = "") -> None:
        self.tag = tag


def _fake_meta(chunk_toks: list[int]) -> Any:
    chunks = [
        SimpleNamespace(num_local_context_tokens=t, request_slice=slice(0, 1))
        for t in chunk_toks
    ]
    return SimpleNamespace(
        prefill=SimpleNamespace(
            chunked_context=SimpleNamespace(chunks=chunks),
            block_table=torch.zeros((1, 4), dtype=torch.int32),
        )
    )


def test_state_machine_learns_order_and_overlaps_all_layers_and_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.attention.ops import kv_ce_overlap as kv_mod
    from vllm.v1.attention.ops import mla_dcp_prefetch as mod

    stream = SimpleNamespace(cuda_stream=1)
    monkeypatch.setattr(mod, "current_stream", lambda: stream)
    monkeypatch.setattr(kv_mod, "current_stream", lambda: stream)
    pf, log = _make_fake_prefetch(world_size=4, max_gathered_tokens=256)
    layers = ["L0", "L1", "L2"]
    caches = {
        n: (torch.zeros((2, 16, 576), dtype=torch.bfloat16), torch.ones(1))
        for n in layers
    }
    meta = _fake_meta([32, 16])

    # Pass 0 learns order (serial); Pass 1 overlaps all 3x2 items.
    fwd0 = _Fwd("fwd0")
    for n in layers:
        pf.begin_layer(n, *caches[n], fwd0, lambda _n: meta, scheduled=True)
        pf.start_first()
        for ci in range(2):
            assert pf.join(n, ci, scheduled=True) is None
            pf.start_next(n, ci)
    assert pf.order.names == layers and not any(ev[0] == "launch" for ev in log)

    log.clear()
    fwd1, ptrs = _Fwd("fwd1"), []
    for n in layers:
        pf.begin_layer(n, *caches[n], fwd1, lambda _n: meta, scheduled=True)
        pf.start_first()
        for ci in range(2):
            reg = pf.join(n, ci, scheduled=True)
            assert reg is not None and reg.shape == (4 * [32, 16][ci], 576)
            ptrs.append(reg.data_ptr())
            pf.start_next(n, ci)
    assert [ev for ev in log if ev[0] == "launch"] == [
        ("launch", "L0", 0, 0, 0, 32),
        ("launch", "L0", 0, 1, 1, 16),
        ("launch", "L1", 1, 0, 0, 32),
        ("launch", "L1", 1, 1, 1, 16),
        ("launch", "L2", 2, 0, 0, 32),
        ("launch", "L2", 2, 1, 1, 16),
    ]
    assert all(ptrs[i] != ptrs[i + 1] for i in range(len(ptrs) - 1))
    assert pf.step is None and pf.head_done_owed is True

    # Unscheduled capture pays no barrier; scheduled serial fallback settles HEAD_DONE.
    log.clear()
    pf.begin_layer("L0", *caches["L0"], _Fwd("cap"), lambda _n: meta, scheduled=False)
    pf.start_first()
    assert pf.join("L0", 0, scheduled=False) is None and pf.head_done_owed and log == []

    meta_over = _fake_meta([9999])
    pf.begin_layer(
        "L0", *caches["L0"], _Fwd("ser"), lambda _n: meta_over, scheduled=True
    )
    pf.start_first()
    assert pf.join("L0", 0, scheduled=True) is None and not pf.head_done_owed
    assert ("barrier", 2) in log


def test_wire_serves_validates_cache_layout() -> None:
    pf_bf16, _ = _make_fake_prefetch(cache_kind="unquantized")
    assert pf_bf16._wire_serves(torch.empty((2, 16, 576), dtype=torch.bfloat16))
    assert not pf_bf16._wire_serves(torch.empty((2, 16, 576), dtype=torch.float16))
    assert not pf_bf16._wire_serves(torch.empty((2, 16, 512), dtype=torch.bfloat16))

    pf_ds, _ = _make_fake_prefetch(cache_kind="fp8_ds_mla")
    assert pf_ds._wire_serves(torch.empty((2, 64, 656), dtype=torch.uint8))
    assert not pf_ds._wire_serves(torch.empty((2, 64, 576), dtype=torch.uint8))

    pf_fp8, _ = _make_fake_prefetch(cache_kind="fp8")
    pf_fp8.wire_dtype = torch.float8_e4m3fn
    assert pf_fp8._wire_serves(
        torch.empty((2, 16, 576), dtype=torch.uint8).view(torch.float8_e4m3fn)
    )
    assert not pf_fp8._wire_serves(torch.empty((2, 16, 576), dtype=torch.bfloat16))


@pytest.mark.parametrize("cache_kind", ["unquantized", "fp8_ds_mla", "fp8"])
@pytest.mark.parametrize("toks, expect_pipe", [(1024, False), (8192, True)])
def test_chunk_copies_staggered_pull_order_and_pipeline_mode(
    cache_kind: str, toks: int, expect_pipe: bool
) -> None:
    world, rank = 4, 2
    pf, _ = _make_fake_prefetch(world, world * toks, cache_kind=cache_kind)
    pf.rank = rank
    pf.wire_peers = [p for p in range(world) if p != rank]
    pf.pull_order = [(rank + s) % world for s in range(1, world)]
    pf.wire_index = {p: i for i, p in enumerate(pf.wire_peers)}
    pf.wire_rows = toks
    pf.wire_row_bytes = 1152 if cache_kind == "unquantized" else 656
    pf.own_wire_ptr, pf.stage_ptr = 0x9000_0000, 0x10_0000_0000
    pf.remote_ptrs = [0x20_0000_0000 + p * 0x1_0000_0000 for p in range(world)]
    plan = build_chunked_dcp_plan(
        num_layers=1,
        world_size=world,
        cache_kind=cache_kind,
        chunk_toks=[toks],
        max_local_tokens=toks,
    )
    assert plan is not None
    step = _StepState(plan, [], [], (0x40_0000_0000, 0x50_0000_0000), 0, set())
    cp = pf._chunk_copies(step, 0, toks)
    want_order = [3, 0, 1]
    if cache_kind == "unquantized":
        assert not cp.pipelined and cp.own_copy is not None
        assert cp.own_copy.copies == [
            (
                0x40_0000_0000 + rank * toks * pf.row_bytes,
                pf.own_wire_ptr,
                toks * pf.row_bytes,
            )
        ]
        peers = [
            (s - 0x20_0000_0000) // 0x1_0000_0000
            for cl in cp.peer_chunks
            for _, s, _ in cl.copies
        ]
        assert peers == want_order
    else:
        assert cp.pipelined is expect_pipe and cp.own_copy is None
        peers = [
            (cl.copies[0][1] - 0x20_0000_0000) // 0x1_0000_0000 for cl in cp.peer_copies
        ]
        assert peers == want_order


def test_mla_dcp_manager_init_kv_gather_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.attention.ops import dcp as dcp_mod
    from vllm.v1.attention.ops import mla_dcp_prefetch as pf_mod

    calls: list[dict[str, Any]] = []
    sentinel = object()

    def fake_get(**kw: Any) -> object:
        calls.append(kw)
        return sentinel

    monkeypatch.setattr(pf_mod, "get_chunked_dcp_kv_prefetch", fake_get)
    monkeypatch.setattr(
        dcp_mod, "get_direct_dcp_kv_gather_workspace", lambda *a, **kw: None
    )

    mgr = object.__new__(dcp_mod.MLADCPManager)
    mgr.group = SimpleNamespace(world_size=4, device_group=object())  # type: ignore[assignment]
    mgr.num_ubatches, mgr.dcp_kv_prefetch, mgr.kv_cache_dtype, mgr.kv_prefetch = (
        1,
        True,
        "auto",
        None,
    )
    ws = torch.empty((320, 576), dtype=torch.bfloat16)

    mgr.init_kv_gather(ws, 256)
    assert mgr.kv_prefetch is None and calls == []
    mgr.init_kv_gather(ws, 256, enable_kv_prefetch=True)
    assert mgr.kv_prefetch is sentinel and len(calls) == 1
    assert (
        calls[0]["max_gathered_tokens"],
        calls[0]["row_dim"],
        calls[0]["kv_cache_dtype"],
    ) == (256, 576, "auto")


def test_dcp_scheduled_step_tp_dcp_and_pcp_dcp(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.attention.ops import cp_common

    capturing, pcp_mgr = False, None
    monkeypatch.setattr(cp_common, "capturing_run", lambda: capturing)
    monkeypatch.setattr(cp_common, "_active_pcp_manager", lambda: pcp_mgr)
    assert dcp_scheduled_step() is True
    capturing = True
    assert dcp_scheduled_step() is False
    capturing, pcp_mgr = False, SimpleNamespace(scheduled_batch=False)
    assert dcp_scheduled_step() is False
    pcp_mgr = SimpleNamespace(scheduled_batch=True)
    assert dcp_scheduled_step() is True
    capturing = True
    assert dcp_scheduled_step() is False


def test_alternating_entry_layers_preserve_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alternating target ([L0, L1]) and draft ([MTP0]) keep separate LayerOrders."""
    from vllm.v1.attention.ops import kv_ce_overlap as kv_mod
    from vllm.v1.attention.ops import mla_dcp_prefetch as mod

    stream = SimpleNamespace(cuda_stream=1)
    monkeypatch.setattr(mod, "current_stream", lambda: stream)
    monkeypatch.setattr(kv_mod, "current_stream", lambda: stream)
    pf, _ = _make_fake_prefetch(world_size=4, max_gathered_tokens=256)
    kv, scale, meta = (
        torch.zeros((2, 16, 576), dtype=torch.bfloat16),
        torch.ones(1),
        _fake_meta([32]),
    )
    for step in range(3):
        for tag, seq in (("target", ("L0", "L1")), ("draft", ("MTP0",))):
            fwd = _Fwd(f"{tag}-{step}")
            for name in seq:
                pf.begin_layer(name, kv, scale, fwd, lambda _n: meta, scheduled=True)
                pf.start_first()
                got = pf.join(name, 0, scheduled=True)
                pf.start_next(name, 0)
                assert (got is None) if step == 0 else (got is not None)


def test_mla_common_base_impl_forward_mha_prefetch_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.attention import mla_attention as mla_mod

    events: list[tuple[Any, ...]] = []
    dcp, lora, rope, nope, vdim, heads, toks = 2, 8, 4, 4, 8, 2, 4
    rdim = lora + rope
    prefetched = torch.arange(dcp * toks * rdim, dtype=torch.bfloat16).view(
        dcp * toks, rdim
    )

    def _rec(ev: tuple[Any, ...], ret: Any) -> Any:
        events.append(ev)
        return ret

    stub_pf = SimpleNamespace(
        begin_layer=lambda n, _kv, _s, _f, _m, scheduled=True: events.append(
            ("begin_layer", n, scheduled)
        ),
        start_first=lambda: events.append(("start_first",)),
        join=lambda n, ci, scheduled=True: _rec(("join", n, ci, scheduled), prefetched),
        start_next=lambda n, ci: events.append(("start_next", n, ci)),
    )
    stub_be = SimpleNamespace(
        run_prefill_new_tokens=lambda **kw: _rec(
            ("run_prefill_new_tokens",),
            (
                torch.zeros((kw["q"].shape[0], heads, vdim), dtype=torch.bfloat16),
                torch.zeros((kw["q"].shape[0], heads)),
            ),
        ),
        run_prefill_context_chunk=lambda **kw: _rec(
            ("run_prefill_context_chunk", kw["k"].shape[0]),
            (
                torch.ones((kw["q"].shape[0], heads, vdim), dtype=torch.bfloat16),
                torch.zeros((kw["q"].shape[0], heads)),
            ),
        ),
    )

    class _ConcreteMLA(mla_mod.MLACommonBaseImpl[Any]):
        def forward_mqa(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    impl = object.__new__(_ConcreteMLA)
    impl._layer_name = "model.layers.0.self_attn.attn"
    impl.dcp_world_size, impl.kv_cache_dtype = dcp, "auto"
    impl.kv_lora_rank, impl.qk_rope_head_dim, impl.qk_nope_head_dim = lora, rope, nope
    impl.v_head_dim, impl.num_heads, impl._use_flashinfer_concat_mla_k = (
        vdim,
        heads,
        False,
    )
    impl.kv_b_proj = cast(
        Any,
        lambda x: (
            torch.zeros((x.shape[0], heads * (nope + vdim)), dtype=torch.bfloat16),
            None,
        ),
    )

    chunk = SimpleNamespace(
        padded_local_seq_lens=[toks],
        local_context_lens_allranks=[[toks, toks]],
        padded_local_cu_seq_lens=torch.tensor([0, toks], dtype=torch.int32),
        padded_local_token_to_seq=torch.zeros(toks, dtype=torch.int32),
        local_starts=[0],
        num_local_context_tokens=toks,
        num_context_tokens=dcp * toks,
        max_seq_len=dcp * toks,
        token_slice=slice(0, 2),
        request_slice=slice(0, 1),
        starts=torch.tensor([0], dtype=torch.int32),
        num_requests=1,
    )
    cc = SimpleNamespace(
        chunks=[chunk],
        workspace=torch.empty((3 * toks, rdim), dtype=torch.bfloat16),
        dcp_manager=SimpleNamespace(kv_prefetch=stub_pf),
        empty_token_slices=[],
    )
    meta = SimpleNamespace(
        prefill=SimpleNamespace(
            prefill_backend=stub_be,
            chunked_context=cc,
            q_data_type=torch.bfloat16,
            block_table=torch.zeros((1, 4), dtype=torch.int32),
        )
    )
    monkeypatch.setattr(
        mla_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={impl._layer_name: meta}),
    )
    monkeypatch.setattr(mla_mod, "dcp_scheduled_step", lambda: True)
    monkeypatch.setattr(mla_mod, "_get_kv_b_proj_input_dtype", lambda *a: None)
    monkeypatch.setattr(
        mla_mod,
        "merge_attn_states",
        lambda output, prefix_output, **kw: output.copy_(prefix_output),
    )

    output = torch.zeros((2, heads * vdim), dtype=torch.bfloat16)
    impl.forward_mha(
        torch.zeros((2, heads, nope + rope), dtype=torch.bfloat16),
        torch.zeros((2, lora), dtype=torch.bfloat16),
        torch.zeros((2, 1, rope), dtype=torch.bfloat16),
        torch.zeros((2, 16, rdim), dtype=torch.bfloat16),
        cast(Any, meta),
        torch.ones(1),
        output,
    )
    assert events == [
        ("begin_layer", "model.layers.0.self_attn.attn", True),
        ("start_first",),
        ("run_prefill_new_tokens",),
        ("join", "model.layers.0.self_attn.attn", 0, True),
        ("start_next", "model.layers.0.self_attn.attn", 0),
        ("run_prefill_context_chunk", dcp * toks),
    ]
    assert torch.all(output == 1.0)


_MULTIGPU_WORKER = r"""
import os, torch, torch.distributed as dist
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    get_dcp_group, init_distributed_environment, initialize_model_parallel,
)
from vllm.model_executor.layers.attention.mla_attention import (
    build_mla_chunked_context_metadata, reorg_kvcache,
)
from vllm.v1.attention.ops.mla_dcp_prefetch import ChunkedDCPKVPrefetch

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
kind = os.environ["TEST_CACHE_KIND"]
torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")
with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(
        world, rank, f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}", rank
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world, decode_context_model_parallel_size=world
    )
group = get_dcp_group()
row_dim, block_size, num_blocks, max_gathered, num_layers = 576, 64, 32, 256, 3
kv_dtype = "auto" if kind == "unquantized" else kind
pf = ChunkedDCPKVPrefetch(
    group, device, max_gathered, row_dim, torch.bfloat16, kv_dtype
)
assert pf.ready

g = torch.Generator(device=device).manual_seed(100 + rank * 17)
caches, scales = [], []
for _ in range(num_layers):
    if kind == "unquantized":
        c = torch.randn(
            (num_blocks, block_size, row_dim),
            generator=g, dtype=torch.bfloat16, device=device,
        )
    elif kind == "fp8_ds_mla":
        raw = torch.zeros(
            (num_blocks, block_size, 656), dtype=torch.uint8, device=device
        )
        raw[:, :, :512] = (
            (torch.randn((num_blocks, block_size, 512), generator=g, device=device)
             * 40).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        raw[:, :, 512:528] = (
            (2.0 ** torch.randint(
                -3, 4, (num_blocks, block_size, 4), generator=g, device=device
            ).float()).view(torch.uint8).view(num_blocks, block_size, 16)
        )
        raw[:, :, 528:] = (
            torch.randn(
                (num_blocks, block_size, 64),
                generator=g, dtype=torch.bfloat16, device=device,
            ).view(torch.uint8).view(num_blocks, block_size, 128)
        )
        c = raw
    else:
        c = (
            (torch.randn((num_blocks, block_size, row_dim), generator=g, device=device)
             * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
        )
    caches.append(c)
    scales.append(torch.tensor([0.75], dtype=torch.float32, device=device))

ws = torch.empty(
    (max_gathered + max_gathered // world, row_dim), dtype=torch.bfloat16, device=device
)
class _F: pass
cases = [
    ([128], [0, 16]),
    ([640], [0, 32]),
    ([645], [0, 16]),
    ([1, 261, 130, 513], [0, 8, 16, 24, 32]),
]
for si, (clens, qs) in enumerate(cases):
    bt = (
        torch.arange(num_blocks, dtype=torch.int32, device=device)
        .view(1, -1).expand(len(clens), -1).contiguous()
    )
    cc = build_mla_chunked_context_metadata(
        context_lens_cpu=torch.tensor(clens, dtype=torch.int32),
        prefill_query_start_loc_cpu=torch.tensor(qs, dtype=torch.int32),
        chunked_prefill_workspace=ws,
        chunked_prefill_workspace_size=max_gathered,
        block_size=block_size,
        align_chunk_to_block=True,
        device=device,
        dcp_world_size=world,
        dcp_local_block_size=1,
        dcp_virtual_block_size=world,
        dcp_manager=None,
    )
    pf_meta = type("P", (), {"chunked_context": cc, "block_table": bt})()
    meta = type("M", (), {"prefill": pf_meta})()
    for rep in range(2 if si == 0 else 1):
        fwd = _F()
        for li in range(num_layers):
            name = f"layer.{li}"
            pf.begin_layer(
                name, caches[li], scales[li], fwd, lambda _n: meta, scheduled=True
            )
            pf.start_first()
            for ci, ch in enumerate(cc.chunks):
                toks = ch.num_local_context_tokens
                got = pf.join(name, ci, scheduled=True)
                pf.start_next(name, ci)
                if got is None:
                    assert si == 0 and rep == 0
                    continue
                reorg_args = (
                    ch.padded_local_seq_lens, ch.local_context_lens_allranks,
                    ch.local_starts, ch.num_context_tokens, ch.max_seq_len, toks,
                )
                gc, gp = reorg_kvcache(
                    *got.unsqueeze(1).split([512, 64], dim=-1), *reorg_args
                )
                cbt = bt[ch.request_slice]
                if kind == "fp8_ds_mla":
                    ops.cp_gather_and_upconvert_fp8_kv_cache(
                        caches[li], ws[:toks], cbt,
                        ch.padded_local_cu_seq_lens, ch.num_requests, ch.starts,
                    )
                elif kind == "fp8":
                    ops.gather_and_maybe_dequant_cache(
                        caches[li], ws, cbt, ch.padded_local_cu_seq_lens,
                        ch.padded_local_token_to_seq, toks, kv_dtype,
                        scales[li], ch.starts,
                    )
                else:
                    ops.cp_gather_cache(
                        caches[li], ws[:toks], cbt,
                        ch.padded_local_cu_seq_lens, ch.num_requests, ch.starts,
                    )
                ref = torch.empty(
                    (world * toks, row_dim), dtype=torch.bfloat16, device=device
                )
                dist.all_gather_into_tensor(ref, ws[:toks], group=group.device_group)
                rc, rp = reorg_kvcache(
                    *ref.unsqueeze(1).split([512, 64], dim=-1), *reorg_args
                )
                assert torch.equal(gc, rc) and torch.equal(gp, rp)
torch.cuda.synchronize()
dist.destroy_process_group()
"""


@pytest.mark.skipif(
    not current_platform.is_cuda() or torch.cuda.device_count() < 2,
    reason="Requires >=2 CUDA devices for symmetric-memory DCP prefetch test",
)
@pytest.mark.parametrize("cache_kind", ["unquantized", "fp8_ds_mla", "fp8"])
def test_chunked_dcp_prefetch_bitwise_multi_rank(
    tmp_path: Any, cache_kind: str
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    worker = tmp_path / "worker.py"
    worker.write_text(_MULTIGPU_WORKER)
    env = {**os.environ, "TEST_CACHE_KIND": cache_kind, "MASTER_PORT": str(port)}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--master_port",
            str(port),
            str(worker),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
