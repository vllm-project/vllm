# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The closed-form bs=1 MLA planner (VLLM_K31_BEAT_DRAFTMETA,
``levers.mla_bs1_meta``) against aiter's serial planner: identical schedules,
identical attention output, and a graph-captured plan that follows the KV length
on replay. Plus the CPU-only parts of VLLM_K31_BEAT_DRAFTMETA: when the wrapped
planner takes the call (``levers.mla``), and which attention groups the target
step drops (``KimiK3ModelState``)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

# Only the planner tests need aiter; the gating and group-dropping tests are
# pure Python and run anywhere.
rocm_only = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="aiter MLA planner is ROCm-only"
)

H = 16  # draft heads per rank, asm-padded (DSpark 96 heads / TP8 -> 12 -> 16)
MAX_QLEN = 8


def _bufs(dtype, max_num_seqs=1):
    from aiter import get_mla_metadata_info_v1

    info = get_mla_metadata_info_v1(
        max_num_seqs, MAX_QLEN, H, dtype, dtype, is_sparse=False, fast_mode=True
    )
    return [
        torch.zeros(s if isinstance(s, tuple) else (s,), dtype=t, device="cuda")
        for s, t in info
    ]


def _aiter_plan(bufs, kv_indptr, qlen, causal, dtype):
    from aiter import get_mla_metadata_v1

    wmd, wind, wis, rind, rfm, rpm = bufs
    get_mla_metadata_v1(
        torch.tensor([0, qlen], dtype=torch.int32, device="cuda"),
        kv_indptr,
        torch.ones(1, dtype=torch.int32, device="cuda"),
        H,
        1,
        causal,
        wmd,
        wis,
        wind,
        rind,
        rfm,
        rpm,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=qlen,
        uni_seqlen_qo=qlen,
        fast_mode=True,
        dtype_q=dtype,
        dtype_kv=dtype,
    )


@rocm_only
@pytest.mark.parametrize("qlen", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("causal", [True, False])
def test_bs1_planner_matches_aiter(qlen, causal):
    from aiter import dtypes

    from vllm.models.kimi_k3.amd.specialized_decode.levers.mla_bs1_meta import (
        AiterMlaBs1Planner,
    )

    fp8 = dtypes.fp8
    planner = AiterMlaBs1Planner(*_bufs(fp8))
    # Dense below 20k (the whole unsaturated split law and every tail-fold
    # boundary at that scale), then strided out to 1M tokens.
    kv_lens = list(range(0, 20000)) + list(range(20000, 1_050_000, 1013))
    assert planner.verify(qlen, H, causal, fp8, fp8, kv_lens=kv_lens)
    assert planner.is_verified(qlen, H, causal, fp8, fp8)


@rocm_only
@pytest.mark.parametrize("max_num_seqs", [1, 8, 64, 256, 1024])
@pytest.mark.parametrize("qlen", [7, 8])
@pytest.mark.parametrize("causal", [True, False])
def test_bs1_planner_verifies_on_builder_sized_buffers(max_num_seqs, qlen, causal):
    """The builder sizes the schedule for max_num_seqs requests; aiter fills
    reduce_indptr to its full length (past 512 entries from ~64 requests), so
    the closed-form plan has to as well or verify turns the planner off."""
    from aiter import dtypes

    from vllm.models.kimi_k3.amd.specialized_decode.levers.mla_bs1_meta import (
        AiterMlaBs1Planner,
    )

    fp8 = dtypes.fp8
    planner = AiterMlaBs1Planner(*_bufs(fp8, max_num_seqs))
    assert planner.verify(qlen, H, causal, fp8, fp8)


@rocm_only
@pytest.mark.parametrize("kv_len", [7, 23, 1000, 4000, 65537, 360000])
def test_bs1_planner_attention_output_identical(kv_len):
    from aiter import dtypes
    from aiter.mla import mla_decode_fwd

    from vllm.models.kimi_k3.amd.specialized_decode.levers.mla_bs1_meta import (
        AiterMlaBs1Planner,
    )

    torch.manual_seed(0)
    fp8, qlen = dtypes.fp8, 7
    kv = (torch.randn(kv_len + 64, 1, 1, 576, device="cuda") * 0.5).to(fp8)
    q = (torch.randn(qlen, H, 576, device="cuda") * 0.5).to(fp8)
    qo_indptr = torch.tensor([0, qlen], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(kv_len + 64, dtype=torch.int32, device="cuda")
    last = torch.ones(1, dtype=torch.int32, device="cuda")
    one = torch.ones(1, dtype=torch.float32, device="cuda")

    def run(bufs):
        wmd, wind, wis, rind, rfm, rpm = bufs
        o = torch.zeros(qlen, H, 512, dtype=torch.bfloat16, device="cuda")
        mla_decode_fwd(
            q, kv, o, qo_indptr, kv_indptr, kv_indices, last, qlen,
            sm_scale=576**-0.5, work_meta_data=wmd, work_indptr=wind,
            work_info_set=wis, reduce_indptr=rind, reduce_final_map=rfm,
            reduce_partial_map=rpm, q_scale=one, kv_scale=one, causal=True,
        )  # fmt: skip
        return o

    ref_bufs = _bufs(fp8)
    _aiter_plan(ref_bufs, kv_indptr, qlen, True, fp8)
    ref = run(ref_bufs)

    planner = AiterMlaBs1Planner(*_bufs(fp8))
    planner.plan(kv_indptr, qlen)
    out = run(
        [
            planner.work_meta_data,
            planner.work_indptr,
            planner.work_info_set,
            planner.reduce_indptr,
            planner.reduce_final_map,
            planner.reduce_partial_map,
        ]
    )
    torch.accelerator.synchronize()
    assert torch.equal(out, ref)


@rocm_only
def test_bs1_planner_graph_replay_tracks_kv_len():
    from aiter import dtypes

    from vllm.models.kimi_k3.amd.specialized_decode.levers.mla_bs1_meta import (
        AiterMlaBs1Planner,
    )

    fp8, qlen = dtypes.fp8, 7
    planner = AiterMlaBs1Planner(*_bufs(fp8))
    kv_indptr = torch.tensor([0, 100], dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        planner.plan(kv_indptr, qlen)
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            planner.plan(kv_indptr, qlen)
    ref_bufs = _bufs(fp8)
    for kv_len in [100, 5000, 131072, 360000, 777777]:
        kv_indptr[1] = kv_len
        graph.replay()
        _aiter_plan(ref_bufs, kv_indptr, qlen, True, fp8)
        torch.accelerator.synchronize()
        _, wind, wis, rind, rfm, rpm = ref_bufs
        nw = int(wind[-1])
        assert torch.equal(planner.work_indptr, wind)
        assert torch.equal(planner.work_info_set[:nw], wis[:nw])
        assert torch.equal(planner.reduce_indptr, rind)
        if nw > 1:
            assert torch.equal(planner.reduce_final_map, rfm)
            assert torch.equal(planner.reduce_partial_map[:nw], rpm[:nw])
        ptrs = planner.work_meta_data.view(torch.int64).tolist()
        assert ptrs == [
            planner.work_indptr.data_ptr(),
            planner.work_info_set.data_ptr(),
        ]


def _group(names, mamba=False):
    from vllm.v1.kv_cache_interface import MambaSpec

    kv_spec = object.__new__(MambaSpec) if mamba else object()
    return SimpleNamespace(layer_names=names, kv_cache_spec=kv_spec)


def test_drop_draft_only_attn_groups(monkeypatch):
    from vllm.models.kimi_k3.amd.specialized_decode import levers
    from vllm.models.kimi_k3.amd.specialized_decode.levers import runner
    from vllm.models.kimi_k3.amd.specialized_decode.model_state import (
        KimiK3ModelState,
        drop_draft_only_attn_groups,
    )

    draft_layers = {"draft.0.attn", "draft.1.attn"}
    target = _group(["model.0.attn", "model.3.attn"])
    draft = _group(["draft.0.attn", "draft.1.attn"])
    mamba = _group(["model.1.kda"], mamba=True)
    mixed = _group(["model.5.attn", "draft.1.attn"])
    groups = [[target], [draft], [mamba], [mixed, draft]]
    dropped = [[target], [], [mamba], [mixed]]

    # Positions kept (block tables index by group id); only draft-only gone.
    assert drop_draft_only_attn_groups(groups, draft_layers) == dropped

    state = object.__new__(KimiK3ModelState)
    state._draft_layers = set()
    state._target_groups = None
    state.note_draft_attn_layers(draft_layers)
    # Gate off (no latency_specialized): every group.
    with runner.target_step():
        assert state.target_attn_groups(groups) is groups
    monkeypatch.setattr(levers, "_active", True)
    # Outside the runner's target step (CUDA-graph capture, the drafter's own
    # calls): every group.
    assert state.target_attn_groups(groups) is groups
    with runner.target_step():
        assert state.target_attn_groups(groups) == dropped
        # Filtered once.
        assert state.target_attn_groups(groups) is state.target_attn_groups(groups)
        # The profile run's Mamba-less copy: filtered too.
        profile = [[g for g in gs if g is not mamba] for gs in groups]
        assert state.target_attn_groups(profile) == [[target], [], [], [mixed]]
        monkeypatch.setenv("VLLM_K31_BEAT_DRAFTMETA", "0")
        assert state.target_attn_groups(groups) is groups
        monkeypatch.setenv("VLLM_K31_BEAT_DRAFTMETA", "1")
        # A drafter without draft-only layers: every group.
        state.note_draft_attn_layers(set())
        assert state.target_attn_groups(groups) is groups


class _MockPlanner:
    def __init__(self, verify_ok=True):
        self.verify_ok = verify_ok
        self.verified: set[tuple] = set()
        self.verify_calls: list[tuple] = []

    def is_verified(self, *key):
        return key in self.verified

    def verify(self, *key):
        self.verify_calls.append(key)
        if self.verify_ok:
            self.verified.add(key)
        return self.verify_ok


def test_use_bs1_planner_gating(monkeypatch):
    from vllm.models.kimi_k3.amd.specialized_decode.levers.mla import use_bs1_planner

    capturing = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    q_dtype, kv_dtype = torch.float8_e4m3fn, torch.float8_e4m3fn
    planner = _MockPlanner()
    bufs = [torch.empty(1) for _ in range(6)]
    builder = SimpleNamespace(
        _sd_bs1_planner=planner,
        _mla_work_meta_data=bufs[0],
        _mla_work_info_set=bufs[1],
        _mla_work_indptr=bufs[2],
        _mla_reduce_indptr=bufs[3],
        _mla_reduce_final_map=bufs[4],
        _mla_reduce_partial_map=bufs[5],
    )
    ok_args = (
        torch.tensor([0, 8]),
        torch.tensor([0, 1000]),
        torch.ones(1),
        H,
        1,
        False,
        *bufs,
    )
    ok_kwargs = dict(
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=8,
        uni_seqlen_qo=8,
        fast_mode=True,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )

    def use(args=None, **kw):
        return use_bs1_planner(builder, args or ok_args, {**ok_kwargs, **kw})

    def with_arg(i, value):
        return ok_args[:i] + (value,) + ok_args[i + 1 :]

    # Anything but one uniform-qo multi-token request on the plain schedule
    # into the builder's own buffers keeps aiter's planner, without running
    # the verify.
    for args, kw in (
        (with_arg(0, torch.tensor([0, 8, 16])), {}),
        (with_arg(1, torch.tensor([0, 10, 20])), {}),
        (with_arg(4, 2), {}),
        (with_arg(5, torch.tensor(True)), {}),
        (with_arg(6, torch.empty(1)), {}),
        (None, dict(max_seqlen_qo=1, uni_seqlen_qo=1)),
        (None, dict(uni_seqlen_qo=-1)),
        (None, dict(page_size=16)),
        (None, dict(is_cp_round_robin=True)),
        (None, dict(max_split_per_batch=4, intra_batch_mode=False)),
    ):
        assert not use(args, **kw), (args, kw)
    assert planner.verify_calls == []

    # An unverified shape is never verified inside graph capture.
    capturing[0] = True
    assert not use()
    assert planner.verify_calls == []

    # Outside capture it verifies once, then the cached result is reused
    # (also during capture).
    capturing[0] = False
    assert use()
    capturing[0] = True
    assert use()
    assert planner.verify_calls == [(8, H, False, q_dtype, kv_dtype)]
    # A new shape (causal) is its own verify.
    capturing[0] = False
    assert use(with_arg(5, True))
    assert len(planner.verify_calls) == 2

    # A failed verify keeps aiter's planner.
    builder._sd_bs1_planner = _MockPlanner(verify_ok=False)
    assert not use()


def test_planner_hook_only_inside_build_decode(monkeypatch):
    """The wrapped AITER planner takes the call only inside a wrapped
    ``_build_decode`` with the gate and the flag on; otherwise AITER's runs."""
    from vllm.models.kimi_k3.amd.specialized_decode import levers
    from vllm.models.kimi_k3.amd.specialized_decode.levers import mla

    calls = []
    planned = []
    monkeypatch.setattr(mla, "use_bs1_planner", lambda b, a, k: True)
    monkeypatch.setattr(
        mla,
        "bs1_planner",
        lambda b: SimpleNamespace(plan=lambda kv, q: planned.append(q)),
    )
    planner = mla._wrap_planner(lambda *a, **k: calls.append(k["max_seqlen_qo"]))
    assert planner.aiter_planner is not planner
    builder = SimpleNamespace(_asm_dcp_verify=False)
    build = mla._wrap_build_decode(lambda self: planner(0, 0, max_seqlen_qo=7))

    planner(0, 0, max_seqlen_qo=1)  # outside _build_decode
    build(builder)  # gate off
    monkeypatch.setattr(levers, "_active", True)
    monkeypatch.setenv("VLLM_K31_BEAT_DRAFTMETA", "0")
    build(builder)  # flag off
    monkeypatch.setenv("VLLM_K31_BEAT_DRAFTMETA", "1")
    builder._asm_dcp_verify = True
    build(builder)  # asm DCP verify
    assert planned == [] and calls == [1, 7, 7, 7]
    builder._asm_dcp_verify = False
    build(builder)
    assert planned == [7] and calls == [1, 7, 7, 7]
    planner(0, 0, max_seqlen_qo=2)  # the builder is unset after the call
    assert calls == [1, 7, 7, 7, 2]
