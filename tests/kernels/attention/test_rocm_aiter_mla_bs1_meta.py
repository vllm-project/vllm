# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The closed-form bs=1 MLA planner (VLLM_K31_BEAT_DRAFTMETA) against aiter's
serial planner: identical schedules, identical attention output, and a
graph-captured plan that follows the KV length on replay."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="aiter MLA planner is ROCm-only"
)

H = 16  # draft heads per rank, asm-padded (DSpark 96 heads / TP8 -> 12 -> 16)
MAX_QLEN = 8


def _bufs(dtype):
    from aiter import get_mla_metadata_info_v1

    info = get_mla_metadata_info_v1(
        1, MAX_QLEN, H, dtype, dtype, is_sparse=False, fast_mode=True
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


@pytest.mark.parametrize("qlen", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("causal", [True, False])
def test_bs1_planner_matches_aiter(qlen, causal):
    from aiter import dtypes

    from vllm.v1.attention.ops.rocm_aiter_mla_bs1_meta import AiterMlaBs1Planner

    fp8 = dtypes.fp8
    planner = AiterMlaBs1Planner(*_bufs(fp8))
    # Dense below 20k (the whole unsaturated split law and every tail-fold
    # boundary at that scale), then strided out to 1M tokens.
    kv_lens = list(range(0, 20000)) + list(range(20000, 1_050_000, 1013))
    assert planner.verify(qlen, H, causal, fp8, fp8, kv_lens=kv_lens)
    assert planner.is_verified(qlen, H, causal, fp8, fp8)


@pytest.mark.parametrize("kv_len", [7, 23, 1000, 4000, 65537, 360000])
def test_bs1_planner_attention_output_identical(kv_len):
    from aiter import dtypes
    from aiter.mla import mla_decode_fwd

    from vllm.v1.attention.ops.rocm_aiter_mla_bs1_meta import AiterMlaBs1Planner

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


def test_bs1_planner_graph_replay_tracks_kv_len():
    from aiter import dtypes

    from vllm.v1.attention.ops.rocm_aiter_mla_bs1_meta import AiterMlaBs1Planner

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


def test_drop_draft_only_attn_groups():
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    spec = object.__new__(DFlashSpeculator)
    spec.draft_attn_layer_names = {"draft.0.attn", "draft.1.attn"}
    runner = SimpleNamespace(speculator=spec)

    def group(names, mamba=False):
        kv_spec = object.__new__(MambaSpec) if mamba else object()
        return SimpleNamespace(layer_names=names, kv_cache_spec=kv_spec)

    target = group(["model.0.attn", "model.3.attn"])
    draft = group(["draft.0.attn", "draft.1.attn"])
    mamba = group(["model.1.kda"], mamba=True)
    mixed = group(["model.5.attn", "draft.1.attn"])
    groups = [[target], [draft], [mamba], [mixed, draft]]

    out = GPUModelRunner._drop_draft_only_attn_groups(runner, groups)
    # Positions kept (block tables index by group id); only draft-only gone.
    assert out == [[target], [], [mamba], [mixed]]

    # Other speculators keep every group (they may reuse target metadata).
    runner.speculator = SimpleNamespace(draft_attn_layer_names={"draft.0.attn"})
    assert GPUModelRunner._drop_draft_only_attn_groups(runner, groups) is groups
