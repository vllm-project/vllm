# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the Qwen3.8-Flash-Next specialized decode kernels on MI355X
against the ops vLLM runs for the same modules, random weights in the
checkpoint's formats at TP2 shapes, every width the decode runs (1 .. 16).

- The HC front (one GPU): ``GatedResidual.mix`` (grouped Gemma norm, the
  down + inject GEMM, hc_silu, the up GEMM, the gate-mix) on a materialized
  state, from the previous block's sums of squares or its own.
- The GDN block (one GPU): in_proj_qkvz (online MXFP4) / in_proj_ba -> AITER's
  conv update and fused gated delta rule -> RMSNormGated, the states it leaves
  behind, pad rows.
- The o_proj and MoE blocks (two GPUs): out_proj (MXFP4) / o_proj (bf16, the
  output gate), the sparse MoE block (softmax top-10, the MXFP4 experts, the
  fused shared expert), each + the all-reduce + the HC combine into the
  multi-stream state, and the sums of squares they hand the next front.

No wait of any launch here gives up (the health words stay zero), except in
``test_stalled_rank_fails_stop``: a rank whose peer never launches gives up its
waits within the bound, records them and raises its flag on the output rank.
"""

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.models.qwen4_exp.amd_specialized_decode.reference import (
    COS_MIN,
    _all_reduce_fp32,
    _check_route,
    _cos,
    _gdn_inputs,
    _gdn_reference,
    _hc_weights,
    _mix_ref,
    _moe_emulation,
    _moe_exact,
    _moe_reference,
    _moe_weights,
    _msq,
    _mxfp4_dequant,
    _mxfp4_linear,
    _mxfp4_qdq,
    _on_gfx950,
    _randn,
    _rel_l2,
)
from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)

bf = torch.bfloat16
TP = 2
WIDTHS = (1, 3, 8, 9, 15, 16)

pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the Flash-Next specialized decode needs gfx950"
)


def _health(device="cuda") -> dict:
    """One GPU's ``health`` words and an ``alarm`` word (its own, no output rank)."""
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.plan import (
        HEALTH_WORDS,
    )

    health = torch.zeros(HEALTH_WORDS + 1, dtype=torch.int32, device=device)
    return {
        "health": health[:HEALTH_WORDS],
        "alarm": health[HEALTH_WORDS:].data_ptr(),
    }


@pytest.mark.parametrize("s", WIDTHS)
@pytest.mark.parametrize("self_msq", [False, True])
@torch.inference_mode()
def test_hc_front_matches_vllm_ops(s: int, self_msq: bool) -> None:
    """``hc_front`` against ``GatedResidual.mix``: the block input and the
    injection logits."""
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import front
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import (
        HC,
        HH,
        HIDDEN,
    )

    g = torch.Generator(device="cuda").manual_seed(s)
    state = _randn(g, s, HH, scale=2.0)
    hw = _hc_weights(g)
    want_x, want_inj, _ = _mix_ref(state, hw)
    key = front.FrontBuild(tokens=s, tp=TP, self_msq=self_msq)
    x = torch.full((s, HIDDEN), float("nan"), dtype=bf, device="cuda")
    inj = torch.full((s, HC), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(front.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    hl = _health()
    front.hc_front(
        key,
        state=state,
        msq_in=None if self_msq else _msq(state),
        **hw,
        inj_out=inj,
        bin_out=x,
        scratch=scratch,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        **hl,
        layer=0,
    )
    torch.accelerator.synchronize()
    assert not hl["health"].any()
    assert _cos(inj, want_inj) > COS_MIN
    torch.testing.assert_close(inj, want_inj, atol=2e-2, rtol=2e-2)
    assert _cos(x, want_x) > COS_MIN
    torch.testing.assert_close(x, want_x, atol=2e-2, rtol=2e-2)


# (1, 1) twice: a width again after others compiled (the JIT globals check)
@pytest.mark.parametrize(
    "s,n_real", [(1, 1), (5, 4), (8, 8), (1, 1), (9, 9), (16, 15), (16, 16)]
)
@pytest.mark.parametrize("shuffled", [True, False])
@torch.inference_mode()
def test_gdn_block_matches_vllm_ops(s: int, n_real: int, shuffled: bool) -> None:
    """The GDN block against in_proj -> AITER's conv update + fused gated delta
    rule -> RMSNormGated on one decode step, the states it leaves behind, and
    the pad rows (null block: no state written, zero core); in_proj_qkvz on
    either of AITER's FP4 GEMMs (``shuffled``: the ASM one)."""
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import Layout

    lay = Layout(TP)
    w = _gdn_inputs(s, n_real, shuffled=shuffled)
    ref = _gdn_reference(w, n_real)
    key = blocks.GdnBuild(tokens=s, tp=TP, shuffled=shuffled)
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    core = torch.full((s, lay.CORE), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(
        blocks.scratch_bytes(blocks.gdn_scratch_layout(key)),
        dtype=torch.uint8,
        device="cuda",
    )
    hl = _health()
    blocks.gdn_block(
        key,
        x=ref["x"].contiguous(),
        w_qkvz=w["w_qkvz"],
        w_qkvzs=w["w_qkvzs"],
        w_ba=w["w_ba"],
        conv_w=w["conv_w"],
        conv_state=conv_state,
        a_log=w["a_log"],
        dt_bias=w["dt_bias"],
        norm_w=w["norm_w"],
        rstate=rstate,
        st_idx=w["st_idx"],
        core=core,
        scratch=scratch,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        **hl,
        layer=0,
    )
    torch.accelerator.synchronize()
    assert not hl["health"].any()
    live = w["st_idx"][:n_real].long()
    errs = {
        "conv_state": _rel_l2(conv_state[live], ref["conv_state"][live]),
        "rstate": _rel_l2(rstate[live], ref["rstate"][live]),
        "core": _rel_l2(core[:n_real], ref["core"][:n_real]),
    }
    assert errs["conv_state"] < 5e-3 and errs["rstate"] < 5e-3, errs
    assert _cos(core[:n_real], ref["core"][:n_real]) > COS_MIN, errs
    assert torch.equal(core[n_real:], torch.zeros_like(core[n_real:]))
    torch.testing.assert_close(rstate[0], w["rstate"][0], atol=0, rtol=0)
    torch.testing.assert_close(conv_state[0], w["conv_state"][0], atol=0, rtol=0)


def _tp_state(rank: int, tp_size: int, device) -> dict:
    """A rank's launch state as ``Launcher`` builds it: the peer buffer (the
    output rank's flags in its head), the health words, the alarm word."""
    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.abi import PEER_HEAD
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import MAX_TOKENS
    from vllm.models.qwen4_exp.amd.specialized_decode.runtime.peer import PeerBuffer

    peers = PeerBuffer(
        PEER_HEAD + blocks.peer_bytes(MAX_TOKENS, TP),
        get_tp_group().cpu_group,
        rank,
        tp_size,
        device,
    )
    peers.bytes.zero_()
    return {
        "buffer": peers,
        "flags": peers.bytes[: 4 * tp_size].view(torch.int32),
        "launch": {
            "peers": peers.addresses + PEER_HEAD,
            "rank": rank,
            "health": _health(device)["health"],
            "alarm": int(peers.addresses[0].item()) + 4 * rank,
        },
    }


@ray.remote(num_gpus=1, max_calls=1)
def _blocks_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import (
        HC,
        HIDDEN,
        Layout,
    )

    lay = Layout(TP)
    tps = _tp_state(rank, tp_size, device)
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    mw = _moe_weights(rank, device)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    gc = torch.Generator(device=device).manual_seed(11)
    oq_raw, os_raw, w_oq, w_os = _mxfp4_linear(gr, HIDDEN, lay.CORE)
    w_oq_ref = _mxfp4_dequant(oq_raw, os_raw).to(bf)
    # the out_proj Triton FP4 GEMM's layout: the same weight, unshuffled
    flat = _mxfp4_linear(gr, HIDDEN, lay.CORE, shuffled=False)
    w_of_ref = _mxfp4_dequant(flat[0], flat[1]).to(bf)
    w_o = _randn(gr, HIDDEN, lay.CORE, scale=0.02)

    def agree(t, what):
        outs = [torch.empty_like(t) for _ in range(tp_size)]
        dist.all_gather(outs, t)
        assert all(torch.equal(outs[0], o) for o in outs), (what, "ranks disagree")

    def launch(block, key, state, inj, **kw):
        s = key.tokens
        out = torch.full_like(state, float("nan"))
        msq = torch.full((blocks.NRED, s, HC), float("nan"), device=device)
        block(key, out=out, state=state, inj=inj, msq_out=msq, **kw)
        torch.accelerator.synchronize()
        return out.unflatten(-1, (HC, HIDDEN)), msq

    def check(block, key, what, layer, **kw):
        """``block`` into a zero state with zero injection logits (every stream
        the all-reduced block output: the multiplier is 1), then into a random
        state against vLLM's HC combine of that output and the sums of
        squares the next front reads; the two launches on mailbox slots
        ``layer`` and ``layer + 2``. -> the block output."""
        s = key.tokens
        zero = torch.zeros(s, HC * HIDDEN, dtype=bf, device=device)
        got, _ = launch(block, key, zero, zero.new_zeros(s, HC), layer=layer, **kw)
        blk = got[:, 0].contiguous()
        assert all(torch.equal(got[:, k], blk) for k in range(HC)), what
        agree(blk, what)
        state = _randn(gc, s, HC * HIDDEN)
        inj = _randn(gc, s, HC, scale=2.0)
        got, msq = launch(block, key, state, inj, layer=layer + 2, **kw)
        m = 2.0 * torch.sigmoid(inj.float() / HC)
        ref = (
            state.float().unflatten(-1, (HC, HIDDEN))
            + blk.float()[:, None] * m[..., None]
        ).to(bf)
        torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2, msg=str(what))
        sq = got.float().unflatten(-1, (blocks.NRED, -1)).square().sum(-1)
        torch.testing.assert_close(msq, sq.permute(2, 0, 1), rtol=1e-4, atol=0)
        agree(got, what)
        return blk

    # each width twice: both epoch parities, one layer's oproj and moe a step
    for s in [w for w in (1, 3, 8, 9, 15, 16) for _ in range(2)]:
        epoch.add_(1)
        # (oq, shuffled): out_proj on the ASM / Triton FP4 GEMM, o_proj bf16
        for slot, (oq, shuffled) in enumerate(
            ((True, True), (True, False), (False, True))
        ):
            core = _randn(gr, s, lay.CORE)
            gate = None if oq else _randn(gr, s, lay.CORE, scale=2.0)
            w, ws, ref_w = (
                (w_oq, w_os, w_oq_ref) if shuffled else (flat[2], flat[3], w_of_ref)
            )
            out = check(
                blocks.oproj_block,
                blocks.OprojBuild(tokens=s, tp=TP, oq=oq, og=not oq, shuffled=shuffled),
                ("oproj", oq, shuffled, s),
                core=core,
                gate=gate,
                w_o=w if oq else w_o,
                w_os=ws if oq else None,
                **tps["launch"],
                epoch=epoch,
                layer=blocks.OPROJ_SLOT + 4 * slot,
            )
            if oq:
                part = F.linear(_mxfp4_qdq(core, shuffled).to(bf), ref_w)
            else:
                assert gate is not None
                gated = (core.float() * torch.sigmoid(gate.float())).to(bf)
                part = F.linear(gated, w_o)
            err = _rel_l2(out, _all_reduce_fp32(part))
            assert err < 1e-2, ("oproj", oq, shuffled, s, err)

        key = blocks.MoeBuild(tokens=s, tp=TP)
        x = _randn(gc, s, HIDDEN)
        scr = blocks.moe_scratch_layout(key)
        scratch = torch.zeros(
            blocks.scratch_bytes(scr), dtype=torch.uint8, device=device
        )
        out = check(
            blocks.moe_block,
            key,
            ("moe", s),
            x=x,
            scratch=scratch,
            **tps["launch"],
            epoch=epoch,
            layer=blocks.MOE_SLOT,
            **{k: v for k, v in mw.items() if k != "raw"},
        )
        ids = _check_route(scratch, scr["route"], x.clone(), mw)
        assert _cos(out, _moe_emulation(x.clone(), mw, ids)) > COS_MIN, ("moe", s)
        exact = _moe_exact(x.clone(), mw, ids)
        err = _rel_l2(out, exact)
        err_stock = _rel_l2(_moe_reference(x.clone(), mw, ids), exact)
        assert err < 1.02 * err_stock, ("moe", s, err, err_stock)
    assert not tps["launch"]["health"].any() and not tps["flags"].any()
    tps["buffer"].close()


@multi_gpu_test(num_gpus=2)
def test_blocks_tp2_match_vllm_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 2, 1, _blocks_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _stall_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    import time

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.abi import (
        FAILED,
        WAIT_SECONDS,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.debug import (
        given_up_waits,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import (
        HC,
        HIDDEN,
        Layout,
    )

    lay = Layout(TP)
    tps = _tp_state(rank, tp_size, device)
    health = tps["launch"]["health"]
    key = blocks.OprojBuild(tokens=1, tp=TP, oq=False, og=True)
    g = torch.Generator(device=device).manual_seed(rank)
    state = _randn(g, 1, HC * HIDDEN)

    def launch_alone() -> float:
        """The o_proj block on this rank only: its peer never sends its partial."""
        t0 = time.monotonic()
        blocks.oproj_block(
            key,
            core=_randn(g, 1, lay.CORE),
            gate=_randn(g, 1, lay.CORE),
            w_o=_randn(g, HIDDEN, lay.CORE, scale=0.02),
            state=state,
            inj=_randn(g, 1, HC),
            out=torch.empty_like(state),
            msq_out=torch.empty(blocks.NRED, 1, HC, device=device),
            **tps["launch"],
            epoch=torch.ones(1, dtype=torch.int32, device=device),
            layer=blocks.OPROJ_SLOT,
        )
        torch.accelerator.synchronize()
        return time.monotonic() - t0

    if rank == 0:
        took = launch_alone()
        assert WAIT_SECONDS * 0.9 < took < WAIT_SECONDS * 6, took
        assert health[FAILED] != 0
        waits = given_up_waits(health.cpu(), blocks.PEER_REGIONS)
        assert waits and all(" ar pair " in w or " final pair " in w for w in waits)
        assert tps["flags"].tolist() == [1, 0], "the output rank's flags"
        # sticky: a failed rank's waits exit at once
        assert launch_alone() < WAIT_SECONDS / 2
    get_tp_group().barrier()
    if rank == 1:
        assert not health.any()
    tps["buffer"].close()


@multi_gpu_test(num_gpus=2)
def test_stalled_rank_fails_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rank whose peer never launches gives up its waits within the bound
    instead of hanging: FAILED, a record naming the peer region, its flag on
    the output rank, and its later launches return at once."""
    multi_process_parallel(monkeypatch, 2, 1, _stall_worker)
