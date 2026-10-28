# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the Qwen3.8 specialized decode kernels on MI355X against the
ops vLLM runs for the same modules, random weights in the checkpoint's formats
at the TP8 rank shapes, widths the decode runs (1 .. 16).

- The GDN block (one GPU): the residual add + input_layernorm (Gemma), then
  in_proj_qkvz (online MXFP4) / in_proj_ba -> AITER's conv update and fused
  gated delta rule -> RMSNormGated (SiLU gate), the residual and the states it
  leaves behind, pad rows.
- The layer block (eight GPUs): out_proj (MXFP4) / o_proj (bf16, the output
  gate) + the all-reduce, the residual add + post_attention_layernorm, the
  sparse MoE block (softmax top-10, the MXFP4 experts, the fused shared
  expert) + the all-reduce.

No wait of any launch here gives up (the health words stay zero), except in
``test_stalled_rank_fails_stop``: a rank whose peers never launch gives up its
waits within the bound, records them and raises its flag on the output rank.
"""

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.models.qwen3_5.amd_specialized_decode.reference import (
    COS_MIN,
    _all_reduce_fp32,
    _block_input,
    _check_route,
    _cos,
    _gdn_inputs,
    _gdn_reference,
    _gemma_norm,
    _moe_emulation,
    _moe_exact,
    _moe_reference,
    _moe_weights,
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
TP = 8

pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the Qwen3.8 specialized decode needs gfx950"
)


def _health(device="cuda") -> dict:
    """One GPU's ``health`` words and an ``alarm`` word (its own, no output rank)."""
    from vllm.models.qwen3_5.amd.specialized_decode.mono.common.plan import (
        HEALTH_WORDS,
    )

    health = torch.zeros(HEALTH_WORDS + 1, dtype=torch.int32, device=device)
    return {
        "health": health[:HEALTH_WORDS],
        "alarm": health[HEALTH_WORDS:].data_ptr(),
    }


# (1, 1) twice: a width again after others compiled (the JIT globals check)
@pytest.mark.parametrize(
    "s,n_real,first",
    [
        (1, 1, False),
        (1, 1, True),
        (5, 4, False),
        (8, 8, False),
        (1, 1, False),
        (9, 9, False),
        (16, 15, True),
        (16, 16, False),
    ],
)
@pytest.mark.parametrize("shuffled", [True, False])
@torch.inference_mode()
def test_gdn_block_matches_vllm_ops(
    s: int, n_real: int, first: bool, shuffled: bool
) -> None:
    """The GDN block against the Gemma residual add + norm, in_proj -> AITER's
    conv update + fused gated delta rule -> RMSNormGated on one decode step,
    the residual and the states it leaves behind, and the pad rows (null
    block: no state written, zero core); in_proj_qkvz on either of AITER's FP4
    GEMMs (``shuffled``: the ASM one)."""
    from vllm.models.qwen3_5.amd.specialized_decode.mono import blocks
    from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L

    w = _gdn_inputs(s, n_real, shuffled=shuffled, first=first)
    ref = _gdn_reference(w, n_real)
    key = blocks.GdnBuild(tokens=s, first=first, shuffled=shuffled)
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    core = torch.full((s, L.CORE), float("nan"), dtype=bf, device="cuda")
    res_out = torch.full_like(w["hidden"], float("nan"))
    scratch = torch.zeros(
        blocks.scratch_bytes(blocks.gdn_scratch_layout(key)),
        dtype=torch.uint8,
        device="cuda",
    )
    hl = _health()
    blocks.gdn_block(
        key,
        hidden=w["hidden"],
        residual=w["residual"],
        res_out=res_out,
        ln_w=w["ln_w"],
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
    torch.testing.assert_close(res_out, ref["res"], atol=0, rtol=0)
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
    from vllm.models.qwen3_5.amd.specialized_decode.mono import blocks
    from vllm.models.qwen3_5.amd.specialized_decode.mono.abi import PEER_HEAD
    from vllm.models.qwen3_5.amd.specialized_decode.mono.layout import MAX_TOKENS
    from vllm.models.qwen3_5.amd.specialized_decode.runtime.peer import PeerBuffer

    peers = PeerBuffer(
        PEER_HEAD + blocks.peer_bytes(MAX_TOKENS),
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
def _layer_worker(
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

    from vllm.models.qwen3_5.amd.specialized_decode.mono import blocks
    from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L

    h, ck = L.HIDDEN, L.CORE
    tps = _tp_state(rank, tp_size, device)
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    mw = _moe_weights(rank, device)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    gc = torch.Generator(device=device).manual_seed(11)  # replicated inputs
    oq_raw, os_raw, w_oq, w_os = _mxfp4_linear(gr, h, ck)
    w_oq_ref = _mxfp4_dequant(oq_raw, os_raw).to(bf)
    # the out_proj Triton FP4 GEMM's layout: the same kind of weight, unshuffled
    flat = _mxfp4_linear(gr, h, ck, shuffled=False)
    w_of_ref = _mxfp4_dequant(flat[0], flat[1]).to(bf)
    w_o = _randn(gr, h, ck, scale=0.02)
    ln_w = _randn(gc, h, scale=0.2)

    def agree(t, what):
        outs = [torch.empty_like(t) for _ in range(tp_size)]
        dist.all_gather(outs, t)
        assert all(torch.equal(outs[0], o) for o in outs), (what, "ranks disagree")

    # each width twice: both epoch parities
    for s in [w for w in (1, 3, 8, 9, 15, 16) for _ in range(2)]:
        epoch.add_(1)
        # (oq, shuffled): out_proj on the ASM / Triton FP4 GEMM, o_proj bf16
        for slot, (oq, shuffled) in enumerate(
            ((True, True), (True, False), (False, True))
        ):
            what = ("layer", oq, shuffled, s)
            key = blocks.LayerBuild(tokens=s, oq=oq, og=not oq, shuffled=shuffled)
            core = _randn(gr, s, ck)
            gate = None if oq else _randn(gr, s, ck, scale=2.0)
            residual = _block_input(gc, s, False)["residual"]
            w, ws, ref_w = (
                (w_oq, w_os, w_oq_ref) if shuffled else (flat[2], flat[3], w_of_ref)
            )
            scr = blocks.layer_scratch_layout(key)
            scratch = torch.zeros(
                blocks.scratch_bytes(scr), dtype=torch.uint8, device=device
            )
            out = torch.full_like(residual, float("nan"))
            res_out = torch.full_like(residual, float("nan"))
            blocks.layer_block(
                key,
                core=core,
                gate=gate,
                residual=residual,
                w_o=w if oq else w_o,
                w_os=ws if oq else None,
                ln_w=ln_w,
                out=out,
                res_out=res_out,
                scratch=scratch,
                **tps["launch"],
                epoch=epoch,
                layer=blocks.LAYER_SLOT + slot,
                **{k: v for k, v in mw.items() if k != "raw"},
            )
            torch.accelerator.synchronize()
            agree(out, what)
            agree(res_out, what)
            if oq:
                part = F.linear(_mxfp4_qdq(core, shuffled).to(bf), ref_w)
            else:
                assert gate is not None
                gated = (core.float() * torch.sigmoid(gate.float())).to(bf)
                part = F.linear(gated, w_o)
            o_sum = _all_reduce_fp32(part)
            x_ref, res_ref = _gemma_norm(o_sum, residual, ln_w)
            # the kernel's AR sum: the same partials in another fp32 order
            assert _rel_l2(res_out, res_ref) < 1e-2, (what, "residual")
            # the MoE on the kernel's own AR sum (xbuf): x as the kernel and
            # vLLM's fused add + norm build it, from the unrounded fp32 sum
            xo = scr["xbuf"][0]
            ar = scratch[xo : xo + s * h * 2].view(bf).view(s, h)
            x, res_got = _gemma_norm(ar, residual, ln_w)
            assert torch.equal(res_got, res_out), (what, "residual out")
            ids = _check_route(scratch, scr["route"], x.clone(), mw)
            emu = _moe_emulation(x.clone(), mw, ids)
            assert _cos(out, emu) > COS_MIN, (what, _cos(out, emu))
            exact = _moe_exact(x.clone(), mw, ids)
            err = _rel_l2(out, exact)
            err_stock = _rel_l2(_moe_reference(x.clone(), mw, ids), exact)
            assert err < 1.02 * err_stock, (what, err, err_stock)
    assert not tps["launch"]["health"].any() and not tps["flags"].any()
    tps["buffer"].close()


@multi_gpu_test(num_gpus=TP)
def test_layer_tp8_matches_vllm_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, TP, 1, _layer_worker)


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
    from vllm.models.qwen3_5.amd.specialized_decode.mono import blocks
    from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L
    from vllm.models.qwen3_5.amd.specialized_decode.mono.abi import (
        FAILED,
        WAIT_SECONDS,
    )
    from vllm.models.qwen3_5.amd.specialized_decode.mono.common.debug import (
        given_up_waits,
    )

    tps = _tp_state(rank, tp_size, device)
    health = tps["launch"]["health"]
    key = blocks.LayerBuild(tokens=1, oq=False, og=True)
    mw = _moe_weights(rank, device)
    g = torch.Generator(device=device).manual_seed(rank)
    scratch = torch.zeros(
        blocks.scratch_bytes(blocks.layer_scratch_layout(key)),
        dtype=torch.uint8,
        device=device,
    )

    def launch_alone() -> float:
        """The layer block on this rank only: its peers never send partials."""
        t0 = time.monotonic()
        residual = _randn(g, 1, L.HIDDEN)
        blocks.layer_block(
            key,
            core=_randn(g, 1, L.CORE),
            gate=_randn(g, 1, L.CORE),
            residual=residual,
            w_o=_randn(g, L.HIDDEN, L.CORE, scale=0.02),
            ln_w=_randn(g, L.HIDDEN, scale=0.2),
            out=torch.empty_like(residual),
            res_out=torch.empty_like(residual),
            scratch=scratch,
            **tps["launch"],
            epoch=torch.ones(1, dtype=torch.int32, device=device),
            layer=blocks.LAYER_SLOT,
            **{k: v for k, v in mw.items() if k != "raw"},
        )
        torch.accelerator.synchronize()
        return time.monotonic() - t0

    if rank == 0:
        took = launch_alone()
        assert WAIT_SECONDS * 0.9 < took < WAIT_SECONDS * 6, took
        assert health[FAILED] != 0
        # the norm CTAs give up on the peers' partials, the CTAs after them on
        # this rank's mailboxes the norm CTAs never filled
        names = (*blocks.PEER_REGIONS, *blocks.LAYER_REGIONS)
        waits = given_up_waits(health.cpu(), names)
        assert waits and all(any(f" {r} pair " in w for r in names) for w in waits)
        assert any(f" {r} pair " in w for w in waits for r in blocks.PEER_REGIONS)
        assert tps["flags"].tolist() == [1] + [0] * (tp_size - 1)
        # sticky: a failed rank's waits exit at once
        assert launch_alone() < WAIT_SECONDS / 2
    get_tp_group().barrier()
    if rank != 0:
        assert not health.any()
    tps["buffer"].close()


@multi_gpu_test(num_gpus=TP)
def test_stalled_rank_fails_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rank whose peers never launch gives up its waits within the bound
    instead of hanging: FAILED, a record naming the peer region, its flag on
    the output rank, and its later launches return at once."""
    multi_process_parallel(monkeypatch, TP, 1, _stall_worker)
