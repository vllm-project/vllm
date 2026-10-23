# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 (one Muse Spark layer from the gated attention core to the deferred FFN
boundary) on MI355X against the ops vLLM runs for the same decode step, random
weights in the post-load formats at TP8 shapes.

Each stage is checked tightly on its own inputs as the kernel saw them
(the mailbox readbacks), so an eps, gain-center or order mistake in one stage
cannot hide behind upstream noise; the chain (``moe``, ``branch_out``) against
vLLM's stock decode. Every case has an eps-scaled row (the attention output,
the pooled rows, one route's expert rows) and one case all-zero norm weights
(gain ``w + 1``); the base case at every width 1 .. 8. Then: the ranks agree
bit for bit; one CUDA graph of two steps at two widths replayed 300 times on
two input sets written in place in turn, every replay bit for bit the eager
step on its own set; pad rows (NaN inputs)
come out as exact zeros and leave the live rows unchanged; an all-pad step
completes. And the fail-stop: a stalled rank fails the step without a hang.
"""

import contextlib

import pytest
import ray
import torch
import torch.distributed as dist

from tests.models.muse_spark.amd_specialized_decode import reference as ref
from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)
from vllm.models.muse_spark.amd.specialized_decode.mono import spec
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (
    FAILED,
    PEER_HEAD,
    RECORD_WORDS,
    STARTUP_WAIT_SECONDS,
    WAIT_BUDGET,
    WAIT_SECONDS,
)
from vllm.platforms import current_platform

bf = torch.bfloat16
f32 = torch.float32
H, HM, CORE, INTER = spec.HIDDEN, spec.MOE_HIDDEN, spec.CORE, spec.INTER
E, TOPK, TP, MT, MP = spec.EXPERTS, spec.TOPK, spec.TP, spec.MAX_TOKENS, spec.MAX_PICKS
SLOT = spec.k2_slot(1)  # the first kernel layer's (layer 0 stays stock)
STALLED = 3
COS_STAGE = 0.99999
COS_CHAIN = 0.9999
REL_CHAIN = 1e-2
CASES = ("base", "eps", "zero_w")
WIDTHS = tuple(range(1, spec.MAX_ROWS + 1))
REPLAYS = 300
EPS_EXPERT = 17  # the "eps" case's forced expert, its w2 scaled by 2^-EPS_W2_SHIFT
# unscaled, a route's R at these weights has mean square ~370 (CPU reference,
# 8 ranks): 2^-17 puts mean((R (w + 1))^2) at ~2e-8, i.e. at POST_EXP_EPS
EPS_W2_SHIFT = 17


def _on_gfx950() -> bool:
    if not torch.cuda.is_available() or not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


needs_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the Muse mono kernels need gfx950 (MI355X)"
)


# The host readback of the mailbox regions.


def _region(runner, name: str) -> tuple[torch.Tensor, int]:
    """(byte buffer, byte offset) of ``name`` as the kernels address it: a
    scratch region, or a peer region in the current epoch's parity half."""
    lay = spec.scratch_layout()
    if name in lay:
        return runner.scratch, lay[name][0]
    e = int(runner.epoch.item()) & spec.EPOCH_MASK
    off = PEER_HEAD + (e & 1) * spec.peer_half_bytes() + spec.peer_layout()[name][0]
    return runner.peers.bytes, off


def _pairs(runner, name: str, index: int, n: int):
    """(values, tags) of pairs ``index`` .. ``index + n`` of region ``name``
    (host copies; ``.view(torch.float32)`` an fp32 payload, two bf16 a packed
    one)."""
    buf, off = _region(runner, name)
    off += 8 * index
    assert off % 8 == 0 and off + 8 * n <= buf.numel(), (name, off, n)
    pairs = buf[off : off + 8 * n].view(torch.int32).view(n, 2)
    return pairs[:, 0].cpu(), pairs[:, 1].cpu()


def _tag(runner) -> int:
    return spec.step_tag(int(runner.epoch.item()), SLOT)


def _owner_sum(part: torch.Tensor) -> torch.Tensor:
    """A rank's 32 owner slices [..., 32] summed in ``stage_owner``'s fp32
    order: lane rp the 4 slices 4 rp .. in order, then the butterfly over the
    8 lanes (xor 4, 2, 1), which every lane ends with (lane 0's order)."""
    lane = part[..., 0::4] + part[..., 1::4]
    lane = (lane + part[..., 2::4]) + part[..., 3::4]  # [..., 8]: lane rp
    for off in (4, 2, 1):
        lane = lane + lane[..., [rp ^ off for rp in range(8)]]
    return lane[..., 0]


def readback(runner, S: int) -> dict:
    """Every mailbox region of the last launch (rows < S), decoded, and the
    tags each pair carried: ``ar`` bf16 [TP, S, H] (src's partial), ``ass`` /
    ``bss`` fp32 [NA, S], ``rlog`` fp32 [S, E] (expert 32 src + i), ``route``
    (ids int32 [S, 8], weights fp32), ``pool`` bf16 [S, HM], ``rsx`` bf16 [TP,
    S, 8, OWN_COLS] (src's down rows of this rank's columns), ``sqx_part``
    fp32 [TP, RED_TASKS, S, 8] (src's owner CTA j's 16 columns), ``sqx`` fp32
    [TP, S, 8] (src's 512 columns, summed as the kernel does), ``moe`` bf16 [S,
    HM], ``branch`` bf16 [S, H]; plain ``xrow`` bf16 [S, H]."""
    out, tags = {}, {}

    def rows(name, n_rows, row_pairs):
        v, t = _pairs(runner, name, 0, n_rows * row_pairs)
        tags[name] = t.view(n_rows, row_pairs)
        return v.view(n_rows, row_pairs)

    out["ar"] = rows("ar", TP * MT, H // 2).view(TP, MT, H // 2)[:, :S]
    tags["ar"] = tags["ar"].view(TP, MT, -1)[:, :S]
    out["ar"] = out["ar"].contiguous().view(bf).view(TP, S, H)
    for name in ("ass", "bss"):
        v = rows(name, spec.NA, MT)
        out[name] = v[:, :S].contiguous().view(f32)
        tags[name] = tags[name][:, :S]
    v = rows("rlog", TP * MT, spec.EXPERTS_RANK).view(TP, MT, -1)[:, :S]
    tags["rlog"] = tags["rlog"].view(TP, MT, -1)[:, :S]
    out["rlog"] = v.contiguous().view(f32).permute(1, 0, 2).reshape(S, E)
    v = rows("route", MT, 16)[:S]
    tags["route"] = tags["route"][:S]
    out["ids"] = v[:, :TOPK].clone()
    out["tw"] = v[:, TOPK:].contiguous().view(f32)
    for name, width in (("pool", HM), ("moe", HM), ("branch", H)):
        v = rows(name, MT, width // 2)[:S]
        tags[name] = tags[name][:S]
        out[name] = v.contiguous().view(bf).view(S, width)
    v = rows("rsx", TP * MP, spec.OWN_COLS // 2).view(TP, MP, -1)[:, : S * TOPK]
    tags["rsx"] = tags["rsx"].view(TP, MP, -1)[:, : S * TOPK]
    out["rsx"] = v.contiguous().view(bf).view(TP, S, TOPK, spec.OWN_COLS)
    v = rows("sqx", TP * MP, spec.RED_TASKS).view(TP, MP, -1)[:, : S * TOPK]
    tags["sqx"] = tags["sqx"].view(TP, MP, -1)[:, : S * TOPK]
    part = v.contiguous().view(f32).view(TP, S, TOPK, spec.RED_TASKS)
    out["sqx_part"] = part.permute(0, 3, 1, 2).contiguous()
    out["sqx"] = _owner_sum(part)
    lay = spec.scratch_layout()
    o = lay[spec.XROW][0]
    xrow = runner.scratch[o : o + spec.XROW_BYTES].cpu().view(bf).view(MT, H)
    out["xrow"] = xrow[:S].clone()
    out["tags"] = tags
    return out


# ---------------------------------------------------------------- [TP8]


def _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    return device


def _gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The unquantized GEMM dispatch of vLLM (``rocm_unquantized_gemm`` on gfx950),
    as ``UnquantizedLinearMethod.apply`` runs a linear layer's weight: the o_proj
    reference only."""
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

    return dispatch_unquantized_gemm()(torch.nn.Module(), x, w, None)


def _linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The pooler's and the unpooler's GEMM: a plain ``F.linear``, as
    ``MuseSparkTorchLinearMethod`` runs them (amd/model.py: not the AITER
    tuned-GEMM dispatch, whose split-K kernels accumulate in bf16)."""
    return torch.nn.functional.linear(x, w)


@contextlib.contextmanager
def _tp_checks(what: str):
    """Rank-local checks as one TP verdict. A check that fails on some ranks
    only must not leave the others in the next collective: vLLM's custom
    all-reduce and NCCL spin with no timeout, and the exclusive node hangs.
    Every rank enters and leaves the block at the same point; on exit the ranks
    vote over the TP CPU (gloo) group and all raise if any failed (the failing
    ranks their own error). No collective inside the block."""
    from vllm.distributed import get_tp_group
    from vllm.models.muse_spark.amd.specialized_decode.runtime.peer import tp_agree

    err: Exception | None = None
    try:
        yield
    except Exception as e:  # voted below, then raised on every rank
        err = e
    if not tp_agree(err is None, get_tp_group().cpu_group):
        if err is not None:
            raise err
        raise AssertionError(f"another TP rank failed {what}")


def _weights(rank: int, device, case: str = "base") -> dict:
    """One rank's K2 weights (post-load formats; replicated tensors from seed
    7, the rank's shards from 100 + rank) and the experts' unshuffled bytes
    and codes (``ckpt``). ``eps``: the pooler scaled so mean(pooled^2) ~
    PRE_EXP_EPS, expert EPS_EXPERT always picked and its w2 scaled by
    2^-EPS_W2_SHIFT (that route's mean square ~ POST_EXP_EPS), the residual
    gate -2.5 (beta ~ 2.4e-4, see ``_inputs``). ``zero_w``:
    every norm weight zero."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xNormPreservingResidualConnection,
    )

    gc = torch.Generator(device=device).manual_seed(7)
    gr = torch.Generator(device=device).manual_seed(100 + rank)

    def rn(g, *shape, s=1.0, dtype=bf):
        return (torch.randn(*shape, generator=g, device=device) * s).to(dtype)

    def u8(g, *shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, generator=g, device=device, dtype=torch.int64
        ).to(torch.uint8)

    conn = MuseSpark1xNormPreservingResidualConnection(H).to(device)
    with torch.no_grad():
        conn.gate.copy_(rn(gc, H, dtype=f32))
        if case == "eps":
            # beta = sigmoid(-2.5 / 0.3) ~ 2.4e-4 (alpha ~ 1) on every channel:
            # the pre-FFN norm's input of a row with a small stream is small
            # too (``_inputs``)
            conn.gate.fill_(-2.5)
    conn.materialize_coefficients()
    w = {
        "attn_alpha": conn.alpha.contiguous(),
        "attn_beta": conn.beta.contiguous(),
        "w_preffn": rn(gc, H, s=0.1),
        # the router multiplier folded in (0.1768) at load
        "w_router": rn(gc, E, H, s=0.02 * 0.1768, dtype=f32),
        "bias": rn(gc, E, s=0.02, dtype=f32),
        "w_pool": rn(gc, HM, H, s=H**-0.5),
        "w_preexp": rn(gc, HM, s=0.1),
        "w_postexp": rn(gc, HM, s=0.1),
        "w_unpool": rn(gc, H, HM, s=HM**-0.5),
        "w_o": rn(gr, H, CORE, s=(TP * CORE) ** -0.5),
    }
    w13 = u8(gr, E, 2 * INTER, HM // 2)
    c13 = u8(gr, E, 2 * INTER, HM // 32, lo=118, hi=123)
    w2 = u8(gr, E, HM, INTER // 2)
    c2 = u8(gr, E, HM, INTER // 32, lo=118, hi=123)
    if case == "eps":
        w["w_pool"] = (w["w_pool"].float() * 3e-3).to(bf)
        w["bias"][EPS_EXPERT] = 10.0
        c2[EPS_EXPERT] -= EPS_W2_SHIFT
    if case == "zero_w":
        for n in ("w_preffn", "w_preexp", "w_postexp"):
            w[n] = torch.zeros_like(w[n])

    def shuf_scale(c):
        # e8m0_shuffle on the [E N, K / 32] view of the uint8 codes, the dtype
        # Quark loads them in (oracle/mxfp4.py AITER_MXFP4_MXFP4 branch): the
        # stock MoE gets exactly the tensor the served layer holds
        s0, s1, n = c.shape
        return e8m0_shuffle(c.reshape(s0 * s1, n)).view(s0, s1, -1).contiguous()

    w["w13"] = shuffle_weight(w13, layout=(16, 16)).contiguous()
    w["w2"] = shuffle_weight(w2, layout=(16, 16)).contiguous()
    # the stock MoE takes typed views (fp4x2 weights, e8m0 scales); the
    # kernel the same bytes by address
    w["w13_fp4"] = w["w13"].view(torch.float4_e2m1fn_x2)
    w["w2_fp4"] = w["w2"].view(torch.float4_e2m1fn_x2)
    # as oracle/mxfp4.py marks the loaded weights: aiter's fused_moe picks its
    # kernel (and the weight layout it assumes) from this attribute
    w["w13_fp4"].is_shuffled = True
    w["w2_fp4"].is_shuffled = True
    w["w13s"] = shuf_scale(c13)
    w["w2s"] = shuf_scale(c2)
    w["ckpt"] = (w13, c13, w2, c2)
    return w


def _inputs(rank: int, device, S: int, case: str = "base", seed: int = 0):
    """(core [S, CORE] bf16 of this rank, stream [S, H] fp32 replicated);
    ``seed`` picks another set of the same width. The
    "eps" case scales row 0's core on every rank by 1e-4: mean(attn_out^2) ~
    POST_ATTN_EPS; and the last row's stream by 3e-3: with that case's beta ~
    2.4e-4 the pre-FFN norm's input alpha stream + beta y has mean square
    ~9e-6 ~ PRE_FFN_EPS."""
    gc = torch.Generator(device=device).manual_seed(11 + S + 1000 * seed)
    gr = torch.Generator(device=device).manual_seed(200 + 8 * S + rank + 1000 * seed)
    core = torch.randn(S, CORE, generator=gr, device=device)
    if case == "eps":
        core[0] *= 1e-4
    stream = torch.randn(S, H, generator=gc, device=device)
    if case == "eps":
        stream[S - 1] *= 3e-3
    return core.to(bf), stream


def _pad(device, rows=()) -> torch.Tensor:
    """The pad flags: whole words (``ld_pad`` reads the word of a row's byte)."""
    p = torch.zeros(16, dtype=torch.bool, device=device)
    for r in rows:
        p[r] = True
    return p


KERNEL_WEIGHTS = (
    "w_o",
    "attn_alpha",
    "attn_beta",
    "w_preffn",
    "w_router",
    "bias",
    "w_pool",
    "w_preexp",
    "w13",
    "w13s",
    "w2",
    "w2s",
    "w_postexp",
    "w_unpool",
)


def _launch(runner, w, core, stream, pad, key=None, *, begin=True):
    """One K2 step (the epoch bumped first): -> (stream_out, branch_out)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import layer

    S = core.shape[0]
    key = key or spec.K2Build(tokens=S, **runner.fault)
    stream_out = torch.empty_like(stream)
    branch_out = torch.empty(S, H, dtype=bf, device=stream.device)
    if begin:
        runner.step_begin()
    layer.k2_launch(
        key,
        {
            **{n: w[n] for n in KERNEL_WEIGHTS},
            "core": core,
            "stream": stream,
            "pad": pad,
            "stream_out": stream_out,
            "branch_out": branch_out,
            "slot": SLOT,
            **runner.launch_args(),
            **runner.peer_args(),
        },
    )
    return stream_out, branch_out


def _assert_healthy(runner) -> None:
    """No wait gave up, no alarm: FAILED and every CTA's record zero (the
    header's WAIT_BUDGET word holds the budget), the output rank's flags 0.
    Collective (``_tp_checks``): only the output rank's flags are ever set
    (``runner.alarm``), so the local verdicts differ by construction."""
    with _tp_checks("the health check"):
        h = runner.health.cpu().clone()
        assert h[FAILED] == 0, h[RECORD_WORDS:].view(-1, RECORD_WORDS).nonzero()
        h[WAIT_BUDGET] = 0
        assert not h.any()
        assert not runner.flags.any()


def _gather(x: torch.Tensor) -> list[torch.Tensor]:
    out = [torch.empty_like(x) for _ in range(TP)]
    dist.all_gather(out, x.contiguous())
    return out


def _all_ranks_equal(x: torch.Tensor, what: str) -> None:
    """Every rank gathers the same tensors, so the verdict is the same on every
    rank (no vote needed)."""
    outs = _gather(x)
    assert all(torch.equal(outs[0], o) for o in outs), f"ranks disagree on {what}"


def _stock_boundary(attn, stream, w):
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
        fused_norm_residual_norm_fwd,
    )

    ffn_in, stream_out = fused_norm_residual_norm_fwd(
        attn,
        stream,
        None,
        w["w_preffn"],
        spec.POST_ATTN_EPS,
        spec.PRE_FFN_EPS,
        alpha=w["attn_alpha"],
        beta=w["attn_beta"],
        post_gain_center=0.0,
        pre_gain_center=spec.GAIN,
        out_dtype=bf,
    )
    return ffn_in, stream_out


def _route_ids(ids: torch.Tensor, device) -> torch.Tensor:
    """The kernel's picks (``route`` readback, pads -1) as the stock MoE's
    ``[S 8, 1]`` int32 ids, a pad row's picks on expert 0 (only live rows are
    compared)."""
    flat = ids.to(device=device, dtype=torch.int32).clamp(min=0)
    return flat.reshape(-1, 1).contiguous()


def _stock_experts(ffn_in, flat_ids, tw, w, unit=None):
    """The routed experts vLLM runs, on ``ffn_in`` with the given picks
    (``flat_ids`` int32 [S 8, 1] on the device: the kernel's, so routing cannot
    differ): pooler, the expanded pre-expert norm, vLLM's AITER MoE op with the
    arguments ``rocm_aiter_fused_experts`` passes for Muse's MXFP4 w4a4 layers
    (one unit-weight top-1 route a row, as ``MuseSparkAiterExperts.apply``
    calls it), the rank-sum in fp32, the post-expert finalize.
    ``unit``: the cached ones buffer (``MuseSparkAiterExperts._unit_weights``),
    allocated here when None. -> (rows bf16 [S, 8, HM] of this rank, moe bf16
    [S, HM])."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        ActivationMethod,
        QuantMethod,
    )
    from vllm.models.muse_spark.amd.ops.fused_norm_expand import (
        fused_rmsnorm_expand,
        take_expanded,
    )
    from vllm.models.muse_spark.common.ops.fused_finalize import (
        fused_post_expert_finalize,
    )

    S = ffn_in.shape[0]
    xe = fused_rmsnorm_expand(
        _linear(ffn_in, w["w_pool"]),
        w["w_preexp"],
        spec.PRE_EXP_EPS,
        spec.GAIN,
        False,
        TOPK,
    )
    expanded = take_expanded(xe, TOPK)
    if unit is None:
        unit = torch.ones(S * TOPK, 1, dtype=f32, device=ffn_in.device)
    # rocm_aiter_fused_experts for Muse (MXFP4 w4a4, SiLU, no bias, no pads:
    # hidden_pad = intermediate_pad = 0, gate_mode "" = AITER's "separated")
    rows = rocm_aiter_ops.fused_moe(
        expanded,
        w["w13_fp4"],
        w["w2_fp4"],
        unit,
        flat_ids,
        activation_method=ActivationMethod.SILU.value,
        quant_method=QuantMethod.BLOCK_1X32.value,
        w1_scale=w["w13s"],
        w2_scale=w["w2s"],
        output_dtype=bf,
        moe_sorting_dispatch_policy=rocm_aiter_ops.get_moe_dispatch_policy(),
        swiglu_limit=0.0,
    ).view(S, TOPK, HM)
    R = tensor_model_parallel_all_reduce(rows.float())
    moe = torch.empty(S, HM, dtype=bf, device=ffn_in.device)
    fused_post_expert_finalize(
        R, w["w_postexp"], tw, spec.POST_EXP_EPS, spec.GAIN, True, moe
    )
    return rows, moe


def _eps_bites(ms: torch.Tensor, eps: float, other: float) -> torch.Tensor:
    """How much the norm scale ``rsqrt(m + eps)`` of rows with mean square
    ``ms`` moves when ``eps`` is swapped for ``other`` (a factor >= 1)."""
    ms = ms.double()
    hi, lo = max(eps, other), min(eps, other)
    return torch.sqrt((ms + hi) / (ms + lo))


def _assert_eps_rows_bite(attn_k, stream_out, rb, R, w) -> None:
    """The "eps" case's premise (design 6.2 (i)): its scaled rows sit where a
    swapped 1e-8 / 1e-5 eps moves the norm by O(1), so the tight per-stage
    checks would catch the swap -- the attention output's row 0, the pre-FFN
    norm's input of the last row, the pooled rows, the forced expert's routes.
    A calibration that misses this makes the case vacuous, so it fails here
    instead of passing silently."""
    big, small = spec.PRE_FFN_EPS, spec.POST_ATTN_EPS  # 1e-5, 1e-8
    a = attn_k[0].double().square().mean()
    assert _eps_bites(a, small, big).item() > 1.25, ("attn row 0", a.item())
    n = stream_out[-1].double().square().mean()
    assert _eps_bites(n, big, small).item() > 1.25, ("pre-FFN last row", n.item())
    pm = rb["pool"].double().square().mean(-1)
    assert (_eps_bites(pm, big, small) > 1.25).all(), ("pooled rows", pm)
    hit = (rb["ids"] == EPS_EXPERT).to(R.device)
    x = R.double() * (w["w_postexp"].double() + spec.GAIN)
    xm = x.square().mean(-1)[hit]
    assert xm.numel() > 0 and (_eps_bites(xm, small, big) > 1.25).all(), (
        "forced route",
        xm,
    )


def _check_step(runner, w, core, stream, pad, stream_out, branch_out, case) -> dict:
    """Every stage of the last launch against vLLM's ops on the inputs the
    kernel saw (design 6.2); returns the readback. The checks are rank-local
    (each rank its own shard) and the references collective, so every check
    block is voted (``_tp_checks``) before the next collective."""
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm import (
        rocm_fp32_router_gemm,
    )
    from vllm.models.muse_spark.common.ops.fused_finalize import (
        fused_post_expert_finalize,
    )
    from vllm.models.muse_spark.common.ops.fused_routing import (
        fused_musespark1x_routing,
    )

    dev = core.device
    rank = runner.rank
    S = core.shape[0]
    p = pad[:S].cpu()
    live = ~p
    lv = live.to(dev)
    where = f"{case} S={S}"
    _assert_healthy(runner)
    rb = readback(runner, S)
    tag = _tag(runner)
    # ---- o_proj: this rank's partial vs the stock GEMM; attn_out = the
    # partials in rank order (the seam's input), vs the stock all-reduce
    part = _gemm(core, w["w_o"])
    with _tp_checks(f"{where}: tags, o_proj"):
        for name, t in rb["tags"].items():
            assert (t == tag).all(), (case, S, name, t[t != tag][:4])
        # pad rows: the kernel pushes zeros (load_core selects them), stock
        # computes them
        ref.check_gemv(
            rb["ar"][rank][live], part[lv], core[lv], w["w_o"], what=f"{case} o_proj"
        )
        assert (rb["ar"][:, p].float() == 0).all(), case
    attn_k = ref.attn_out(list(rb["ar"])).to(dev)
    attn_stock = tensor_model_parallel_all_reduce(part.clone())
    _all_ranks_equal(rb["ar"].to(dev), "ar")
    with _tp_checks(f"{where}: boundary, router, routing, pooler"):
        assert ref.cos(attn_k[lv], attn_stock[lv]) > COS_STAGE
        # the seam's sums of squares: the 64 slices' partials
        ss = attn_k.double().square().sum(-1).cpu()
        torch.testing.assert_close(
            rb["ass"].double().sum(0)[live], ss[live], rtol=1e-5, atol=0
        )
        # ---- boundary on the kernel's own attn_out
        ffn_ref, stream_ref = _stock_boundary(attn_k, stream, w)
        got_s, ref_s = stream_out.cpu(), stream_ref.cpu()
        x = attn_k.float().cpu()
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + spec.POST_ATTN_EPS)
        mag = (w["attn_alpha"].cpu() * stream.cpu()).abs() + (
            w["attn_beta"].cpu() * y
        ).abs()
        err = (got_s - ref_s).abs()
        assert (err[live] <= 1e-5 * mag[live] + 1e-30).all(), (case, err[live].max())
        assert (got_s[p] == 0).all()
        ns = got_s.double()
        torch.testing.assert_close(
            rb["bss"].double().sum(0)[live],
            ns.square().sum(-1)[live],
            rtol=1e-5,
            atol=0,
        )
        rs = torch.rsqrt(ns.square().mean(-1, keepdim=True) + spec.PRE_FFN_EPS)
        own = (ns * rs * (w["w_preffn"].cpu().double() + spec.GAIN)).float().to(bf)
        assert ref.bf16_ulps(rb["xrow"][live], own[live]).max() <= 1, case
        u = ref.bf16_ulps(rb["xrow"][live], ffn_ref.cpu()[live])
        assert (u == 0).float().mean().item() >= 0.999, case
        big = ref_s.abs() >= 1e-3 * mag
        assert (u[big[live]] <= 1).all(), case
        assert (rb["xrow"][p].float() == 0).all()
        # chain guard: against the stock all-reduce's boundary
        _, stream_stock = _stock_boundary(attn_stock, stream, w)
        assert ref.cos(got_s[live], stream_stock.cpu()[live]) > COS_STAGE
        # ---- router logits on the kernel's own ffn_in: the README
        # tolerance, except where the two fp32 orders differ only by
        # accumulation noise on a cancelling dot
        xrow = rb["xrow"].to(dev)
        logits = rocm_fp32_router_gemm(xrow, w["w_router"]).cpu()
        absdot = rb["xrow"].double().abs() @ w["w_router"].double().abs().T.cpu()
        close = torch.isclose(rb["rlog"], logits, rtol=1e-5, atol=1e-6)
        noise = (rb["rlog"].double() - logits.double()).abs() <= 2**-20 * absdot
        assert (close | noise)[live].all(), (case, "router")
        assert close[live].float().mean().item() >= 0.99, (case, "router")
        # ---- routing on the kernel's own logits
        tw_s, ids_s = fused_musespark1x_routing(
            rb["rlog"].to(dev),
            w["bias"],
            TOPK,
            spec.ROUTE_SCALE,
            router_score_division_eps=spec.ROUTE_EPS,
            is_padding=pad[:S].contiguous(),
        )
        assert torch.equal(rb["ids"], ids_s.cpu()), case
        assert torch.equal(rb["tw"], tw_s.cpu().float()), case
        if case == "eps":
            assert (rb["ids"][live] == EPS_EXPERT).any(-1).all()
        # ---- pooler on the kernel's own ffn_in
        pooled = _linear(xrow, w["w_pool"]).cpu()
        ref.check_gemv(
            rb["pool"][live],
            pooled[live],
            rb["xrow"][live].float(),
            w["w_pool"].cpu(),
            what=f"{case} pool",
        )
        assert ref.cos(rb["pool"][live], pooled[live]) > COS_STAGE
    # ---- the owner reduce: R = the 8 ranks' rows of this rank's columns in
    # rank order (fp32, never rounded); the finalize on the gathered R
    R_own = ref.owner_sum(list(rb["rsx"].float()))  # [S, 8, 512]
    R = torch.cat(_gather(R_own.to(dev)), -1)  # [S, 8, HM], rank r's 512 r ..
    with _tp_checks(f"{where}: owner reduce, finalize"):
        sq = ref.owner_sumsq(R.cpu(), w["w_postexp"].cpu())
        for src in range(TP):
            lo = src * spec.OWN_COLS
            part_sq = ref.owner_sumsq(
                R[..., lo : lo + spec.OWN_COLS].cpu(),
                w["w_postexp"][lo : lo + spec.OWN_COLS].cpu(),
            )
            torch.testing.assert_close(
                rb["sqx"][src][live], part_sq[live], rtol=1e-5, atol=1e-30
            )
        torch.testing.assert_close(
            rb["sqx"].double().sum(0)[live], sq.double()[live], rtol=1e-5, atol=1e-30
        )
        # this rank's 32 owner slices (16 columns each), as every rank got them
        for jj in range(spec.RED_TASKS):
            c0 = rank * spec.OWN_COLS + jj * spec.ROWS
            torch.testing.assert_close(
                rb["sqx_part"][rank][jj][live],
                ref.owner_sumsq(
                    R[..., c0 : c0 + spec.ROWS].cpu(),
                    w["w_postexp"][c0 : c0 + spec.ROWS].cpu(),
                ),
                rtol=1e-5,
                atol=1e-30,
            )
        moe_ref = torch.empty(S, HM, dtype=bf, device=dev)
        fused_post_expert_finalize(
            R,
            w["w_postexp"],
            rb["tw"].to(dev),
            spec.POST_EXP_EPS,
            spec.GAIN,
            True,
            moe_ref,
        )
        ref.check_ulps(rb["moe"][live], moe_ref.cpu()[live], what=f"{case} finalize")
        assert (rb["moe"][p].float() == 0).all()
        if case == "eps":
            _assert_eps_rows_bite(attn_k, got_s, rb, R, w)
    # ---- the experts' chain (stock MoE on the kernel's ffn_in and picks)
    rows_s, moe_s = _stock_experts(
        xrow, _route_ids(rb["ids"], dev), rb["tw"].to(dev), w
    )
    mine = rb["rsx"].to(dev)  # [src, t, k, this rank's 512]: src's rows
    # every rank's stock rows (its expert shard), then this rank's columns:
    # rsx[src] holds src's rows of the columns THIS rank owns
    lo = rank * spec.OWN_COLS
    rows_all = torch.stack(_gather(rows_s.contiguous()))[..., lo : lo + spec.OWN_COLS]
    moe_k = rb["moe"].to(dev)
    with _tp_checks(f"{where}: experts, unpooler"):
        assert ref.cos(mine[:, lv], rows_all[:, lv]) > COS_CHAIN, case
        assert ref.cos(moe_k[lv], moe_s[lv]) > COS_CHAIN, case
        assert ref.rel_l2(moe_k[lv], moe_s[lv]) < REL_CHAIN, case
        # ---- unpooler on the kernel's own moe; gather = the branch rows
        branch = _linear(moe_k, w["w_unpool"]).cpu()
        ref.check_gemv(
            branch_out.cpu()[live],
            branch[live],
            rb["moe"][live].float(),
            w["w_unpool"].cpu(),
            what=f"{case} unpool",
        )
        assert torch.equal(branch_out.cpu(), rb["branch"]), case
        assert (branch_out.cpu()[p].float() == 0).all()
        branch_s = _linear(moe_s, w["w_unpool"])
        assert ref.cos(branch_out[lv], branch_s[lv]) > COS_CHAIN, case
        assert ref.rel_l2(branch_out[lv], branch_s[lv]) < REL_CHAIN, case
    # ---- every rank the same rows
    for name, t in (
        ("stream_out", stream_out),
        ("branch_out", branch_out),
        ("xrow", xrow),
        ("route", torch.cat([rb["ids"], rb["tw"].view(torch.int32)], -1).to(dev)),
        ("moe", moe_k),
    ):
        _all_ranks_equal(t, name)
    return rb


def _stock_span(w, core, stream, pad, out: dict, unit: torch.Tensor) -> None:
    """The decode vLLM runs from o_proj's input to the unpooler's output (design
    7.0): o_proj + all-reduce, the boundary, the router GEMM and routing, the
    pooler, the expanded pre-expert norm, aiter's MoE, the fp32 all-reduce,
    the finalize, the unpooler -- and nothing else: the routing's ids go to the
    MoE as a view and ``unit`` is the experts' cached ones buffer, as in the
    served layer. Writes ``out["stream"]``, ``out["branch"]``."""
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm import (
        rocm_fp32_router_gemm,
    )
    from vllm.models.muse_spark.common.ops.fused_routing import (
        fused_musespark1x_routing,
    )

    S = core.shape[0]
    attn = tensor_model_parallel_all_reduce(_gemm(core, w["w_o"]))
    ffn_in, stream_out = _stock_boundary(attn, stream, w)
    logits = rocm_fp32_router_gemm(ffn_in, w["w_router"])
    tw, ids = fused_musespark1x_routing(
        logits,
        w["bias"],
        TOPK,
        spec.ROUTE_SCALE,
        router_score_division_eps=spec.ROUTE_EPS,
        is_padding=pad[:S],
    )
    _, moe = _stock_experts(ffn_in, ids.view(S * TOPK, 1), tw, w, unit)
    out["stream"] = stream_out
    out["branch"] = _linear(moe, w["w_unpool"])


@ray.remote(num_gpus=1, max_calls=1)
def _k2_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # the first launch of each build compiles it: the startup budget
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    for case in CASES:
        w = _weights(rank, device, case)
        # every served width (a partial, guarded second round of load_xq's
        # groups and load_inter's ISC codes runs only at S = 5 .. 7); the eps
        # and zero_w premises at both ends
        for S in WIDTHS if case == "base" else (1, MT):
            core, stream = _inputs(rank, device, S, case)
            pad = _pad(device)
            stream_out, branch_out = _launch(runner, w, core, stream, pad)
            torch.accelerator.synchronize()
            _check_step(runner, w, core, stream, pad, stream_out, branch_out, case)

    # Pad rows (NaN / Inf inputs) come out as exact zeros and leave the live
    # rows unchanged
    w = _weights(rank, device)
    core, stream = _inputs(rank, device, 8)
    pad = _pad(device, (5, 6, 7))
    clean_s, clean_b = _launch(runner, w, core, stream, pad)
    torch.accelerator.synchronize()
    _assert_healthy(runner)
    _check_step(runner, w, core, stream, pad, clean_s, clean_b, "pad")
    dirty_c, dirty_s = core.clone(), stream.clone()
    dirty_c[5:] = float("nan")
    dirty_s[5:] = float("nan")
    dirty_c[6, 3] = float("inf")
    got_s, got_b = _launch(runner, w, dirty_c, dirty_s, pad)
    torch.accelerator.synchronize()
    _assert_healthy(runner)
    with _tp_checks("the NaN pad rows"):
        assert (got_s[5:] == 0).all() and (got_b[5:].float() == 0).all()
        assert torch.equal(got_s[:5], clean_s[:5])
        assert torch.equal(got_b[:5], clean_b[:5])
        rb = readback(runner, 8)
        assert (rb["ids"][5:] == -1).all() and (rb["tw"][5:] == 0).all()
        assert (rb["moe"][5:].float() == 0).all()
    # every row a pad row (a capture warm-up's step): nothing in the union,
    # the launch completes with zero outputs
    for S in (1, 8):
        core, stream = _inputs(rank, device, S)
        all_pad = torch.ones(16, dtype=torch.bool, device=device)
        s_out, b_out = _launch(runner, w, core, stream, all_pad)
        torch.accelerator.synchronize()
        _assert_healthy(runner)
        with _tp_checks(f"the all-pad step S={S}"):
            assert (s_out == 0).all() and (b_out.float() == 0).all(), S

    # One CUDA graph of two steps at two widths (each step's epoch bump
    # captured with it), replayed on two input sets written in place in turn:
    # every replay matches the eager step on its own set. A wait that took a
    # stale tag of the same parity (two epochs back: the previous replay's)
    # would read the other set's rows, which identical inputs would hide.
    sets = [{S: _inputs(rank, device, S, seed=k) for S in (1, MT)} for k in (0, 1)]
    pad = _pad(device)
    # the eager references (and the warm-ups: code objects loaded)
    want = [{S: _launch(runner, w, *sets[k][S], pad) for S in (1, MT)} for k in (0, 1)]
    ins = {S: tuple(t.clone() for t in sets[0][S]) for S in (1, MT)}
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    outs = {}
    with torch.cuda.graph(graph):
        for S in (1, MT):
            outs[S] = _launch(runner, w, *ins[S], pad)
    bad = torch.zeros(REPLAYS, 2, 2, dtype=torch.bool, device=device)
    for i in range(REPLAYS):
        k = i % 2
        for S in (1, MT):
            for dst, src in zip(ins[S], sets[k][S]):
                dst.copy_(src)
        graph.replay()
        for j, S in enumerate((1, MT)):
            for o, (got, ref_out) in enumerate(zip(outs[S], want[k][S])):
                bad[i, j, o] = (got != ref_out).any()
    torch.accelerator.synchronize()
    with _tp_checks("the graph replays"):
        # (replay, width index, output: stream_out 0 / branch_out 1)
        assert not bad.any(), bad.nonzero()[:8].tolist()
    _assert_healthy(runner)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_tp8_matches_vllm_ops_and_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _fault_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.mono.common.debug import (
        given_up_waits,
    )
    from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (
        WAIT_TICKS,
    )
    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # rank STALLED stalls 1.5 serving budgets in the epoch-2 launch of slot
    # SLOT (the epoch-1 launch compiles the build: the startup budget)
    runner.fault = {
        "fault_rank": STALLED,
        "fault_epoch": 2,
        "fault_slot": SLOT,
        "fault_ticks": WAIT_TICKS * 3 // 2,
    }
    w = _weights(rank, device)
    core, stream = _inputs(rank, device, 8)
    pad = _pad(device)
    _launch(runner, w, core, stream, pad)  # epoch 1: no stall
    torch.accelerator.synchronize()
    _assert_healthy(runner)
    runner.set_wait_budget(WAIT_SECONDS)
    _launch(runner, w, core, stream, pad)  # epoch 2: every launch returns
    torch.accelerator.synchronize()
    waits = given_up_waits(runner.health.cpu(), spec.REGIONS)
    gave_up = torch.tensor([int(bool(waits))], device=device)
    dist.all_reduce(gave_up)
    assert gave_up.item() > 0, "no rank's wait gave up"
    if rank == 0:
        assert runner.flags.any(), "the output rank's flags missed the failure"


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_stalled_rank_fails_the_step_without_a_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _fault_worker)
