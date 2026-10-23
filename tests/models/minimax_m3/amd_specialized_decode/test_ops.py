# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Op-level tests of the fused FFN launch (TP=1 builds unless noted) against
vLLM's and AITER's native ops, with MXFP4 weights from vLLM's real post-load
path: stage parity, routing ties, padding, residency, graph replay, fail-stop
sites, epoch reset, the gfx950 hand-off ISA, and the hand-off stresses (one-GPU
cross-XCD; TP4 over XGMI, optional)."""

import glob
import os
import shutil
import subprocess
import tempfile
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 only", allow_module_level=True)

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402

from tests.models.minimax_m3.amd_specialized_decode import (  # noqa: E402
    _xcd_stress as xs,
)
from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (  # noqa: E402
    gemma_fused_add_rmsnorm,
)
from vllm.models.minimax_m3.amd.specialized_decode.mono import (  # noqa: E402
    ffn,
    sync,
)
from vllm.models.minimax_m3.amd.specialized_decode.mono.residency import (  # noqa: E402
    code_object,
)
from vllm.models.minimax_m3.amd.specialized_decode.mono.runner import (  # noqa: E402
    FfnRunner,
    _writable_flydsl_cache,
)

E = ffn.EXPERTS + 1
H, INTER = ffn.HIDDEN, ffn.INTER
SHORT_WAIT = 10_000_000  # 100 ms of the 100 MHz clock


@pytest.mark.parametrize("query_lens", [(64, 3), (129, 65)])
@pytest.mark.parametrize("quantized_query", [False, True])
def test_layer_dense_prefill_preserves_cache_and_query_scales(
    query_lens, quantized_query
):
    """Cached mixed prefills honor paged addressing and non-unit Q/K/V scales."""
    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionImpl,
        TritonAttentionMetadata,
    )

    dev = torch.device("cuda")
    generator = torch.Generator(device=dev).manual_seed(19)
    lengths = (257, 513)
    cache = torch.randn(8, 1, 128, 256, device=dev, generator=generator).to(
        torch.float8_e4m3fn
    )
    table = torch.tensor([[6, 2, 7, 0, 0], [5, 1, 4, 0, 3]], device=dev)
    query = torch.randn(
        sum(query_lens), 16, 128, device=dev, generator=generator
    ).bfloat16()
    layer = torch.nn.Module()
    for name, scale in (("q", 0.75), ("k", 0.5), ("v", 0.25)):
        layer.register_buffer(f"_{name}_scale", torch.tensor(scale, device=dev))
    if quantized_query:
        query = query.to(torch.float8_e4m3fn)
    output = torch.empty(query.shape, dtype=torch.bfloat16, device=dev)
    metadata = TritonAttentionMetadata(
        num_actual_tokens=query.shape[0],
        max_query_len=max(query_lens),
        query_start_loc=torch.tensor(
            [0, query_lens[0], sum(query_lens)], device=dev, dtype=torch.int32
        ),
        max_seq_len=max(lengths),
        seq_lens=torch.tensor(lengths, device=dev, dtype=torch.int32),
        block_table=table.to(torch.int32),
        slot_mapping=torch.empty(query.shape[0], dtype=torch.int64, device=dev),
        seq_threshold_3D=256,
        num_par_softmax_segments=256,
        softmax_segm_output=torch.empty(0, device=dev),
        softmax_segm_max=torch.empty(0, device=dev),
        softmax_segm_expsum=torch.empty(0, device=dev),
        causal=True,
        use_aiter_prefill=True,
    )
    impl = TritonAttentionImpl(16, 128, 128**-0.5, 1, None, None, "fp8")
    impl.forward(layer, query, None, None, cache, metadata, output)
    offset = 0
    for seq, (query_len, length) in enumerate(zip(query_lens, lengths)):
        pages = cache[table[seq, : (length + 127) // 128]].float()
        key, value = pages.reshape(-1, 256)[:length].split(128, dim=-1)
        q = query[offset : offset + query_len].float()
        if quantized_query:
            q = (q * layer._q_scale).bfloat16().float()
        scores = torch.einsum("qhd,kd->qhk", q, key * layer._k_scale) * 128**-0.5
        positions = length - query_len + torch.arange(query_len, device=dev)
        scores.masked_fill_(
            torch.arange(length, device=dev)[None, None, :] > positions[:, None, None],
            -torch.inf,
        )
        reference = scores.softmax(-1) @ (value * layer._v_scale)
        actual = output[offset : offset + query_len].float()
        assert actual.isfinite().all()
        assert (actual - reference).norm() / reference.norm() < 0.01
        assert (
            torch.nn.functional.cosine_similarity(
                actual.flatten(), reference.flatten(), dim=0
            )
            > 0.9999
        )
        offset += query_len


def _metrics(got, want):
    got, want = got.float().flatten(), want.float().flatten()
    cos = torch.nn.functional.cosine_similarity(got, want, dim=0).item()
    rel = ((got - want).norm() / want.norm()).item()
    return cos, rel


@pytest.fixture(scope="module")
def weights():
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    )

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)

    def codes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev, generator=g)

    def scales(*shape):
        # e8m0 codes around 2^-7: gate/up pre-activations of a few units
        return torch.randint(
            119, 123, shape, dtype=torch.uint8, device=dev, generator=g
        )

    def param(t):
        return torch.nn.Parameter(t, requires_grad=False)

    w13, w2, s13, s2, _, _ = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
        layer=torch.nn.Module(),
        w13_weight=param(codes(E, 2 * INTER, H // 2)),
        w2_weight=param(codes(E, H, INTER // 2)),
        w13_weight_scale=param(scales(E, 2 * INTER, H // 32)),
        w2_weight_scale=param(scales(E, H, INTER // 32)),
    )
    return SimpleNamespace(
        w13=w13,
        w2=w2,
        s13=s13,
        s2=s2,
        post_norm=torch.randn(H, device=dev, generator=g).mul(0.1).bfloat16(),
        router=torch.randn(ffn.EXPERTS, H, device=dev, generator=g).mul(0.02),
        router_bias=torch.randn(ffn.EXPERTS, device=dev, generator=g).mul(0.01),
    )


def _native_experts(x, w, ids, wts):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    return fused_moe(
        x,
        w.w13,
        w.w2,
        wts,
        ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=w.s13,
        w2_scale=w.s2,
        gate_mode="separated",
        swiglu_limit=ffn.LIMIT,
    )


def _native_route(logits, bias):
    from aiter import biased_grouped_topk

    s = logits.shape[0]
    ids = torch.empty(s, ffn.TOPK, dtype=torch.int32, device=logits.device)
    wts = torch.empty(s, ffn.TOPK, dtype=torch.float32, device=logits.device)
    biased_grouped_topk(logits, bias, wts, ids, 1, 1, True, ffn.ROUTED_SCALE)
    return ids, wts


def _clear_choices(logits, bias, margin):
    """Rows whose 4th and 5th biased scores differ by more than ``margin``."""
    top = (logits.sigmoid() + bias).topk(ffn.TOPK + 1, dim=-1).values
    return (top[:, ffn.TOPK - 1] - top[:, ffn.TOPK]) > margin


def _pairs(region, n):
    return region[: n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()


def _runner(weights, s, tp=1, widths=None, **build):
    dev = torch.device("cuda")
    bufs = [torch.zeros(ffn.peer_bytes(tp), dtype=torch.uint8, device=dev)]
    bufs += [torch.zeros_like(bufs[0]) for _ in range(tp - 1)]
    peers = torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64, device=dev)
    runner = FfnRunner(tp, 0, bufs[0].data_ptr(), peers, dev, widths or (s,), **build)
    runner.warm(weights)
    runner.peer_bufs = bufs
    return runner


def _error(buf):
    return buf[: 4 * ffn.ERROR_WORDS].view(torch.int32).tolist()


def _inputs(s, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    part = torch.randn(s, H, device="cuda", generator=g).bfloat16()
    residual = torch.randn(s, H, device="cuda", generator=g).mul(4).bfloat16()
    return part, residual


def _slots(s):
    return torch.arange(s, dtype=torch.int64, device="cuda")


def _run(weights, s, seed=1, inputs=None, slots=None):
    part, residual = _inputs(s, seed) if inputs is None else inputs
    runner = _runner(weights, s)
    out, res_out = runner(
        part, residual, weights, _slots(s) if slots is None else slots
    )
    torch.accelerator.synchronize()
    assert _error(runner.peer_bufs[0])[0] == 0
    x = runner.region(s, "xn")[: s * H * 2].view(torch.bfloat16).view(s, H)
    parts = _pairs(runner.region(s, "logit"), s * ffn.EXPERTS * ffn.ROUTER_PARTS)
    parts = parts.view(torch.float32).view(s, ffn.EXPERTS, ffn.ROUTER_PARTS)
    route = _pairs(runner.region(s, "route"), s * 2 * ffn.TOPK).view(s, 2 * ffn.TOPK)
    return SimpleNamespace(
        part=part,
        residual=residual,
        out=out,
        res_out=res_out,
        x=x,
        logits=parts.sum(-1),
        ids=route[:, : ffn.TOPK].contiguous(),
        wts=route[:, ffn.TOPK :].contiguous().view(torch.float32),
    )


@pytest.mark.parametrize("s", [1, 3, 16])
def test_ffn_stages_match_native(weights, s):
    """Seam residual (exact), Gemma norm, FP32 router, top-4 and the shared +
    routed A16W4 experts, each against its native op."""
    r = _run(weights, s)
    want_x, want_res = gemma_fused_add_rmsnorm(
        r.part, r.residual, weights.post_norm, ffn.EPS
    )
    torch.testing.assert_close(r.res_out, want_res, rtol=0, atol=0)
    cos, rel = _metrics(r.x, want_x)
    assert cos > 0.99999 and rel < 4e-3

    want_logits = r.x.float() @ weights.router.T
    torch.testing.assert_close(r.logits, want_logits, rtol=1e-4, atol=1e-4)

    ids, wts = _native_route(r.logits, weights.router_bias)
    clear = _clear_choices(r.logits, weights.router_bias, 1e-4)
    assert clear.any()
    got_ids, order = r.ids.sort(-1)
    want_ids, want_order = ids.sort(-1)
    assert torch.equal(got_ids[clear], want_ids[clear])
    torch.testing.assert_close(
        r.wts.gather(1, order)[clear],
        wts.gather(1, want_order)[clear],
        rtol=1e-5,
        atol=1e-6,
    )

    shared = torch.full((s, 1), ffn.SHARED, dtype=torch.int32, device=r.x.device)
    all_ids = torch.cat([r.ids, shared], 1)
    all_wts = torch.cat([r.wts, torch.ones_like(r.wts[:, :1])], 1)
    cos, rel = _metrics(r.out, _native_experts(r.x, weights, all_ids, all_wts))
    assert cos > 0.9999 and rel < 1e-2


def test_route_ties_and_near_ties(weights):
    """Experts 0 and 1 share a router row and bias: an exact 4th-place tie,
    which the kernel gives to the lower expert. Elsewhere the selection matches
    AITER's on native FP32 logits wherever the 4th/5th margin exceeds the
    logits' measured difference times the maximal sigmoid slope, twice."""
    w = SimpleNamespace(**vars(weights))
    w.router = weights.router.clone()
    w.router_bias = weights.router_bias.clone()
    w.router[1] = w.router[0]
    w.router_bias[:2] = 0.5
    w.router_bias[[10, 11, 12]] = 2.0
    w.router_bias[2:10] = -1.0
    w.router_bias[13:] = -1.0
    r = _run(w, 16)
    assert (r.ids[:, :3].sort(-1).values == torch.tensor([10, 11, 12]).cuda()).all()
    assert (r.ids[:, 3] == 0).all()

    r = _run(weights, 16, seed=3)
    logits = r.x.float() @ weights.router.T
    bound = (r.logits - logits).abs().max().item() * 0.25 * 2
    ids, _ = _native_route(logits, weights.router_bias)
    clear = _clear_choices(logits, weights.router_bias, bound)
    assert clear.float().mean() > 0.5
    assert torch.equal(r.ids.sort(-1).values[clear], ids.sort(-1).values[clear])


def test_padded_rows_are_inert(weights):
    """Rows whose slot is -1 (graph padding) hold NaN: the real rows match a
    launch without them bit for bit, the padded rows pick no expert and return
    zeros; an all-padded launch completes."""
    real, pad = 3, 5
    part, residual = _inputs(real)
    nan = torch.full((pad, H), float("nan"), device="cuda").bfloat16()
    slots = torch.tensor([7, 9, 11] + [-1] * pad, dtype=torch.int64, device="cuda")
    padded = _run(
        weights,
        real + pad,
        inputs=(torch.cat([part, nan]), torch.cat([residual, nan])),
        slots=slots,
    )
    alone = _run(weights, real, inputs=(part, residual))
    assert torch.equal(padded.out[:real], alone.out)
    assert torch.equal(padded.res_out[:real], alone.res_out)
    assert torch.equal(padded.ids[:real], alone.ids)
    assert (padded.ids[real:] == -1).all() and (padded.out[real:] == 0).all()

    r = _run(weights, 4, slots=torch.full((4,), -1, dtype=torch.int64, device="cuda"))
    assert (r.ids == -1).all() and (r.out == 0).all()


def test_every_width_is_resident(weights):
    """The driver's occupancy of every production (TP4) build: one 512-thread
    CTA per CU on each of the 256 CUs."""
    dev = torch.device("cuda")
    peer = torch.zeros(ffn.peer_bytes(4), dtype=torch.uint8, device=dev)
    peers = torch.tensor([peer.data_ptr()] * 4, dtype=torch.int64, device=dev)
    runner = FfnRunner(4, 0, peer.data_ptr(), peers, dev)
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    runner.warm(weights, compute_units=cus)
    assert sorted(runner.residency) == list(range(1, ffn.MAX_TOKENS + 1))
    assert all(n * cus >= ffn.BLOCKS for n in runner.residency.values())


def test_graph_replay_is_bit_stable(weights):
    """Widths captured once and replayed 300 rounds in alternation: each replay
    matches an eager launch bit for bit, also after the persistent slot buffer
    marks a different set of rows as padding."""
    widths = (1, 4, 16)
    runner = _runner(weights, 1, widths=widths)
    inputs = {s: _inputs(s, seed=s) for s in widths}
    slots = {s: _slots(s) for s in widths}

    def eager(s):
        out, res = runner(*inputs[s], weights, slots[s])
        return out.clone(), res.clone()

    graphs, outs = {}, {}
    for s in widths:
        eager(s)
        graphs[s] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[s]):
            outs[s] = runner(*inputs[s], weights, slots[s])
    refs = {s: eager(s) for s in widths}
    for _ in range(300):
        for s in widths:
            graphs[s].replay()
        for s in widths:
            assert torch.equal(outs[s][0], refs[s][0])
            assert torch.equal(outs[s][1], refs[s][1])

    for s in widths[1:]:
        slots[s][s // 2 :] = -1
        ref = eager(s)
        graphs[s].replay()
        assert torch.equal(outs[s][0], ref[0])
        assert (outs[s][0][s // 2 :] == 0).all()
    torch.accelerator.synchronize()
    assert _error(runner.peer_bufs[0])[0] == 0
    assert runner.epoch[0].item() > 300 * len(widths)


@pytest.mark.parametrize(
    "build, arm, final, want",
    [
        ({}, None, True, [0, 0, 0, 0]),
        ({"tp": 2}, None, False, [1, 0, 0, None]),
        ({"drop_mark": 7}, None, False, [1, 0, 0, ffn.SITE_EPOCH]),
        ({}, "inject", True, [1, 0, 0, ffn.SITE_INJECT]),
        ({}, "wrap", False, [1, 0, None, ffn.SITE_WRAP]),
    ],
    ids=["handshake", "missing_peer", "missing_epoch_mark", "late_fault", "wrap"],
)
def test_fail_stop_sites(weights, build, arm, final, want):
    """Every bounded wait and backstop sets the sticky error word [failed,
    rank, epoch, site] on every rank and the launch still completes; a clean
    step-end handshake sets nothing. A missing epoch mark does not advance the
    epoch."""
    runner = _runner(weights, 1, wait_ticks=SHORT_WAIT, **build)
    if arm == "inject":
        runner.inject.copy_(torch.tensor([1, 100_000], dtype=torch.int32))
    if arm == "wrap":
        runner.epoch[0] = ffn.EPOCH_LIMIT
    runner(*_inputs(1), weights, _slots(1), final=final)
    torch.accelerator.synchronize()
    for buf in runner.peer_bufs:
        got = _error(buf)
        assert [g if w is not None else None for g, w in zip(got, want)] == want
        # stages that start waiting together race to report their site
        assert got[3] > 0 or not got[0]
    if "drop_mark" in build:
        assert runner.epoch[0].item() == 0


def _reset_executor(runner, monkeypatch, agree):
    """A TP1 FfnExecutor over ``runner`` that resets every 3 output steps, its
    TP agreement ``agree``."""
    from vllm.models.minimax_m3.amd.specialized_decode.runtime import executor as ex

    monkeypatch.setattr(ex, "tp_agree", agree)
    executor = object.__new__(ex.FfnExecutor)
    executor.rank, executor.runner, executor.resets = 0, runner, 0
    executor.peer = SimpleNamespace(bytes=runner.peer_bufs[0], addresses=runner.peers)
    executor._layers = dict.fromkeys(range(3, 60))
    executor.outputs, executor._since_reset = 0, 0
    executor._reset_launches = 3 * len(executor._layers)
    executor._epoch_event = None
    executor._epoch_host = torch.zeros(1, dtype=torch.int32)
    return executor


def _broken(*_):
    raise RuntimeError("injected")


@pytest.mark.parametrize(
    "where, msg, marked",
    [
        ("clear", r"\(clear\) failed on this TP rank", True),
        ("agreement", r"\(drain\) failed on this TP rank", True),
        ("peer", r"\(drain\) failed on another TP rank", False),
    ],
)
def test_failed_epoch_reset_fails_every_rank(weights, monkeypatch, where, msg, marked):
    """A reset phase failing here (a clear or an agreement that raises) or on
    another rank raises on this rank and does not resume; the failing rank
    marks every rank's error word."""
    runner = _runner(weights, 1)
    runner(*_inputs(1), weights, _slots(1))
    torch.accelerator.synchronize()
    agree = {"agreement": _broken, "peer": lambda ok: False}.get(where, lambda ok: ok)
    executor = _reset_executor(runner, monkeypatch, agree)
    if where == "clear":
        monkeypatch.setattr(runner, "reset", _broken)
    epoch = runner.epoch[0].item()
    executor.output_step()
    executor.output_step()
    with pytest.raises(RuntimeError, match=msg):
        executor.output_step()
    torch.accelerator.synchronize()
    assert executor.resets == 0 and executor._since_reset == 3
    assert runner.epoch[0].item() == epoch
    assert _error(runner.peer_bufs[0]) == ([1, 0, -1, 0] if marked else [0] * 4)


def test_forced_epoch_reset_clears_and_resumes(weights, monkeypatch):
    """The executor's step hook resets every region and the epoch once the
    launches since the last reset reach its bound; launches then resume from
    tag one with the same results, the error word kept."""
    runner = _runner(weights, 2)
    inputs = _inputs(2)
    before = runner(*inputs, weights, _slots(2))[0].clone()
    executor = _reset_executor(runner, monkeypatch, lambda ok: ok)
    runner.peer_bufs[0][4] = 7  # an error-word field the reset must keep
    for _ in range(3):
        runner(*inputs, weights, _slots(2))
        executor.output_step()
    torch.accelerator.synchronize()
    assert executor.resets == 1 and runner.epoch[0].item() == 0
    assert runner.peer_bufs[0][4].item() == 7
    assert not runner.peer_bufs[0][4 * ffn.ERROR_WORDS :].any()
    assert not any(buf.any() for buf in runner.scratch.values())
    runner.peer_bufs[0][4] = 0
    after = runner(*inputs, weights, _slots(2))[0]
    torch.accelerator.synchronize()
    assert torch.equal(before, after) and _error(runner.peer_bufs[0])[0] == 0


OBJDUMP = shutil.which("llvm-objdump") or next(
    iter(
        glob.glob(
            "/usr/local/lib/python3.12/dist-packages/_rocm_sdk_*/lib/llvm/bin/llvm-objdump"
        )
    ),
    None,
)


def _disassemble(compiled):
    """The linked code object's instructions, each split into words."""
    if OBJDUMP is None:
        pytest.skip("needs llvm-objdump")
    assert OBJDUMP is not None
    elf = Path(tempfile.mkdtemp()) / "kernel.elf"
    elf.write_bytes(code_object(compiled._keepalive._ir_text))
    isa = subprocess.run(
        [OBJDUMP, "-d", "--mcpu=gfx950", str(elf)], capture_output=True, text=True
    ).stdout
    ops = [line.split("//")[0].split() for line in isa.splitlines() if "\t" in line]
    return [op for op in ops if op]


def test_isa_encodes_the_handoff_contract(weights):
    """The linked TP4 S=1 kernel has the shapes ``mono/sync.py`` argues from:
    every system-scope access one aligned 64-bit pair or the 32-bit error
    word; no buffer op carries cache bits; every global access (only hand-offs
    use them) carries ``sc1``, alone for agent scope, and every agent store /
    load form ``mono/sync.py`` uses is lowered that way; the handshake's
    system release is a write-back then a drain, its acquire a drain then an
    invalidate, and (performance policy) no other L2 maintenance remains."""
    dev = torch.device("cuda")
    peer = torch.zeros(ffn.peer_bytes(4), dtype=torch.uint8, device=dev)
    peers = torch.tensor([peer.data_ptr()] * 4, dtype=torch.int64, device=dev)
    runner = FfnRunner(4, 0, peer.data_ptr(), peers, dev, (1,))
    runner.warm(weights)
    ops = _disassemble(runner._compiled[1])

    system = [op[0] for op in ops if "sc0" in op and "sc1" in op]
    assert {o for o in system if "store" in o} == {"global_store_dwordx2"}
    assert {o for o in system if "load" in o} <= {
        "global_load_dwordx2",
        "global_load_dword",
    }
    data = [op for op in ops if op[0].startswith(("buffer_load", "buffer_store"))]
    assert data and not [op for op in data if "sc0" in op or "sc1" in op]

    accesses = [op for op in ops if op[0].startswith(("global_load", "global_store"))]
    assert accesses and all("sc1" in op for op in accesses)
    assert {
        "global_store_dword",
        "global_store_dwordx2",
        "global_store_short",
        "global_load_dword",
        "global_load_dwordx2",
        "global_load_dwordx4",
    } <= {op[0] for op in accesses if "sc0" not in op}

    for i, op in enumerate(ops):
        if op[0] == "buffer_wbl2":
            assert ops[i + 1][:2] == ["s_waitcnt", "vmcnt(0)"]
        if op[0] == "buffer_inv":
            assert ops[i - 1][0] == "s_waitcnt" and "vmcnt(0)" in ops[i - 1][1]
    kinds = {" ".join(op) for op in ops if op[0] in ("buffer_wbl2", "buffer_inv")}
    assert kinds == {"buffer_wbl2 sc0 sc1", "buffer_inv sc0 sc1"}


_STREAM = fx.Stream(None)


@flyc.kernel(name="handoff_probe", known_block_size=[64, 1, 1])
def _handoff_probe(data: fx.Int64, pair: fx.Int64, out: fx.Int64, v: fx.Int32):
    tid = fx.Int32(fx.thread_idx.x)
    at = data + fx.Int64(tid) * 64
    sync.store_dev(at, 0, [v])
    sync.store_dev(at, 2, [v, tid])
    sync.store16(at + 16, 0, fx.Float32(v).to(fx.BFloat16), "agent")
    sync.sync_release("agent")
    sync.store64(pair, tid, sync.pack(v, tid), "agent")
    got = sync.load64(pair, 63 - tid, "agent")
    sync.sync_acquire("agent")
    back = data + fx.Int64(63 - tid) * 64
    a = sync.load_dev(back, 0)
    b = sync.load_dev(back, 2, 2)
    c = sync.load_dev(back, 4, 4)
    sync.store32(out, tid, a + b[1] + c[3] + fx.Int32(got), "agent")


@flyc.jit
def _launch_handoff_probe(
    data: fx.Int64,
    pair: fx.Int64,
    out: fx.Int64,
    v: fx.Int32,
    stream: fx.Stream = _STREAM,
):
    _handoff_probe(data, pair, out, v).launch(grid=(1,), block=(64,), stream=stream)


def test_isa_orders_handoff_data_around_av_none_fences():
    """The agent ``av none`` release / acquire on gfx950: a completion wait,
    not an L2 write-back / invalidate, between the data stores and the pair
    store, and between the pair load and the data loads (the data's addresses
    do not depend on the pair, so only the acquire orders them)."""
    buf = torch.zeros(1 << 14, dtype=torch.uint8, device="cuda")
    base = buf.data_ptr()
    stream = torch.cuda.current_stream()
    ops = _disassemble(
        flyc.compile(_launch_handoff_probe, base, base + 8192, base + 12288, 7, stream)
    )
    stores = [i for i, op in enumerate(ops) if op[0].startswith("global_store")]
    loads = [i for i, op in enumerate(ops) if op[0].startswith("global_load")]
    assert all("sc1" in ops[i] and "sc0" not in ops[i] for i in stores + loads)
    *data_stores, pair_store, _out = stores
    pair_load, *data_loads = loads
    assert ops[pair_store][0] == "global_store_dwordx2"
    assert ops[pair_load][0] == "global_load_dwordx2"
    assert {ops[i][0] for i in data_loads} == {
        "global_load_dword",
        "global_load_dwordx2",
        "global_load_dwordx4",
    }

    def drained(lo, hi):
        return any(
            ops[i][0] == "s_waitcnt" and "vmcnt(0)" in ops[i][1] for i in range(lo, hi)
        )

    assert max(data_stores) < pair_store < pair_load < min(data_loads)
    assert drained(max(data_stores) + 1, pair_store)
    assert drained(pair_load + 1, min(data_loads))
    assert not [op for op in ops if op[0] in ("buffer_wbl2", "buffer_inv")]


XCD_LAUNCHES = int(os.environ.get("MINIMAX_M3_XCD_STRESS_LAUNCHES", "200"))


def test_handoffs_between_xcds():
    """The same-GPU hand-off protocol under the adversarial one-GPU stress
    (``_xcd_stress.py``): every hand-off between CTAs on different XCDs
    delivers the producer's current payload, and no wait times out."""
    _writable_flydsl_cache()
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    assert cus >= xs.BLOCKS, "every CTA must be resident"
    err, out = xs.run(XCD_LAUNCHES)
    assert err[0].item() == 0, err.tolist()
    bad = out[:, 2]
    assert bad.sum().item() == 0, out[bad.nonzero().flatten()][:8].tolist()

    xcc = out[:, 0]
    assert sorted(set(xcc.tolist())) == list(range(8))
    # every round's producer / consumer pair on two XCDs
    blocks = torch.arange(xs.BLOCKS)
    for k in range(1, 128, 2):
        assert (xcc != xcc[(blocks + xs.BLOCKS - k) % xs.BLOCKS]).all(), k


@pytest.mark.parametrize("stores, loads", [(False, True), (True, False)])
def test_plain_payload_accesses_are_caught(stores, loads):
    """The stress has teeth: with either side of the payload a plain cached
    access, consumers read stale data."""
    _writable_flydsl_cache()
    err, out = xs.run(4, stores, loads, stop_on_error=False)
    assert err[0].item() == 0, err.tolist()
    assert (out[:, 2] > 0).any()


STRESS_LAUNCHES = int(os.environ.get("MINIMAX_M3_STRESS_LAUNCHES", "120000"))
STRESS_WIDTHS = (1, 4, 16)
MAX_SKEW_CYCLES = 200_000  # ~0.1 ms at the shader clock
RESET_EVERY = 997  # launches between collective epoch resets


def _stress_rank(rank, port, results):
    from vllm.models.deepseek_v41.amd.mono.common import peer_memory
    from vllm.models.minimax_m3.amd.specialized_decode.runtime import executor as ex

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4
    )
    dev = torch.device("cuda", rank)
    w = weights.__wrapped__()
    peer = peer_memory.PeerBuffer(ffn.peer_bytes(4), dist.group.WORLD, rank, 4, dev)
    peer.bytes.zero_()
    runner = FfnRunner(4, rank, peer.local, peer.addresses, dev, STRESS_WIDTHS)
    runner.warm(w)
    torch.accelerator.synchronize()
    dist.barrier()

    def inputs(s, which):
        g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"{s}/{which}".encode()))
        residual = torch.randn(s, H, device=dev, generator=g).bfloat16()
        g.manual_seed(zlib.crc32(f"{s}/{which}/{rank}".encode()))
        part = torch.randn(s, H, device=dev, generator=g).bfloat16()
        return part, residual, torch.arange(s, dtype=torch.int64, device=dev)

    cases, graphs, outs, refs, keep = [], {}, {}, {}, {}
    for s in STRESS_WIDTHS:
        for which in (0, 1):
            case = (s, which)
            cases.append(case)
            # graphs read their inputs by address: they must outlive them
            keep[case] = part, residual, slots = inputs(s, which)
            final = which == 1
            refs[case] = runner(part, residual, w, slots, final)[0].clone()
            graphs[case] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[case]):
                outs[case] = runner(part, residual, w, slots, final)[0]
    torch.accelerator.synchronize()
    dist.barrier()

    # the executor's step hook, its epoch reset agreed over this group
    ex.tp_agree = lambda ok: peer_memory.tp_agree(ok, dist.group.WORLD)
    executor = object.__new__(ex.FfnExecutor)
    executor.rank, executor.runner, executor.peer = rank, runner, peer
    executor._layers = {0: None}
    executor.outputs = executor._since_reset = executor.resets = 0
    executor._reset_launches = RESET_EVERY
    executor._epoch_event = None
    executor._epoch_host = torch.zeros(1, dtype=torch.int32)

    order = torch.Generator().manual_seed(1234)
    skew = torch.Generator().manual_seed(rank)
    mismatches = torch.zeros((), dtype=torch.int64, device=dev)
    for n in range(STRESS_LAUNCHES):
        case = cases[int(torch.randint(len(cases), (1,), generator=order))]
        torch.cuda._sleep(int(torch.randint(MAX_SKEW_CYCLES, (1,), generator=skew)))
        graphs[case].replay()
        mismatches += (outs[case] != refs[case]).any()
        executor.output_step()
        if n % 4096 == 4095:
            torch.accelerator.synchronize()
    torch.accelerator.synchronize()

    outs_cpu = torch.cat([refs[c].flatten() for c in cases]).view(torch.int32).cpu()
    gathered = [torch.zeros_like(outs_cpu) for _ in range(4)]
    dist.all_gather(gathered, outs_cpu)
    results[rank] = SimpleNamespace(
        mismatches=int(mismatches.item()),
        error=_error(peer.bytes),
        epoch=int(runner.epoch[0].item()),
        resets=executor.resets,
        same_across_ranks=all(torch.equal(gathered[0], g) for g in gathered),
    )
    dist.barrier()
    graphs.clear()
    peer.close()
    dist.destroy_process_group()


@pytest.mark.optional
@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="needs 4 GPUs")
def test_peer_handoffs_under_rank_skew():
    """TP4 over XGMI: four ranks replay graphs of three widths, alternating two
    payload sets (half ending in the step-end fault handshake), with random
    per-rank device delays before every launch and the executor's collective
    epoch reset every ``RESET_EVERY`` launches. Every launch reproduces its
    first result bit for bit, the reduced output matches across ranks, and no
    wait times out."""
    from vllm.utils.network_utils import get_open_port

    timeout = 600 + STRESS_LAUNCHES // 1000
    results = mp.Manager().dict()
    ranks = mp.start_processes(
        _stress_rank, args=(get_open_port(), results), nprocs=4, join=False
    )
    deadline = time.monotonic() + timeout
    while not ranks.join(timeout=10):
        if time.monotonic() > deadline:
            for p in ranks.processes:
                p.kill()
            pytest.fail(f"TP4 stress ranks still running after {timeout} s")
    for rank in range(4):
        r = results[rank]
        assert r.error[0] == 0, r.error
        assert r.mismatches == 0
        assert r.same_across_ranks
        # every replayed launch since the last reset moved the epoch on
        assert r.resets == STRESS_LAUNCHES // RESET_EVERY
        eager = 0 if r.resets else 2 * len(STRESS_WIDTHS)
        assert r.epoch == STRESS_LAUNCHES % RESET_EVERY + eager


def test_full_layer_metadata_keeps_cache_mappings_separate():
    """Speculative rows retain each cache's block ids and keep padding inert."""
    from vllm.models.minimax_m3.amd.specialized_decode.mono.layer.step import (
        expand_rows,
    )

    main = torch.arange(9, device="cuda", dtype=torch.int32).view(3, 3)
    index = torch.arange(15, device="cuda", dtype=torch.int32).view(3, 5) + 100
    lengths = torch.tensor([132, 260, 388], device="cuda", dtype=torch.int32)
    slots = torch.tensor(list(range(128, 140)) + [-1] * 4, device="cuda")
    index_slots = torch.tensor(list(range(768, 780)) + [-1] * 4, device="cuda")
    out_main = torch.empty(16, 256, device="cuda", dtype=torch.int32)
    out_index = torch.empty_like(out_main)
    out_lengths = torch.empty(16, device="cuda", dtype=torch.int32)
    out_slots, out_index_slots = torch.empty_like(slots), torch.empty_like(slots)
    batch = torch.empty_like(out_lengths)
    expand_rows[(16, 1)](
        main,
        index,
        lengths,
        slots,
        index_slots,
        out_main,
        out_index,
        out_lengths,
        out_slots,
        out_index_slots,
        batch,
        MAIN_STRIDE=3,
        INDEX_STRIDE=5,
        WIDTH=256,
        Q=4,
        REQUESTS=3,
        BLOCK=256,
    )
    request = torch.arange(16, device="cuda").div(4, rounding_mode="floor").clamp(max=2)
    assert torch.equal(out_main[:, :3], main[request])
    assert torch.equal(out_index[:, :5], index[request])
    assert not out_main[:, 3:].any() and not out_index[:, 5:].any()
    expected_lengths = (
        lengths.repeat_interleave(4) - 3 + torch.arange(12, device="cuda") % 4
    )
    assert torch.equal(out_lengths[:12], expected_lengths)
    assert torch.equal(out_slots[:12], slots[:12] // 128 * 256 + slots[:12] % 128)
    assert torch.equal(out_index_slots, index_slots)
    assert (out_lengths[12:] == 1).all() and (out_slots[12:] == -1).all()
    assert (batch[12:] == -1).all()


def _mxfp8(x):
    groups = x.float().unflatten(-1, (-1, 32))
    codes = (groups.abs().amax(-1) / 448).log2().ceil().add(127).clamp(1, 254)
    scale = torch.exp2(codes - 127)
    q = (groups / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (
        q.flatten(-2),
        codes.to(torch.int32),
        (q.float() * scale[..., None]).flatten(-2),
    )


def _plain_expert(weight, scales, expert):
    """Invert AITER's permutations before an independent torch matmul."""
    n, packed_k = weight.shape[1:]
    packed = (
        weight[expert]
        .view(torch.uint8)
        .view(n // 16, packed_k // 32, 2, 16, 16)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(n, packed_k)
    )
    cols = 2 * packed_k // 32
    codes = (
        scales.view(-1, n // 32, cols // 8, 4, 16, 2, 2)[expert]
        .permute(0, 5, 3, 1, 4, 2)
        .contiguous()
        .view(n, cols)
    )
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=weight.device,
        dtype=torch.float32,
    )
    indices = torch.stack((packed & 15, packed >> 4), -1).long()
    return (
        levels[indices].view(n, cols, 32) * torch.exp2(codes.float() - 127)[..., None]
    ).flatten(-2)


def _check_layer_post_reference(weights, s, padded, fuse_k1, *, blocks=1):
    """Sparse attention through MoE against independent Torch MXFP8 math.

    Uniform attention over constant values makes its exact result known.
    Distinct requests occupy interleaved K/V blocks with scalar cache scales;
    fixed routes isolate the projection, normalization, and expert arithmetic.
    Both narrow and wide task schedules must preserve that result.
    """
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.minimax_m3.amd.specialized_decode.mono.layer.kernels.post_attn import (  # noqa: E501
        K4_ABI,
        build_post_attn_kernel,
    )
    from vllm.models.minimax_m3.amd.specialized_decode.mono.layer.layout import (
        SCRATCH,
        SCRATCH_BYTES,
        sym_layout,
    )

    dev = weights.w13.device
    live = s - padded
    length = 16 if blocks == 1 else blocks * 128
    selected = min(blocks, 16)
    assert blocks == 1 or fuse_k1
    g = torch.Generator(device=dev).manual_seed(42)
    values = torch.randn(s, blocks, 128, device=dev, generator=g).to(
        torch.float8_e4m3fn
    )
    if blocks > 1:
        # Basis values identify selected blocks and keep split means exact in BF16.
        ids = (
            torch.arange(blocks, device=dev)[None, :]
            + 17 * torch.arange(s, device=dev)[:, None]
        ) % 128
        basis = torch.nn.functional.one_hot(ids, 128).float() * 2
        values = basis.to(torch.float8_e4m3fn)
    cache = torch.zeros(
        s * blocks, 2, 8, 128, 16, device=dev, dtype=torch.float8_e4m3fn
    )
    cache[:, 1].copy_(values.view(-1, 128)[:, None, :, None].expand(-1, 8, 128, 16))
    expected_values = cache[:, 1].clone()
    keys = cache.view(s * blocks * 16, 1, 8, 16, 16)
    vals = cache.view(s * blocks * 16, 1, 1, 128, 16)[8:]
    scale = torch.tensor([0.5], device=dev)
    table = torch.zeros(s, max(16, blocks), dtype=torch.int32, device=dev)
    table[:, :blocks] = torch.arange(s * blocks, device=dev).view(s, blocks)
    lengths = torch.full((s,), length, dtype=torch.int32, device=dev)
    query = torch.zeros(s, 2048, dtype=torch.bfloat16, device=dev)
    residual = torch.randn(s, H, device=dev, generator=g).bfloat16()
    output = torch.empty_like(residual)
    next_residual = torch.empty_like(residual)
    wo = torch.randn(H, 2048, device=dev, generator=g).to(torch.float8_e4m3fn)
    packed_wo = rocm_aiter_ops.shuffle_weight(wo)
    wo_scale = torch.full((H,), 2048**-0.5, device=dev)
    router = torch.zeros_like(weights.router, dtype=torch.bfloat16)
    bias = torch.zeros_like(weights.router_bias)
    bias[:4] = torch.tensor([4, 3, 2, 1], device=dev)
    scratch = torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=dev)
    sym = torch.zeros(sym_layout(1)["_bytes"], dtype=torch.uint8, device=dev)
    peers = torch.tensor([sym.data_ptr()], dtype=torch.int64, device=dev)
    batch_ids = torch.arange(s, dtype=torch.int32, device=dev)
    batch_ids[live:] = -1
    pointers = dict(
        h_in=residual.data_ptr(), q=query.data_ptr(), block_table=table.data_ptr(),
        seq_lens=lengths.data_ptr(), k_cache=keys.data_ptr(), v_cache=vals.data_ptr(),
        k_scale=scale.data_ptr(), v_scale=scale.data_ptr(), w_o=packed_wo.data_ptr(),
        s_o=wo_scale.data_ptr(), g_post=weights.post_norm.data_ptr(),
        w_gate=router.data_ptr(), bias=bias.data_ptr(), w13=weights.w13.data_ptr(),
        s13=weights.s13.data_ptr(), w2=weights.w2.data_ptr(), s2=weights.s2.data_ptr(),
        h_mid=next_residual.data_ptr(), ar_out=output.data_ptr(),
        scratch=scratch.data_ptr(), sym=sym.data_ptr(), peers=peers.data_ptr(),
        rank=0, layer=3, bt_width=table.shape[1], q_len=1, tl=0, k1_args=0,
        positions=0, slot_mapping=0, res=0, batch_ids=batch_ids.data_ptr(),
        index_block_table=0, index_bt_width=0, index_slot_mapping=0, ar_in=0,
        mode=1,
    )  # fmt: skip
    tokens_per_block = min(length, 128)
    total_keys = selected * tokens_per_block
    value_sum = values[:, -selected:].float().sum(1) * tokens_per_block
    if fuse_k1:
        from vllm.models.minimax_m3.amd.specialized_decode.mono.layer.kernels.pre_attn import (  # noqa: E501
            K1_ARGS,
        )
        from vllm.models.minimax_m3.amd.specialized_decode.mono.layer.kernels.pre_attn import (  # noqa: E501
            SCRATCH_BYTES as PRE_BYTES,
        )

        # Exact by construction: norm(.5 + .5) rounds to one; each projection
        # row selects one input, and identity RoPE leaves the normalized heads.
        hidden = torch.full_like(residual, 0.5)
        previous_residual = hidden.clone()
        residual.fill_(1)
        g_in = torch.zeros(H, dtype=torch.bfloat16, device=dev)
        g_q = torch.full((128,), -1, dtype=torch.bfloat16, device=dev)
        g_head = torch.zeros_like(g_q)
        qkv_weight = torch.zeros(2560, H, dtype=torch.bfloat16, device=dev)
        qkv_weight[torch.arange(2560), torch.arange(2560)] = 1
        qkv_packed = rocm_aiter_ops.shuffle_weight(qkv_weight.to(torch.float8_e4m3fn))
        qkv_scale = torch.ones(2560, device=dev)
        cos_sin = torch.cat((torch.ones(length, 32), torch.zeros(length, 32)), -1).to(
            device=dev, dtype=torch.bfloat16
        )
        positions = torch.full((s,), length - 1, dtype=torch.int64, device=dev)
        last_offset = (length - 1) % 128
        main_pages = table[:, blocks - 1].long()
        main_slots = main_pages * 256 + last_offset
        index_table = table.clone()
        index_table[:, :blocks] = table[:, :blocks].flip(1)
        index_slots = index_table[:, blocks - 1].long() * 128 + 31
        main_slots[live:] = -1
        index_slots[live:] = -1
        index_cache = torch.full(
            (s * blocks * 128, 128), -7, dtype=torch.float8_e4m3fn, device=dev
        )
        if blocks > 1:
            # For q=1, each key's dot product is its logical block id exactly.
            block_ids = torch.arange(blocks, device=dev)[:, None]
            dimensions = torch.arange(128, device=dev)[None, :]
            index_keys = (block_ids // 128 + (dimensions < block_ids % 128)).to(
                torch.float8_e4m3fn
            )
            index_cache.view(s, blocks, 128, 128).copy_(
                index_keys.flip(0)[None, :, None, :].expand(s, blocks, 128, 128)
            )
        expected_index = index_cache.clone()
        expected_index[index_slots[:live]] = 1
        for row in range(live):
            expected_values[main_pages[row], last_offset // 16, :, last_offset % 16] = 2
        iq = torch.empty(s, 128, dtype=torch.bfloat16, device=dev)
        pre = torch.zeros(PRE_BYTES, dtype=torch.uint8, device=dev)
        k1_values = dict(
            ar=0, g_in=g_in.data_ptr(), w_qkv=qkv_packed.data_ptr(),
            s_qkv=qkv_scale.data_ptr(), g_q=g_q.data_ptr(), g_k=g_head.data_ptr(),
            g_iq=g_head.data_ptr(), g_ik=g_head.data_ptr(), cos_sin=cos_sin.data_ptr(),
            index_cache=index_cache.data_ptr(), iq_out=iq.data_ptr(),
            scratch=pre.data_ptr(),
        )  # fmt: skip
        k1 = torch.tensor(
            [k1_values[n] for n in K1_ARGS], dtype=torch.int64, device=dev
        )
        pointers.update(
            k1_args=k1.data_ptr(),
            positions=positions.data_ptr(),
            slot_mapping=main_slots.data_ptr(),
            res=previous_residual.data_ptr(),
            index_block_table=index_table.data_ptr(),
            index_bt_width=index_table.shape[1],
            index_slot_mapping=index_slots.data_ptr(),
            ar_in=hidden.data_ptr(),
        )
        # Uniform attention over the highest-scoring blocks; one V is replaced.
        value_sum[:live] += 2 - values[:live, -1].float()
    # Uniform probabilities quantize to 448. Keep the FP32 PV scale step:
    # cancelling 448 algebraically changes BF16 midpoint rounding.
    attention = (
        ((value_sum * 448) * (scale * (1.0 / 448)) / total_keys)
        .repeat(1, 16)
        .bfloat16()
    )
    _writable_flydsl_cache()
    launch = build_post_attn_kernel(
        1,
        128**-0.5,
        ffn.EPS,
        2.0,
        1.0,
        7.0,
        0,
        1,
        tokens=s,
        debug=True,
        fuse_k1=fuse_k1,
    )
    compiled = flyc.compile(launch, *K4_ABI.zeros())
    compiled(*K4_ABI.pack(pointers), torch.cuda.current_stream())

    if blocks > 1:
        from vllm.models.minimax_m3.amd.specialized_decode.mono.layer.layout import (
            SPARSE_TABLE_WORDS,
        )

        start = SCRATCH["sparse_table"]
        selected_pages = (
            scratch[start : start + s * SPARSE_TABLE_WORDS * 4]
            .view(torch.int32)
            .view(s, SPARSE_TABLE_WORDS)[:, :128]
        )
        expected_pages = (
            table[:, blocks - selected : blocks, None] * 16
            + torch.arange(8, device=dev)
        ).flatten(1)
        assert torch.equal(selected_pages.sort(1).values, expected_pages.sort(1).values)

    if fuse_k1:
        assert torch.equal(query[:live], torch.zeros_like(query[:live]))
        assert torch.equal(iq[:live], torch.ones_like(iq[:live]))
        assert torch.equal(
            index_cache.view(torch.uint8), expected_index.view(torch.uint8)
        )
        for row in range(live):
            page = main_pages[row] * 16 + last_offset // 16
            assert (keys[page, 0, :, last_offset % 16, :].float() == 2).all()
        assert cache[:, 0].view(torch.uint8).count_nonzero().item() == live * 128
        assert torch.equal(
            cache[:, 1].view(torch.uint8), expected_values.view(torch.uint8)
        )
    attention = attention.float()
    a_scale = attention.abs().amax(-1, keepdim=True) / 448
    aq = (attention / a_scale).to(torch.float8_e4m3fn).float()
    part = ((aq @ wo.float().T) * a_scale * wo_scale).bfloat16()
    x, want_residual = gemma_fused_add_rmsnorm(
        part, residual, weights.post_norm, ffn.EPS
    )
    # ATOM sums the route-weighted experts in FP32 before rounding this TP shard.
    want = torch.zeros(s, H, dtype=torch.float32, device=dev)
    for expert in (0, 1, 2, 3, ffn.SHARED):
        upgate = _mxfp8(x)[2] @ _plain_expert(weights.w13, weights.s13, expert).T
        gate, up = upgate.chunk(2, -1)
        gate, up = gate.clamp(max=ffn.LIMIT), up.clamp(-ffn.LIMIT, ffn.LIMIT)
        mid = (gate * (ffn.ALPHA * gate).sigmoid() * (up + ffn.BETA)).bfloat16()
        term = _mxfp8(mid)[2] @ _plain_expert(weights.w2, weights.s2, expert).T
        want += term * (1.0 if expert == ffn.SHARED else 0.5)
    cos, rel = _metrics(output[:live], want[:live].bfloat16())
    residual_cos, residual_rel = _metrics(next_residual[:live], want_residual[:live])
    diagnostic = scratch[SCRATCH["diag"] :].view(torch.int32).view(256, 8)
    assert not diagnostic[:, 0].any(), diagnostic[diagnostic[:, 0] != 0]
    assert cos > 0.9999 and rel < 1e-2, (cos, rel)
    assert residual_cos > 0.9999 and residual_rel < 1e-2


@pytest.mark.parametrize(
    "s,padded",
    [(1, 0), (4, 0), (8, 0), (12, 0), (16, 0), (4, 1), (8, 1), (12, 1), (16, 1)],
)
@pytest.mark.parametrize("fuse_k1", [False, True])
def test_layer_post_matches_mxfp8_reference(weights, s, padded, fuse_k1):
    _check_layer_post_reference(weights, s, padded, fuse_k1)


@pytest.mark.parametrize("rows,blocks", [(4, 32), (4, 544), (16, 544)])
def test_layer_index_selection_uses_its_own_block_table(weights, rows, blocks):
    """Short and long selectors use the independent index cache mapping."""
    _check_layer_post_reference(weights, rows, 0, True, blocks=blocks)


def _layer_end_rank(rank, port, release_rank, results):
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.minimax_m3.amd.specialized_decode.mono.layer.step import (
        begin_fence,
        check_diagnostics,
    )

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4
    )
    device = torch.device("cuda", rank)
    peer = PeerBuffer(256 + 4096 + 64, dist.group.WORLD, rank, 4, device)
    peer.bytes.zero_()
    addresses = peer.addresses + 256
    epoch = torch.zeros(1, dtype=torch.int64, device=device)
    diag = torch.zeros(256, 8, dtype=torch.int32, device=device)
    begin_fence(
        addresses.data_ptr(),
        rank,
        epoch.data_ptr(),
        4096,
        4,
        mode=0,
        stream=torch.cuda.current_stream(),
    )
    check_diagnostics[(1,)](diag, addresses, rank, epoch, TP=4)
    torch.accelerator.synchronize()
    dist.barrier()
    # Rank 2 cannot publish its late diagnostic until rank 0 proves it has
    # reached the fence and cannot pass it. No timing assumption orders this.
    if rank == 2:
        assert release_rank.wait(60)
        diag[0, 0] = 1
    check_diagnostics[(1,)](diag, addresses, rank, epoch, TP=4)
    before, after = torch.cuda.Event(), torch.cuda.Event()
    before.record()
    begin_fence(
        addresses.data_ptr(),
        rank,
        epoch.data_ptr(),
        4096,
        4,
        stream=torch.cuda.current_stream(),
    )
    after.record()
    if rank == 0:
        before.synchronize()
        assert not after.query(), "sampling could overtake a rank's diagnostics"
        release_rank.set()
    after.synchronize()
    results[rank] = peer.bytes[:16].view(torch.int32).tolist()
    dist.barrier()
    peer.close()
    dist.destroy_process_group()


@pytest.mark.distributed(num_gpus=4)
def test_layer_step_end_fence_orders_remote_diagnostics():
    """A rank that finishes early cannot sample before a late peer's fault
    becomes visible through the real TP4 IPC buffers.
    """
    from vllm.utils.network_utils import get_open_port

    with mp.get_context("spawn").Manager() as manager:
        results = manager.dict()
        release_rank = manager.Event()
        ranks = mp.start_processes(
            _layer_end_rank,
            args=(get_open_port(), release_rank, results),
            nprocs=4,
            join=False,
        )
        deadline = time.monotonic() + 120
        while not ranks.join(timeout=10):
            if time.monotonic() > deadline:
                for process in ranks.processes:
                    process.kill()
                pytest.fail("TP4 step-end fence did not finish")
        assert len(results) == 4
        for words in results.values():
            assert words == [1, 2, 0, 2]
